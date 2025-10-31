from __future__ import annotations

"""
Deterministic Phase-1 Router (NL improvements)

Key upgrades:
- Base-table detection from natural nouns in the NL (plural/singular + a few synonyms)
- Bare dimensions like "by name" auto-map to the chosen base table if unique
- Extra date phrases: "this month/week/year", "last month/week/year", "today", "yesterday"

Public entry remains:
    DeterministicRouter().route(nl) -> LogicalQueryRequest
"""

import re
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import load_all_configs
from .models import (
    LogicalQueryRequest,
    Filter,
    OrderBy,
    Operator,
    Direction,
    Resolution,
)

# --------------------------- Errors --------------------------- #

@dataclass
class RouterError(Exception):
    code: str
    reason: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.reason}"


# --------------------------- Utilities --------------------------- #

_WORD = re.compile(r"[a-z0-9_]+")
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def _now_date() -> date:
    return datetime.now().date()


def _tokenize_lower(s: str) -> List[str]:
    return _WORD.findall(s.lower())


# --------------------------- Geo disambiguation helpers --------------------------- #

_US_STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware",
    "florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
    "louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi",
    "missouri","montana","nebraska","nevada","new hampshire","new jersey","new mexico",
    "new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania",
    "rhode island","south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming","district of columbia",
    "washington dc","dc",
}
_US_STATE_ABBR = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks","ky",
    "la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd",
    "oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy","dc"
}

def _place_looks_like_state(value: str) -> bool:
    v = value.strip().lower()
    if v in _US_STATE_NAMES or v in _US_STATE_ABBR:
        return True
    # two-letter uppercase like "TX", "CA"
    if len(value.strip()) == 2 and value.isupper():
        return True
    return False

def _choose_geo_column(base_table: str, value: str, s: Surfaces) -> Optional[str]:
    cols = set(s.fields_by_table.get(base_table) or [])
    # If it looks like a US state, prefer state
    if _place_looks_like_state(value) and "state" in cols:
        return "state"
    # Otherwise prefer city if present; else state; else region
    if "city" in cols:
        return "city"
    if "state" in cols:
        return "state"
    if "region" in cols:
        return "region"
    return None


# --------------------------- Surfaces --------------------------- #

@dataclass
class Surfaces:
    tables: List[str]
    # dimensions/metrics surfaces
    dims: Dict[str, Dict[str, Any]]             # "table.col" -> meta
    mets: Dict[str, Dict[str, Any]]             # "table.count_rows", "table.sum_col", ...
    dims_by_table: Dict[str, List[str]]
    datetime_dims_by_table: Dict[str, List[str]]  # "table.col" for datetime dims
    bare_dim_index: Dict[str, List[str]]          # bare col -> ["table.col", ...]
    # fields & pks for projections
    fields_by_table: Dict[str, List[str]]
    pks_by_table: Dict[str, List[str]]
    forbidden_fields: set[str]

def _extract_forbidden_fields(forbidden: Any) -> set[str]:
    out: set[str] = set()
    if forbidden is None:
        return out
    if isinstance(forbidden, list):
        out.update(str(x) for x in forbidden if isinstance(x, str))
        return out
    if isinstance(forbidden, dict):
        if isinstance(forbidden.get("fields"), list):
            out.update(str(x) for x in forbidden["fields"] if isinstance(x, str))
        for k in list(forbidden.keys()):
            if isinstance(k, str) and "." in k:
                out.add(k)
        return out
    return out

def _build_surfaces(config_dir: Path | None = None) -> Surfaces:
    bundle = load_all_configs((config_dir or Path("src/orchestrator/config")).resolve())

    tables = list((bundle.entities or {}).keys())
    dims: Dict[str, Dict[str, Any]] = bundle.dimensions or {}
    mets: Dict[str, Dict[str, Any]] = bundle.metrics or {}

    dims_by_table: Dict[str, List[str]] = {}
    dt_dims_by_table: Dict[str, List[str]] = {}
    bare_dim_index: Dict[str, List[str]] = {}

    for key, meta in dims.items():
        if "." not in key:
            continue
        t, c = key.split(".", 1)
        dims_by_table.setdefault(t, []).append(key)
        if isinstance(meta, dict) and meta.get("type") == "datetime":
            dt_dims_by_table.setdefault(t, []).append(key)
        bare_dim_index.setdefault(c, []).append(key)

    fields_by_table: Dict[str, List[str]] = {}
    for t, cols in (bundle.fields or {}).items():
        if isinstance(cols, dict):
            fields_by_table[t] = list(cols.keys())

    pks_by_table: Dict[str, List[str]] = {}
    for t, meta in (bundle.entities or {}).items():
        if isinstance(meta, dict) and isinstance(meta.get("pk"), list):
            pks_by_table[t] = list(meta["pk"]) or []
        else:
            pks_by_table[t] = []

    forbidden_fields = _extract_forbidden_fields(bundle.forbidden)

    return Surfaces(
        tables=tables,
        dims=dims,
        mets=mets,
        dims_by_table=dims_by_table,
        datetime_dims_by_table=dt_dims_by_table,
        bare_dim_index=bare_dim_index,
        fields_by_table=fields_by_table,
        pks_by_table=pks_by_table,
        forbidden_fields=forbidden_fields,
    )

# --------------------------- Date parsing --------------------------- #

@dataclass
class DateFilter:
    start: date
    end: date
    grain: Optional[Resolution]


def _parse_relative_range(n: int, unit: str) -> DateFilter:
    today = _now_date()
    unit = unit.rstrip("s")
    if unit == "day":
        start = today - timedelta(days=n)
        grain = Resolution.DAY
    elif unit == "week":
        start = today - timedelta(weeks=n)
        grain = Resolution.WEEK
    elif unit == "month":
        start = today - timedelta(days=30 * n)
        grain = Resolution.MONTH
    elif unit == "year":
        start = date(today.year - n, today.month, today.day)
        grain = Resolution.YEAR
    else:
        start = today
        grain = None
    return DateFilter(start=start, end=today, grain=grain)


def _month_year_range(month_name: str, year: Optional[int]) -> Optional[DateFilter]:
    m = _MONTHS.get(month_name.lower())
    if not m:
        return None
    y = year or _now_date().year
    start = date(y, m, 1)
    last_day = calendar.monthrange(y, m)[1]
    end = date(y, m, last_day)
    return DateFilter(start=start, end=end, grain=Resolution.MONTH)


def detect_date_phrase(nl: str) -> Optional[DateFilter]:
    s = nl.lower()

    # this <unit>
    m = re.search(r"\bthis\s+(day|week|month|year)\b", s)
    if m:
        unit = m.group(1)
        today = _now_date()
        if unit == "day":
            return DateFilter(today, today, Resolution.DAY)
        if unit == "week":
            # ISO week: Monday as start
            start = today - timedelta(days=today.weekday())
            return DateFilter(start, today, Resolution.WEEK)
        if unit == "month":
            start = today.replace(day=1)
            return DateFilter(start, today, Resolution.MONTH)
        if unit == "year":
            start = date(today.year, 1, 1)
            return DateFilter(start, today, Resolution.YEAR)

    # last <unit> (calendar previous)
    m = re.search(r"\blast\s+(week|month|year)\b", s)
    if m:
        unit = m.group(1)
        today = _now_date()
        if unit == "week":
            # previous ISO week
            end = today - timedelta(days=today.weekday() + 1)
            start = end - timedelta(days=6)
            return DateFilter(start, end, Resolution.WEEK)
        if unit == "month":
            y, mon = today.year, today.month
            mon -= 1
            if mon == 0:
                mon, y = 12, y - 1
            start = date(y, mon, 1)
            end = date(y, mon, calendar.monthrange(y, mon)[1])
            return DateFilter(start, end, Resolution.MONTH)
        if unit == "year":
            y = today.year - 1
            return DateFilter(date(y, 1, 1), date(y, 12, 31), Resolution.YEAR)

    # last N units / in the last N units
    m = re.search(r"(?:in\s+the\s+)?last\s+(\d{1,3})\s+(days|day|weeks|week|months|month|years|year)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        return _parse_relative_range(n, unit)

    # explicit month year: e.g., October 2025
    m2 = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", s)
    if m2:
        return _month_year_range(m2.group(1), int(m2.group(2)))

    # month alone (assume current year)
    m3 = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", s)
    if m3:
        return _month_year_range(m3.group(1), None)

    # today / yesterday
    if re.search(r"\btoday\b", s):
        t = _now_date()
        return DateFilter(t, t, Resolution.DAY)
    if re.search(r"\byesterday\b", s):
        t = _now_date() - timedelta(days=1)
        return DateFilter(t, t, Resolution.DAY)

    return None


# --------------------------- Core routing --------------------------- #

@dataclass
class ParsedIntent:
    base_table: Optional[str]
    dim_names: List[str]           # e.g., "table.col" or bare names
    met_names: List[str]           # metric keys or hints ("count", "sum:col")
    order_field: Optional[str]
    order_dir: Optional[Direction]
    limit: Optional[int]
    date_filter: Optional[DateFilter]
    is_detail_hint: bool = False

def parse_intent(nl: str) -> ParsedIntent:
    s = nl.lower()
    limit = None
    order_dir: Optional[Direction] = None

    if m := re.search(r"\blimit\s+(\d{1,5})\b", s):
        limit = int(m.group(1))
    if m := re.search(r"\btop\s+(\d{1,5})\b", s):
        limit = int(m.group(1)); order_dir = Direction.DESC
    if m := re.search(r"\bbottom\s+(\d{1,5})\b", s):
        limit = int(m.group(1)); order_dir = Direction.ASC
    if m := re.search(r"\b(latest|recent|newest)\s+(\d{1,5})\b", s):
        limit = int(m.group(2))

    order_field, findir = _parse_order_clause(s)
    order_dir = findir or order_dir

    df = detect_date_phrase(s)

    dim_names: List[str] = []
    for m in re.finditer(r"\b(?:by|group\s+by)\s+([a-z0-9_\.]+(?:\s+and\s+[a-z0-9_\.]+)*)", s):
        clause = m.group(1)
        for part in re.split(r"\s+and\s+|,\s*", clause):
            name = part.strip().replace(" ", "_")
            if name:
                dim_names.append(name)

    met_names: List[str] = []
    if re.search(r"\bcount\b", s):
        met_names.append("count")
    for m in re.finditer(r"\bsum\s+([a-z0-9_\.]+)", s):
        met_names.append(f"sum:{m.group(1)}")
    for m in re.finditer(r"\bavg(?:erage)?\s+([a-z0-9_\.]+)", s):
        met_names.append(f"avg:{m.group(1)}")

    is_detail_hint = bool(re.search(r"\b(show|list|latest|recent|newest|most\s+recent)\b", s))

    return ParsedIntent(
        base_table=None,
        dim_names=dim_names,
        met_names=met_names,
        order_field=order_field,
        order_dir=order_dir,
        limit=limit,
        date_filter=df,
        is_detail_hint=is_detail_hint,
    )


# --------------------------- Resolution helpers --------------------------- #

# simple synonyms for table nouns in NL → canonical table name
_SYN_TABLE: Dict[str, str] = {
    # common pairs in your schema
    "dealer": "dealers", "dealers": "dealers",
    "advertiser": "advertisers", "advertisers": "advertisers",
    "host": "hosts", "hosts": "hosts",
    "license": "licenses", "licenses": "licenses",
    "city": "cities", "cities": "cities",
}


def _detect_base_table_from_text(nl: str, surfaces: Surfaces) -> Optional[str]:
    tokens = _tokenize_lower(nl)
    # include singular versions of actual table names
    singular_map = {t.rstrip('s'): t for t in surfaces.tables if t.endswith('s')}

    # candidates in order of appearance
    order: List[str] = []
    seen: set[str] = set()
    for i, tok in enumerate(tokens):
        # synonym map first
        if tok in _SYN_TABLE:
            t = _SYN_TABLE[tok]
            if t in surfaces.tables and t not in seen:
                order.append(t); seen.add(t)
                continue
        # direct match
        if tok in surfaces.tables and tok not in seen:
            order.append(tok); seen.add(tok)
            continue
        # singular → plural map
        if tok in singular_map:
            t = singular_map[tok]
            if t in surfaces.tables and t not in seen:
                order.append(t); seen.add(t)
    return order[0] if order else None


def _resolve_dimensions(names: List[str], surfaces: Surfaces, base_table: Optional[str]) -> Tuple[List[str], Optional[str]]:
    resolved: List[str] = []
    inferred_table: Optional[str] = None

    for name in names:
        # Qualified, exact dimension key
        if "." in name and name in surfaces.dims:
            resolved.append(name)
            t = name.split(".", 1)[0]
            inferred_table = inferred_table or t
            continue

        # Bare name lookup via dimension index
        matches = surfaces.bare_dim_index.get(name, [])
        if base_table:
            matches = [k for k in matches if k.startswith(base_table + ".")]
        if len(matches) == 1:
            resolved.append(matches[0])
            inferred_table = inferred_table or matches[0].split(".", 1)[0]
            continue
        if len(matches) > 1:
            # ambiguous; skip for Phase-1
            continue

        # NEW: natural-to-column fallback (e.g., "name" -> "dealer_name")
        if base_table:
            col = _natural_to_column(base_table, name, surfaces)
            if col:
                key = f"{base_table}.{col}"
                if key in surfaces.dims:
                    resolved.append(key)
                    inferred_table = inferred_table or base_table

    return resolved, inferred_table


def _resolve_metrics(hints: List[str], surfaces: Surfaces, base_table: Optional[str]) -> Tuple[List[str], Optional[str]]:
    resolved: List[str] = []
    inferred_table: Optional[str] = None

    # exact keys
    for h in list(hints):
        if "." in h and h in surfaces.mets:
            resolved.append(h)
            inferred_table = inferred_table or h.split(".", 1)[0]

    # sum/avg of <col>
    for h in hints:
        if h.startswith("sum:"):
            col = h.split(":", 1)[1]
            if base_table and f"{base_table}.sum_{col}" in surfaces.mets:
                resolved.append(f"{base_table}.sum_{col}"); inferred_table = inferred_table or base_table
        if h.startswith("avg:") or h.startswith("average:"):
            col = h.split(":", 1)[1]
            if base_table and f"{base_table}.avg_{col}" in surfaces.mets:
                resolved.append(f"{base_table}.avg_{col}"); inferred_table = inferred_table or base_table

    # fallback count
    if not resolved:
        if base_table and f"{base_table}.count_rows" in surfaces.mets:
            resolved.append(f"{base_table}.count_rows"); inferred_table = inferred_table or base_table
        else:
            # try to infer a table from dims later; otherwise pick first table (avoid: will be corrected by base detection)
            pass

    return resolved, inferred_table


def _pick_base_table(nl: str, intent: ParsedIntent, surfaces: Surfaces, from_dims_table: Optional[str], from_mets_table: Optional[str]) -> str:
    # 1) explicit mention in text
    mention = _detect_base_table_from_text(nl, surfaces)
    if mention:
        return mention
    # 2) inferred from dims/metrics
    if from_dims_table:
        return from_dims_table
    if from_mets_table:
        return from_mets_table
    # 3) no idea
    raise RouterError("AMBIGUOUS_REQUEST", "Please mention a base table (e.g., 'dealers', 'advertisers').")


def _default_datetime_dim(base_table: str, surfaces: Surfaces) -> Optional[str]:
    arr = surfaces.datetime_dims_by_table.get(base_table) or []
    return arr[0] if arr else None


def _apply_date_filter(base_table: str, df: DateFilter, surfaces: Surfaces, filters: List[Filter]) -> None:
    dim_key = _default_datetime_dim(base_table, surfaces)
    if not dim_key:
        return
    filters.append(Filter(field=dim_key, op=Operator.BETWEEN, value=[df.start.isoformat(), df.end.isoformat()]))


def _parse_order_clause(s: str) -> Tuple[Optional[str], Optional[Direction]]:
    # Capture 'order by <field>' optionally followed by asc/desc; stop at end or 'limit N'
    m = re.search(r"order\s+by\s+([a-z0-9_\. ]+?)(?:\s+(asc|desc))?(?=$|\s+limit\s+\d+)", s)
    if not m:
        # Fallback (no limit present)
        m = re.search(r"order\s+by\s+([a-z0-9_\. ]+)(?:\s+(asc|desc))?(?:$|\b)", s)
    if not m:
        return None, None
    field = m.group(1).strip()
    dir_ = Direction(m.group(2)) if m.group(2) else None
    return field, dir_

_NAMEISH = re.compile(r"(^name$|_name$|^name_|\bname\b)")

def _natural_to_column(base_table: str, phrase: str, s: Surfaces) -> Optional[str]:
    phrase = phrase.strip().lower()
    cols = set(s.fields_by_table.get(base_table, []))

    direct_map = {
        "date created": ["date_created", "created_at", "created_on"],
        "created date": ["date_created", "created_at", "created_on"],
        "date updated": ["date_updated", "updated_at", "updated_on"],
        "updated date": ["date_updated", "updated_at", "updated_on"],
        "name": ["name"],
        "region": ["region"],
        "state": ["state", "province"],
        "city": ["city"],
        "status": ["status"],
    }
    if phrase in direct_map:
        for c in direct_map[phrase]:
            if c in cols:
                return c
        if phrase == "name":
            for c in cols:
                if _NAMEISH.search(c):
                    return c
        if phrase == "state":
            for alt in ["province", "region_state"]:
                if alt in cols:
                    return alt

    underscore = phrase.replace(" ", "_")
    if underscore in cols:
        return underscore

    if phrase == "name":
        for c in sorted(cols):
            if _NAMEISH.search(c):
                return c

    return None

def _default_projections(base_table: str, s: Surfaces, max_cols: int = 5) -> List[str]:
    cols = s.fields_by_table.get(base_table, []) or []
    colset = set(cols)
    picks: List[str] = []

    def add(c: str):
        qc = f"{base_table}.{c}"
        if c in colset and qc not in s.forbidden_fields and c not in picks and len(picks) < max_cols:
            picks.append(c)

    # 1) declared PKs
    for c in s.pks_by_table.get(base_table, []):
        add(c)
    # 2) id fallbacks
    for c in ["id", f"{base_table.rstrip('s')}_id"]:
        add(c)
    # 3) human-friendly fields
    for c in ["name", "title", "status", "region", "state", "city", "date_created", "date_updated"]:
        if c == "name" and c not in colset:
            nameish = None
            for col in cols:
                if _NAMEISH.search(col):
                    nameish = col; break
            if nameish:
                add(nameish); continue
        add(c)

    return [f"{base_table}.{c}" for c in picks]

def _trim_value_span(txt: str) -> str:
    # stop at common delimiters so we don't eat "order by", "group by", etc.
    guards = [
        r"\border\s+by\b", r"\bgroup\s+by\b", r"\blimit\s+\d+\b",
        r"\bin\s+the\s+last\b", r"\blast\s+\d+\b", r"\bthis\s+\w+\b",
        r"\bby\b", r",", r"\band\b",
    ]
    cut = len(txt)
    for g in guards:
        m = re.search(g, txt, flags=re.IGNORECASE)
        if m:
            cut = min(cut, m.start())
    return txt[:cut].strip()

def _extract_value_filters(nl: str, base_table: str, s: Surfaces) -> List[Filter]:
    """
    Heuristics:
    - Implicit status keywords: active/inactive/enabled/disabled/pending/... → status = <kw>
    - Explicit pairs: (status|region|state|city|name) <value>
    - "in <place>" (but NOT "in the last ..."):
        prefer state → city → region (first available column on base_table)
    - Uses exact-string equality (safe). LIKE / IN / negations are Phase-2.

    NOTE: We only add filters for columns that exist on `base_table`.
    """
    filters: List[Filter] = []
    used: set[tuple[str, str]] = set()  # (col,value) to avoid dupes

    text = nl

    def has_col(col: str) -> bool:
        return col in (s.fields_by_table.get(base_table) or [])

    # ---------- 0) Implicit status keywords (no need to say "status active") ----------
    status_kw_map = {
        "active": "active",
        "inactive": "inactive",
        "enabled": "enabled",
        "disabled": "disabled",
        "pending": "pending",
        "approved": "approved",
        "denied": "denied",
        "suspended": "suspended",
    }
    if has_col("status"):
        for kw, val in status_kw_map.items():
            if re.search(rf"\b{kw}\b", text, flags=re.IGNORECASE):
                key = ("status", val)
                if key not in used:
                    filters.append(Filter(field=f"{base_table}.status", op=Operator.EQ, value=val))
                    used.add(key)
                break  # take the first matched status keyword only

    # ---------- 1) Explicit attribute/value pairs ----------
    def _trim_value_span(txt: str) -> str:
        guards = [
            r"\border\s+by\b", r"\bgroup\s+by\b", r"\blimit\s+\d+\b",
            r"\bin\s+the\s+last\b", r"\blast\s+\d+\b", r"\bthis\s+\w+\b",
            r"\bby\b", r",", r"\band\b",
        ]
        cut = len(txt)
        for g in guards:
            m = re.search(g, txt, flags=re.IGNORECASE)
            if m:
                cut = min(cut, m.start())
        return txt[:cut].strip()

    attr_tokens = ["status", "region", "state", "city", "name"]
    for tok in attr_tokens:
        m = re.search(
            rf"\b{tok}\b\s*(?:=|is|equals)?\s*([A-Za-z][A-Za-z0-9 _\-]{{1,80}})",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        raw = _trim_value_span(m.group(1))
        if not raw:
            continue

        col = _natural_to_column(base_table, tok, s)
        if not col or not has_col(col):
            continue

        key = (col, raw)
        if key in used:
            continue
        used.add(key)
        filters.append(Filter(field=f"{base_table}.{col}", op=Operator.EQ, value=raw))

    # ---------- 2) "in <place>"  (but NOT "in the last ...") ----------
    m = re.search(r"\bin\s+(?!the\s+last\b)([A-Za-z][A-Za-z0-9 _\-,]{1,80})", text, flags=re.IGNORECASE)
    if m:
        raw = _trim_value_span(m.group(1))
        if raw:
            pref_col = _choose_geo_column(base_table, raw, s)
            if pref_col and has_col(pref_col):
                col = _natural_to_column(base_table, pref_col, s) or pref_col
                key = (col, raw)
                if key not in used:
                    used.add(key)
                    filters.append(Filter(field=f"{base_table}.{col}", op=Operator.EQ, value=raw))

    return filters


# --------------------------- Public API --------------------------- #

def route(nl: str, *, config_dir: Path | None = None) -> LogicalQueryRequest:
    return DeterministicRouter(config_dir=config_dir).route(nl)


class DeterministicRouter:
    def __init__(self, config_dir: Path | None = None) -> None:
        self.surfaces = _build_surfaces(config_dir)

    def route(self, nl: str, *, dialect: Optional[str] = None, tz: Optional[str] = None,
            locale: Optional[str] = None, tenant: Optional[str] = None) -> LogicalQueryRequest:
        if not nl or not nl.strip():
            raise RouterError("MISSING_PARAMETER", "Empty natural-language query.")

        s = nl.lower()
        intent = parse_intent(nl)

        base_table_hint = _detect_base_table_from_text(nl, self.surfaces)

        dims, dims_table = _resolve_dimensions(intent.dim_names, self.surfaces, base_table_hint)
        mets, mets_table = _resolve_metrics(intent.met_names, self.surfaces, base_table_hint)

        aggregate_cues = bool(dims) or bool(intent.met_names) or bool(re.search(r"\b(top|bottom|group\s+by)\b", s))
        detail_cues = intent.is_detail_hint or bool(intent.order_field and not aggregate_cues)
        is_detail = detail_cues and not aggregate_cues

        base_table = _pick_base_table(nl, intent, self.surfaces, dims_table, mets_table or base_table_hint)

        # ORDER BY
        order_by: List[OrderBy] = []
        if intent.order_field:
            col = _natural_to_column(base_table, intent.order_field, self.surfaces) or intent.order_field
            candidate = None
            if base_table and f"{base_table}.{col}" in self.surfaces.mets:
                candidate = f"{base_table}.{col}"
            elif base_table and f"{base_table}.{col}" in self.surfaces.dims:
                candidate = f"{base_table}.{col}"
            elif base_table and col in (self.surfaces.fields_by_table.get(base_table) or []):
                candidate = f"{base_table}.{col}"
            direction = intent.order_dir or (Direction.DESC if re.search(r"\b(latest|recent|newest|most\s+recent)\b", s) else Direction.DESC)
            order_by.append(OrderBy(field=candidate or col, direction=direction))
        else:
            if is_detail and re.search(r"\b(latest|recent|newest|most\s+recent)\b", s):
                for pref in ("date_created", "date_updated"):
                    if pref in (self.surfaces.fields_by_table.get(base_table) or []):
                        order_by.append(OrderBy(field=f"{base_table}.{pref}", direction=Direction.DESC))
                        break

        filters: List[Filter] = []
        if intent.date_filter:
            _apply_date_filter(base_table, intent.date_filter, self.surfaces, filters)
        # If user added value filters and there are no aggregate cues, treat as detail
        
        # Value filters from natural phrases (status/region/state/city/name/in <place>)
        vfilters = _extract_value_filters(nl, base_table, self.surfaces)
        if vfilters:
            filters.extend(vfilters)
    
        if not is_detail and not aggregate_cues:
            peek_v = _extract_value_filters(nl, base_table, self.surfaces)
            if peek_v:
                is_detail = True
        
        if is_detail:
            projections = _default_projections(base_table, self.surfaces)

            #  Ensure the ORDER BY column is included in projections if it's a base-table field
            if order_by:
                ob_field = order_by[0].field
                if isinstance(ob_field, str) and ob_field and ob_field.startswith(base_table + "."):
                    if ob_field not in projections:
                        projections = [ob_field] + projections

            return LogicalQueryRequest(
                base_table=base_table,
                dimensions=[],
                metrics=[],
                projections=projections,
                filters=filters,
                order_by=order_by,
                limit=intent.limit,
                distinct=False,
                resolution=intent.date_filter.grain if intent.date_filter else None,
                dialect=dialect,
                tz=tz,
                locale=locale,
                tenant=tenant,
            )

        # Aggregate path: prefer base table for dims/mets
        fixed_dims: List[str] = []
        for d in dims:
            t, col = d.split(".", 1)
            if t == base_table:
                fixed_dims.append(d)
            else:
                candidates = [k for k in self.surfaces.bare_dim_index.get(col, []) if k.startswith(base_table + ".")]
                if len(candidates) == 1:
                    fixed_dims.append(candidates[0])
        dims = list(dict.fromkeys(fixed_dims))

        if not mets:
            if f"{base_table}.count_rows" in self.surfaces.mets:
                mets = [f"{base_table}.count_rows"]
        else:
            fixed_mets: List[str] = []
            for m in mets:
                t = m.split(".", 1)[0]
                if t == base_table:
                    fixed_mets.append(m)
                elif m.endswith(".count_rows") and f"{base_table}.count_rows" in self.surfaces.mets:
                    fixed_mets.append(f"{base_table}.count_rows")
                elif ".sum_" in m:
                    col = m.split(".sum_", 1)[1]
                    key = f"{base_table}.sum_{col}"
                    if key in self.surfaces.mets:
                        fixed_mets.append(key)
                elif ".avg_" in m:
                    col = m.split(".avg_", 1)[1]
                    key = f"{base_table}.avg_{col}"
                    if key in self.surfaces.mets:
                        fixed_mets.append(key)
            if fixed_mets:
                mets = fixed_mets

        # If user wrote "order by <dim>" for a top/bottom query, treat that <dim> as a grouping key,
        # but enforce deterministic ranking by the metric per policy.
        rank_token = re.search(r"\b(top|bottom)\s+\d+\b", s)
        dim_from_order: Optional[str] = None
        if intent.order_field:
            nat = _natural_to_column(base_table, intent.order_field, self.surfaces) or intent.order_field
            cand_dim = f"{base_table}.{nat}"
            if cand_dim in self.surfaces.dims and cand_dim not in dims:
                dim_from_order = cand_dim
                dims.append(cand_dim)

        order_by = list(order_by)  # copy any earlier decisions (rare in agg path)
        if rank_token and mets:
            direction = Direction.DESC if "top" in rank_token.group(1) else Direction.ASC
            order_by = [OrderBy(field=mets[0], direction=direction)]

        return LogicalQueryRequest(
            base_table=base_table,
            dimensions=dims,
            metrics=mets,
            projections=[],
            filters=filters,
            order_by=order_by,
            limit=intent.limit,
            distinct=False,
            resolution=intent.date_filter.grain if intent.date_filter else None,
            dialect=dialect,
            tz=tz,
            locale=locale,
            tenant=tenant,
        )
