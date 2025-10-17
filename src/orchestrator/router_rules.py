# src/orchestrator/router_rules.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Set

import sqlglot
from sqlglot import exp

from .date_normalizer import infer_time_range
from .models import Direction, LogicalQueryRequest, OrderBy

# -----------------------------------------------------------------------------
# Rank & limit patterns (accept "top 10", "top-10", "top10"; same for bottom)
# -----------------------------------------------------------------------------
_TOP_RE = re.compile(r"\btop\s*[-]?\s*(?P<n>\d+)\b", re.IGNORECASE)
_BOTTOM_RE = re.compile(r"\bbottom\s*[-]?\s*(?P<n>\d+)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\s+(?P<n>\d+)\b", re.IGNORECASE)

def _extract_rank_and_limit(nl_text: str) -> Tuple[Optional[int], Optional[str]]:
    m = _TOP_RE.search(nl_text)
    if m:
        return int(m.group("n")), "desc"
    m = _BOTTOM_RE.search(nl_text)
    if m:
        return int(m.group("n")), "asc"
    m = _LIMIT_RE.search(nl_text)
    if m:
        return int(m.group("n")), None
    return None, None

# -----------------------------------------------------------------------------
# BY / PER segment: capture "by dealer nebula", "per state", bounded by clause
# starters like in/on/between/where/top/bottom/limit or string end.
# -----------------------------------------------------------------------------
_BY_PER_RE = re.compile(
    r"\b(?:by|per)\s+(?P<phrase>[a-z0-9_. ]+?)(?=$|\s+(?:in|on|between|where|top|bottom|limit|for)\b)",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------
# Explicit filter patterns (deterministic; Phase 1)
# -----------------------------------------------------------------------------
_EQ_RE   = re.compile(r"(?P<f>[\w.]+)\s*(=|:| is )\s*(?P<val>[^,;]+)", re.IGNORECASE)
_NEQ_RE  = re.compile(r"(?P<f>[\w.]+)\s*(!=|<>| is not )\s*(?P<val>[^,;]+)", re.IGNORECASE)
_CMP_RE  = re.compile(r"(?P<f>[\w.]+)\s*(?P<op>>=|<=|>|<)\s*(?P<val>[^,;]+)", re.IGNORECASE)
_IN_RE   = re.compile(r"(?P<f>[\w.]+)\s+in\s*\((?P<vals>[^)]*)\)", re.IGNORECASE)
_BET_RE  = re.compile(r"(?P<f>[\w.]+)\s+between\s+(?P<a>[^,\s]+)\s+and\s+(?P<b>[^,\s]+)", re.IGNORECASE)
_LIKE_RE = re.compile(
    r"(?:(?P<f>[\w.]+)\s+like\s+(?P<q>['\"][^'\"]*['\"]))|(?:(?P<f2>[\w.]+)\s+contains\s+(?P<q2>['\"][^'\"]*['\"]))",
    re.IGNORECASE,
)

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1]
    return s

def _cast_scalar(s: str):
    t = s.strip()
    if not t:
        return ""
    t = t.rstrip(").; ")
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return _strip_quotes(t)
    tl = t.lower()
    if tl in ("true", "yes", "y"):
        return True
    if tl in ("false", "no", "n"):
        return False
    try:
        if "." in t:
            return float(t)
        return int(t)
    except Exception:
        return t

def _split_csv_outside_quotes(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    in_single = False
    in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue
        if ch == "," and not in_single and not in_double:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts

def _parse_in_list(s: str) -> List[Any]:
    return [_cast_scalar(p) for p in _split_csv_outside_quotes(s)]

def _map_field_to_canonical(token: str, dims_all: List[str], fields_cfg: Dict[str, str],
                            dim_syn: Optional[Dict[str, List[str]]]) -> Optional[str]:
    token_lc = token.strip().lower()
    # 1) exact canonical name
    for c in dims_all:
        if token_lc == c.lower():
            return c
    # 2) synonyms
    for canon, aliases in (dim_syn or {}).items():
        for a in aliases or []:
            if token_lc == a.lower():
                return canon
    # 3) leaf equals / contains token (fallback)
    for c in dims_all:
        leaf = c.lower().split(".", 1)[-1]
        if token_lc == leaf or token_lc in leaf:
            return c
    return None

def _parse_filters(nl_text: str,
                   dims_all: List[str],
                   fields_cfg: Dict[str, str],
                   dim_syn: Optional[Dict[str, List[str]]]) -> Tuple[List[Dict[str, Any]], bool]:
    text = nl_text or ""
    found: List[Dict[str, Any]] = []
    has_between = False

    for m in _BET_RE.finditer(text):
        field = _map_field_to_canonical(m.group("f"), dims_all, fields_cfg, dim_syn)
        if not field:
            continue
        a = _cast_scalar(m.group("a"))
        b = _cast_scalar(m.group("b"))
        found.append({"field": field, "op": "between", "value": [a, b]})
        has_between = True

    for m in _IN_RE.finditer(text):
        field = _map_field_to_canonical(m.group("f"), dims_all, fields_cfg, dim_syn)
        if not field:
            continue
        vals = _parse_in_list(m.group("vals"))
        if vals:
            found.append({"field": field, "op": "in", "value": vals})

    for m in _LIKE_RE.finditer(text):
        f = m.group("f") or m.group("f2")
        q = m.group("q") or m.group("q2")
        field = _map_field_to_canonical(f, dims_all, fields_cfg, dim_syn)
        if not field:
            continue
        found.append({"field": field, "op": "like", "value": _strip_quotes(q)})

    for m in _CMP_RE.finditer(text):
        field = _map_field_to_canonical(m.group("f"), dims_all, fields_cfg, dim_syn)
        if not field:
            continue
        op_map = {">=": "gte", "<=": "lte", ">": "gt", "<": "lt"}
        found.append({"field": field, "op": op_map[m.group("op")], "value": _cast_scalar(m.group("val"))})

    for m in _NEQ_RE.finditer(text):
        field = _map_field_to_canonical(m.group("f"), dims_all, fields_cfg, dim_syn)
        if not field:
            continue
        found.append({"field": field, "op": "neq", "value": _cast_scalar(m.group("val"))})

    for m in _EQ_RE.finditer(text):
        field = _map_field_to_canonical(m.group("f"), dims_all, fields_cfg, dim_syn)
        if not field:
            continue
        found.append({"field": field, "op": "eq", "value": _cast_scalar(m.group("val"))})

    return found, has_between

# -----------------------------------------------------------------------------
# Helpers: entities/tables, metrics & dimensions
# -----------------------------------------------------------------------------
def _all_entity_names(config: Any) -> Set[str]:
    names: Set[str] = set()
    entities_map = getattr(config, "entities", {}) or {}
    for k in entities_map.keys():
        names.add(str(k).lower())
    for mapping in (getattr(config, "fields", {}) or {}).values():
        if isinstance(mapping, str) and "." in mapping:
            names.add(mapping.split(".", 1)[0].lower())
    for mapping in (getattr(config, "dimensions", {}) or {}).values():
        if isinstance(mapping, str) and "." in mapping:
            names.add(mapping.split(".", 1)[0].lower())
    return names

def _find_entity_in_text(text: str, entities: Set[str]) -> Optional[str]:
    tokens = sorted(entities, key=lambda s: (-len(s), s))
    low = (text or "").lower()
    for t in tokens:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", low):
            return t
    return None

def _default_metric_for_entity(entity: str, metrics_cfg: Dict[str, Any]) -> Optional[str]:
    cand = f"{entity}.count"
    if cand in metrics_cfg:
        return cand
    for name, mdef in metrics_cfg.items():
        expr = getattr(mdef, "expression", None) or (mdef.get("expression") if isinstance(mdef, dict) else None)
        if isinstance(expr, str) and f"{entity}." in expr:
            return name
    return None

def _metric_tables(metric_names: List[str], metrics_cfg: Dict[str, Any]) -> Tuple[Set[str], Optional[str]]:
    tables: Set[str] = set()
    for name in metric_names:
        mdef = metrics_cfg.get(name)
        if not mdef:
            return set(), f"Unknown metric: {name}"
        expr_sql = getattr(mdef, "expression", None) or (mdef.get("expression") if isinstance(mdef, dict) else None)
        if not isinstance(expr_sql, str) or not expr_sql.strip():
            return set(), f"Metric '{name}' has no valid 'expression'"
        try:
            node = sqlglot.parse_one(expr_sql)
        except Exception as e:
            return set(), f"Metric '{name}' expression parse error: {e}"
        for col in node.find_all(exp.Column):
            if not col.table:
                return set(), f"Metric '{name}' uses unqualified column '{col.name}'. Use 'table.column' in config."
            tables.add(col.table)
    return tables, None

def _dims_on_entity(entity: str, dims_cfg: Dict[str, str], fields_cfg: Dict[str, str]) -> List[str]:
    out = []
    for n, m in (dims_cfg or {}).items():
        if isinstance(m, str) and m.startswith(f"{entity}."):
            out.append(n)
    for n, m in (fields_cfg or {}).items():
        if isinstance(m, str) and m.startswith(f"{entity}."):
            out.append(n)
    seen: Set[str] = set()
    res: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

def _pick_dimension_for_phrase(entity: str, phrase: str,
                               dims_cfg: Dict[str, str],
                               fields_cfg: Dict[str, str],
                               dim_syn: Optional[Dict[str, List[str]]]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Parse "<token> [free text]" → (canonical_dimension, trailing_value, suggestions_if_ambiguous)
    Only consider dimensions that live on the same entity table.
    """
    phrase = (phrase or "").strip()
    if not phrase:
        return None, None, []

    parts = phrase.split()
    dim_token = parts[0].lower()
    remainder = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

    entity_dims = _dims_on_entity(entity, dims_cfg, fields_cfg)

    # exact canonical match by leaf
    exact = [d for d in entity_dims if d.lower().split(".", 1)[-1] == dim_token]
    if len(exact) == 1:
        return exact[0], (remainder or None), []

    # synonyms
    for canon, aliases in (dim_syn or {}).items():
        if canon in entity_dims:
            for a in aliases or []:
                if a.lower() == dim_token:
                    return canon, (remainder or None), []

    # heuristic: token contained in leaf (e.g., "dealer" -> "licenses.dealer_id")
    partial = [d for d in entity_dims if dim_token in d.lower().split(".", 1)[-1]]
    if len(partial) == 1:
        return partial[0], (remainder or None), []

    # ambiguous: suggest top few leaf names
    suggestions = [d for d in entity_dims if dim_token in d.lower()]
    return None, None, suggestions[:5]

def _name_like(dim_name: str) -> bool:
    leaf = dim_name.lower().split(".", 1)[-1]
    if leaf.endswith("_id") or leaf == "id":
        return False
    return ("name" in leaf) or not leaf.endswith("_id")

# -----------------------------------------------------------------------------
# Main router
# -----------------------------------------------------------------------------
def route(
    nl_text: str,
    config: Any,
    *,
    tz: str = "Asia/Manila",
) -> Dict[str, Any]:
    """
    Deterministic rule router that converts NL → LogicalQueryRequest (or a structured error).
    Now supports:
      - entity→default metric inference ("licenses" → licenses.count)
      - by/per <dimension> [value] with single-table guardrails
      - automatic resolutions[] when value is for an *_id field (name→id via resolvers.yaml at AST time)
    """
    trace: Dict[str, Any] = {
        "router": "rules+",
        "tz": tz,
        "signals": {},
        "policies": ["deterministic", "single-table", "no-joins", "value→resolution when *_id"],
    }

    text = nl_text or ""
    metrics_cfg = getattr(config, "metrics", {}) or {}
    fields_cfg = getattr(config, "fields", {}) or {}
    dims_cfg = getattr(config, "dimensions", {}) or {}
    syn_cfg = getattr(getattr(config, "synonyms", None), "dict", None)
    dim_syn = syn_cfg().get("dimensions", {}) if callable(syn_cfg) else getattr(getattr(config, "synonyms", None), "dimensions", None)
    met_syn = syn_cfg().get("metrics", {}) if callable(syn_cfg) else getattr(getattr(config, "synonyms", None), "metrics", None)

    metric_names = sorted(list(metrics_cfg.keys()))
    dimension_names = sorted(list(dims_cfg.keys()))
    all_dim_field_names = sorted(list(set(list(dims_cfg.keys()) + list(fields_cfg.keys()))))

    # 1) Date/time range inference (unless explicit BETWEEN later)
    rng = infer_time_range(text, tz=tz)
    if rng:
        trace["signals"]["time_range"] = {"start": rng.start, "end": rng.end}

    # 2) Rank / limit
    lim, rank_dir = _extract_rank_and_limit(text)
    if lim is not None:
        trace["signals"]["limit"] = lim
    if rank_dir is not None:
        trace["signals"]["rank_dir"] = rank_dir

    # 3) Detect metrics directly (canonical and synonyms), else try entity → default metric
    detected_metrics: List[str] = []
    low = text.lower()
    # direct canonical mention
    for m in metric_names:
        patt = rf"(?<![A-Za-z0-9_]){re.escape(m.lower())}(?![A-Za-z0-9_])"
        if re.search(patt, low):
            detected_metrics.append(m)
    # synonyms
    for canon, aliases in (met_syn or {}).items():
        for a in aliases or []:
            patt = rf"(?<![A-Za-z0-9_]){re.escape(a.lower())}(?![A-Za-z0-9_])"
            if re.search(patt, low) and canon not in detected_metrics:
                detected_metrics.append(canon)

    if not detected_metrics:
        # Try entity→default metric
        ent = _find_entity_in_text(text, _all_entity_names(config))
        if ent:
            dm = _default_metric_for_entity(ent, metrics_cfg)
            if dm:
                detected_metrics = [dm]
                trace["signals"]["entity"] = ent

    if not detected_metrics:
        # No metric and no entity we can map → guide user
        suggestions = metric_names[:10]
        return {
            "error": "[MISSING_PARAMETER] Unable to identify metric or entity. "
                     "Please specify a metric (e.g., licenses.count, hosts.count).",
            "trace": {**trace, "suggest": {"metrics": suggestions}},
        }

    # 4) Determine base table from metrics (must be single-table)
    tables, err = _metric_tables(detected_metrics, metrics_cfg)
    if err:
        return {"error": f"[SCHEMA_MISMATCH] {err}", "trace": trace}
    if not tables:
        return {"error": "[SCHEMA_MISMATCH] Unable to derive base table from metric(s).", "trace": trace}
    if len(tables) > 1:
        return {"error": f"[SCHEMA_MISMATCH] Multiple base tables in metrics: {sorted(tables)}", "trace": trace}
    base_table = next(iter(tables))
    trace["signals"]["base_table"] = base_table

    # 5) BY/PER handling → choose dimension on the same base table; maybe a trailing value
    sel_dimensions: List[str] = []
    implicit_filters: List[Dict[str, Any]] = []
    resolutions: List[Dict[str, Any]] = []

    m = _BY_PER_RE.search(text)
    if m:
        phrase = m.group("phrase").strip()
        dname, trailing_value, suggestions = _pick_dimension_for_phrase(
            base_table, phrase, dims_cfg, fields_cfg, dim_syn
        )
        if dname:
            # enforce same-table (should be by construction)
            mapping = dims_cfg.get(dname) or fields_cfg.get(dname)
            if not mapping or not mapping.startswith(f"{base_table}."):
                return {
                    "error": f"[SCHEMA_MISMATCH] Dimension '{dname}' is not on base table '{base_table}'.",
                    "trace": trace,
                }
            sel_dimensions.append(dname)

            if trailing_value:
                # If it's name-like, emit LIKE; else if it's *_id and value isn't numeric -> resolution
                if _name_like(dname):
                    implicit_filters.append({"field": dname, "op": "like", "value": trailing_value})
                else:
                    # resolve via resolvers.yaml at AST time
                    resolutions.append({"target_fk": dname, "value": trailing_value})
        else:
            if suggestions:
                return {
                    "error": "[AMBIGUOUS_REQUEST] Unable to determine dimension for grouping.",
                    "trace": {**trace, "suggest": {"dimensions": suggestions, "base_table": base_table}},
                }
            # silent if no dimension phrase could be mapped
            trace["signals"]["by_per_unmapped"] = phrase

    # 6) Explicit filters in NL (eq/neq/cmp/in/between/like)
    explicit_filters, has_between = _parse_filters(text, all_dim_field_names, fields_cfg, dim_syn)
    implicit_filters.extend(explicit_filters)

    # 7) Time range: if explicit BETWEEN was used on a date column, suppress inferred time_range
    time_range_obj = None
    if rng and not has_between:
        time_range_obj = {"start": rng.start, "end": rng.end}

    # 8) OrderBy via rank; otherwise leave empty
    order_by: List[OrderBy] = []
    if rank_dir:
        order_by.append(OrderBy(field=detected_metrics[0], direction=Direction(rank_dir)))

    # 9) Limit: explicit or default
    limit_val = lim if lim is not None else 100

    # 10) Build LQR
    lqr = LogicalQueryRequest(
        metrics=detected_metrics,
        dimensions=sel_dimensions,
        filters=implicit_filters,
        order_by=order_by,
        limit=limit_val,
        time_range=time_range_obj,
        resolutions=resolutions,
    )

    return {
        "lqr": lqr.model_dump(),
        "trace": trace,
    }
