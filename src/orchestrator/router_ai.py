from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
from datetime import date


# ---- tiny helpers -----------------------------------------------------------

def _lc(s: Optional[str]) -> str:
    return (s or "").lower()


def _has(q: str, *needles: str) -> bool:
    ql = _lc(q)
    return any(n in ql for n in needles)


def _parse_top_limit(q: str) -> Optional[int]:
    m = re.search(r"\btop[-\s]?(\d+)\b", _lc(q))
    if m:
        return int(m.group(1))
    return None


def _last_quarter_today(tz: str) -> Tuple[str, str]:
    """
    Previous calendar quarter [start, end) (dates only).
    Example: if today is 2025-10-21, last quarter -> [2025-07-01, 2025-10-01).
    """
    today = date.today()
    y = today.year
    q = (today.month - 1) // 3 + 1  # 1..4
    if q == 1:
        y -= 1
        q = 4
    else:
        q -= 1
    start_month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    start = date(y, start_month, 1)
    end = date(y + 1, 1, 1) if q == 4 else date(y, start_month + 3, 1)
    return (start.isoformat(), end.isoformat())


def _exists(name: str, container: Dict[str, Any]) -> bool:
    return isinstance(container, dict) and name in container


def _first_matching(prefix: str, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k.startswith(prefix):
            return k
    return None


def _collect_aliases_for_entity(entity: str, cfg: Any) -> List[str]:
    fields = list((getattr(cfg, "fields", {}) or {}).keys())
    dims = list((getattr(cfg, "dimensions", {}) or {}).keys())
    lookups = list((getattr(cfg, "lookup_projections", {}) or {}).keys())
    aliases = [a for a in fields + dims + lookups if isinstance(a, str) and a.startswith(f"{entity}.")]
    # de-dupe keep order
    seen = set()
    out: List[str] = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# ---- main entry -------------------------------------------------------------

def translate(question: str, cfg: Any, *, tz: str = "Asia/Manila") -> Dict[str, Any]:
    """
    Boxed AI translator:
      - Returns an LQR JSON dict (never SQL)
      - Picks safe base + metric/dimensions/projections from cfg enums
      - Respects Phase-1 single-base guardrails
      - STRICT MODE: if query uses vague metric words (e.g., "performance") and
        no specific metric exists for the base, return [MISSING_PARAMETER].
    """
    q = _lc(question)

    metrics_cfg = getattr(cfg, "metrics", {}) or {}
    dims_cfg = getattr(cfg, "dimensions", {}) or {}
    fields_cfg = getattr(cfg, "fields", {}) or {}
    lookups_cfg = getattr(cfg, "lookup_projections", {}) or {}

    metric_names = list(metrics_cfg.keys())
    dim_names = list(dims_cfg.keys())

    # ---- Choose base table --------------------------------------------------
    # "installs" => hosts (install_date semantics); else prefer licenses if present.
    if _has(q, "install", "installs"):
        base = "hosts"
    elif _has(q, "host", "hosts"):
        base = "hosts"
    elif _has(q, "license", "licenses"):
        base = "licenses"
    else:
        base = "licenses"

    # ---- Mode: aggregate if "by ..." or "top N" or "performance" -----------
    wants_aggregate = _has(q, " by ", " per ", " performance") or (_parse_top_limit(q) is not None)

    # ---- Aggregate path -----------------------------------------------------
    if wants_aggregate:
        # STRICT: if user said "performance" but no specific metric exists for base, reject.
        # "Specific" here means: any metric under base that is NOT "<base>.count".
        if _has(q, "performance"):
            base_metrics = [m for m in metric_names if m.startswith(f"{base}.")]
            specific = [m for m in base_metrics if not m.endswith(".count")]
            if not specific:
                suggest = base_metrics[:6] or metric_names[:6]
                return {
                    "error": "[MISSING_PARAMETER] 'performance' requires a concrete metric for "
                             f"base '{base}'. Define one in metrics.yaml or specify it explicitly.",
                    "trace": {"ai": "strict-metric-missing", "base": base},
                    "suggest": {"metrics": suggest},
                }

        # pick a safe count metric on the base (when allowed)
        preferred_metric = f"{base}.count"
        if not _exists(preferred_metric, metrics_cfg):
            m = _first_matching(f"{base}.", metric_names)
            if not m:
                return {"error": "[MISSING_PARAMETER] No metric available for base table",
                        "trace": {"ai": "no-metric-under-base", "base": base}}
            preferred_metric = m

        metrics = [preferred_metric]

        # dimension guesses from hints; keep within the base table
        dims: List[str] = []
        if base == "hosts" and _has(q, "store"):
            # "store" → prefer hosts.name if modeled as a dimension; else hosts.host_id
            cand = "hosts.name"
            dims.append(cand if cand in dim_names else "hosts.host_id")

        if base == "licenses" and _has(q, "partner"):
            # "partner" → safe Phase-1 grouping: licenses.dealer_id
            cand = "licenses.dealer_id"
            dims.append(cand if cand in dim_names else "licenses.dealer_id")

        if not dims:
            # generic fallback: <base>.<singular>_id if available, else first dimension
            pk = f"{base}.{base[:-1]}_id" if base.endswith("s") else f"{base}.{base}_id"
            dims.append(pk if pk in dim_names else (dim_names[0] if dim_names else pk))

        # time window
        filters: List[Dict[str, Any]] = []
        tr = None
        if _has(q, "last quarter"):
            start, end = _last_quarter_today(tz)
            if base == "hosts" and _has(q, "install"):
                # Use explicit install_date filters to force the correct date basis
                filters.append({"field": "hosts.install_date", "op": "gte", "value": start})
                filters.append({"field": "hosts.install_date", "op": "lt", "value": end})
                tr = None
            else:
                tr = {"start": start, "end": end}

        # order/limit
        order_by = [{"field": metrics[0], "direction": "desc"}] if _has(q, "top", "performance") else []
        limit = _parse_top_limit(q) or 100

        lqr = {
            "metrics": metrics,
            "dimensions": dims,
            "filters": filters,
            "order_by": order_by,
            "limit": limit,
            "time_range": tr,
            "projections": [],
            "resolutions": [],
        }
        return {"lqr": lqr, "trace": {"ai": "aggregate-heuristic", "base": base}}

    # ---- Detail path --------------------------------------------------------
    # If user says "show/list" without a "by", prefer detail with simple projections
    aliases = _collect_aliases_for_entity(base, cfg)

    # choose a minimal, safe projection set: <base>.<pk> and a name-like field
    pk = f"{base}.{base[:-1]}_id" if base.endswith("s") else f"{base}.{base}_id"
    proj: List[str] = []
    if pk in aliases:
        proj.append(pk)
    else:
        # fallback: any *_id on base
        any_id = next((a for a in aliases if a.split(".")[-1].endswith("_id")), None)
        if any_id:
            proj.append(any_id)

    # name-ish alias preferences
    for cand in (f"{base}.name", f"{base}.host_name", f"{base}.business_name"):
        if cand in aliases and cand not in proj:
            proj.append(cand)
            break

    # Limit: leading number like "10 hosts ..." else 100
    m = re.match(r"^\s*(\d+)\b", q or "")
    limit = int(m.group(1)) if m else 100

    lqr = {
        "metrics": [],
        "projections": proj or [pk],  # ensure non-empty for detail
        "dimensions": [],
        "filters": [],
        "order_by": [],
        "limit": limit,
        "time_range": None,
        "resolutions": [],
    }
    return {"lqr": lqr, "trace": {"ai": "detail-heuristic", "base": base}}
