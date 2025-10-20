# src/orchestrator/router_runtime.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from .date_normalizer import infer_time_range
from .models import Direction, LogicalQueryRequest, OrderBy
from .tokenizer import tokenize, TokType, collect_until, tokens_to_text


# ---------- Helpers from config ----------

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
    # de-dupe preserving order
    seen: Set[str] = set(); res: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    return res

def _leaf(name: str) -> str:
    return name.lower().split(".", 1)[-1]

def _name_like(dim_name: str) -> bool:
    leaf = _leaf(dim_name)
    if leaf.endswith("_id") or leaf == "id": return False
    return ("name" in leaf) or not leaf.endswith("_id")

def _is_id_leaf(leaf: str) -> bool:
    return leaf.endswith("_id") or leaf == "id"

def _pick_fk_or_name_for_token(entity: str, token: str,
                               dims_cfg: Dict[str, str],
                               fields_cfg: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    tok = token.lower()
    entity_dims = _dims_on_entity(entity, dims_cfg, fields_cfg)
    fk_candidates = []; name_candidates = []
    for d in entity_dims:
        leaf = _leaf(d)
        if tok in leaf:
            if _is_id_leaf(leaf): fk_candidates.append(d)
            elif "name" in leaf: name_candidates.append(d)
    exact_fk = [d for d in fk_candidates if _leaf(d) in (f"{tok}_id", "id")]
    exact_name = [d for d in name_candidates if _leaf(d) in (f"{tok}_name", "name")]
    fk = exact_fk[0] if exact_fk else (fk_candidates[0] if fk_candidates else None)
    nm = exact_name[0] if exact_name else (name_candidates[0] if name_candidates else None)
    return fk, nm


# ---------- Boundaries (from patterns.yaml if present, else sensible defaults) ----------

_DEFAULT_STOP_BY_PER = {"in", "on", "between", "where", "top", "bottom", "limit", "for"}
_DEFAULT_STOP_FOR    = {"in", "on", "between", "where", "top", "bottom", "limit", "by", "per"}


def _load_boundaries(config: Any) -> Tuple[Set[str], Set[str]]:
    patt = getattr(config, "patterns", None)
    if patt and getattr(patt, "shapes", None):
        shapes = patt.shapes or {}
        b = shapes.get("boundaries", {})
        stop_by_per = set(b.get("stop_after_by_per", _DEFAULT_STOP_BY_PER))
        stop_for    = set(b.get("stop_after_for", _DEFAULT_STOP_FOR))
        return stop_by_per, stop_for
    return _DEFAULT_STOP_BY_PER, _DEFAULT_STOP_FOR


# ---------- Main router (deterministic, single-table) ----------

def route(nl_text: str, config: Any, *, tz: str = "Asia/Manila") -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "router": "runtime",
        "tz": tz,
        "signals": {},
        "policies": ["deterministic", "single-table", "value→resolution when *_id"],
    }

    text = nl_text or ""
    toks = tokenize(text)
    stop_by_per, stop_for = _load_boundaries(config)

    # treat these as segment boundaries; treat of/under like for
    stop_show = set(stop_by_per) | set(stop_for) | {"for", "by", "per", "of", "under"}
    like_for_tokens = {"for", "of", "under"}

    metrics_cfg = getattr(config, "metrics", {}) or {}
    fields_cfg = getattr(config, "fields", {}) or {}
    dims_cfg = getattr(config, "dimensions", {}) or {}
    lookup_proj_cfg = getattr(config, "lookup_projections", {}) or {}
    
    # ---------- 1) rank/limit ----------
    limit_val: Optional[int] = None
    rank_dir: Optional[str] = None
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type == TokType.KEYWORD and t.value.lower() in ("top", "bottom"):
            j = i + 1
            if j < len(toks) and toks[j].type == TokType.HYPHEN:
                j += 1
            if j < len(toks) and toks[j].type == TokType.NUMBER:
                limit_val = int(toks[j].value)
                rank_dir = "desc" if t.value.lower() == "top" else "asc"
                trace["signals"]["limit"] = limit_val
                trace["signals"]["rank_dir"] = rank_dir
                i = j + 1
                continue
        if t.type == TokType.KEYWORD and t.value.lower() == "limit":
            j = i + 1
            if j < len(toks) and toks[j].type == TokType.NUMBER:
                limit_val = int(toks[j].value)
                trace["signals"]["limit"] = limit_val
                i = j + 1
                continue
        i += 1

    # ---------- 2) time range (inferred; explicit BETWEEN parsed later) ----------
    rng = infer_time_range(text, tz=tz)
    if rng:
        trace["signals"]["time_range"] = {"start": rng.start, "end": rng.end}

    # ---------- 3) detect metric(s) or entity ----------
    detected_metrics: List[str] = []
    metric_names = list(metrics_cfg.keys())
    metric_low = {m.lower(): m for m in metric_names}

    for t in toks:
        if t.type in (TokType.IDENT, TokType.KEYWORD):
            m = metric_low.get(t.value.lower())
            if m and m not in detected_metrics:
                detected_metrics.append(m)

    entities = _all_entity_names(config)
    base_entity_hint: Optional[str] = None
    for t in toks:
        if t.type in (TokType.IDENT, TokType.KEYWORD):
            tv = t.value.lower()
            if tv in entities:
                base_entity_hint = tv
                break

    if not detected_metrics and base_entity_hint:
        dm = _default_metric_for_entity(base_entity_hint, metrics_cfg)
        if dm:
            detected_metrics = [dm]
            trace["signals"]["entity"] = base_entity_hint

    # ---------- 4) BY/PER (aggregates) ----------
    sel_dimensions: List[str] = []
    filters: List[Dict[str, Any]] = []
    resolutions: List[Dict[str, Any]] = []

    by_idx = None
    for idx, t in enumerate(toks):
        if t.type == TokType.KEYWORD and t.value.lower() in ("by", "per"):
            by_idx = idx
            break

    if by_idx is not None:
        seg, next_i = collect_until(toks, by_idx + 1, stop_by_per)
        dim_token = None
        for st in seg:
            if st.type == TokType.IDENT:
                dim_token = st.value
                break
        if dim_token:
            entity_dims = _dims_on_entity(base_entity_hint or "", dims_cfg, fields_cfg) if base_entity_hint else []
            matches = [d for d in entity_dims if _leaf(d) == dim_token.lower()] or \
                      [d for d in entity_dims if dim_token.lower() in _leaf(d)]
            if len(matches) == 1:
                dname = matches[0]
                sel_dimensions.append(dname)
                # Any trailing literal becomes LIKE or a resolution depending on field shape
                after_first = []
                seen_first_ident = False
                for st in seg:
                    if not seen_first_ident and st.type == TokType.IDENT:
                        seen_first_ident = True
                        continue
                    after_first.append(st)
                val = tokens_to_text(after_first).strip()
                if val:
                    if _name_like(dname):
                        filters.append({"field": dname, "op": "like", "value": val})
                    else:
                        resolutions.append({"target_fk": dname, "value": val})
            elif len(matches) > 1:
                return {
                    "error": "[AMBIGUOUS_REQUEST] Dimension token maps to multiple fields on base table.",
                    "trace": {**trace, "suggest": {"dimensions": matches[:5]}},
                }
            else:
                trace["signals"]["by_per_unmapped"] = tokens_to_text(seg)

    # ---------- 4.5) FOR/OF/UNDER <token> <value> → prefer *_id resolution; else LIKE on name ----------
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type == TokType.KEYWORD and t.value.lower() in like_for_tokens:
            if i + 1 < len(toks) and toks[i + 1].type == TokType.IDENT:
                tok_ident = toks[i + 1].value
                seg, i2 = collect_until(toks, i + 2, stop_show)
                val = tokens_to_text(seg).strip()
                if tok_ident and val:
                    fk_dim, name_dim = _pick_fk_or_name_for_token(base_entity_hint or "", tok_ident, dims_cfg, fields_cfg)
                    if fk_dim:
                        resolutions.append({"target_fk": fk_dim, "value": val})
                    elif name_dim:
                        filters.append({"field": name_dim, "op": "like", "value": val})
                    else:
                        trace["signals"]["for_unmapped"] = {"token": tok_ident, "value": val}
                i = i2
                continue
        i += 1

    # ---------- 5) explicit filters ----------
    def _as_scalar(tok: str):
        low = tok.lower()
        if low in ("true", "yes", "y"): return True
        if low in ("false", "no", "n"): return False
        try:
            if "." in tok: return float(tok)
            return int(tok)
        except Exception:
            return tok

    i = 0
    while i + 1 < len(toks):
        if toks[i].type == TokType.IDENT and "." in toks[i].value:
            field = toks[i].value
            # like | contains
            if i + 2 < len(toks) and toks[i+1].type == TokType.KEYWORD and toks[i+1].value.lower() in ("like", "contains"):
                if toks[i+2].type in (TokType.STRING, TokType.IDENT, TokType.NUMBER):
                    val = toks[i+2].value
                    filters.append({"field": field, "op": "like", "value": val})
                    i += 3
                    continue
            # in (...)
            if i + 4 < len(toks) and toks[i+1].type == TokType.KEYWORD and toks[i+1].value.lower() == "in" and toks[i+2].type == TokType.LPAREN:
                j = i + 3
                vals: List[str] = []
                cur: List[str] = []
                while j < len(toks) and toks[j].type != TokType.RPAREN:
                    if toks[j].type == TokType.COMMA:
                        if cur:
                            vals.append("".join(cur).strip()); cur = []
                        j += 1; continue
                    if toks[j].type in (TokType.STRING, TokType.IDENT, TokType.NUMBER):
                        cur.append(toks[j].value)
                    j += 1
                if cur: vals.append("".join(cur).strip())
                if j < len(toks) and toks[j].type == TokType.RPAREN:
                    filters.append({"field": field, "op": "in", "value": [_as_scalar(v) for v in vals]})
                    i = j + 1
                    continue
            # between A and B
            if (i + 4 < len(toks) and
                toks[i+1].type == TokType.KEYWORD and toks[i+1].value.lower() == "between" and
                toks[i+2].type in (TokType.STRING, TokType.IDENT, TokType.NUMBER) and
                toks[i+3].type == TokType.KEYWORD and toks[i+3].value.lower() == "and" and
                toks[i+4].type in (TokType.STRING, TokType.IDENT, TokType.NUMBER)):
                a = toks[i+2].value; b = toks[i+4].value
                filters.append({"field": field, "op": "between", "value": [a, b]})
                i += 5
                continue
            # comparisons
            if i + 2 < len(toks) and toks[i+1].type == TokType.OP and toks[i+2].type in (TokType.STRING, TokType.IDENT, TokType.NUMBER):
                opmap = {"=": "eq", "!=": "neq", "<>": "neq", ">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}
                op = opmap.get(toks[i+1].value)
                if op:
                    filters.append({"field": field, "op": op, "value": _as_scalar(toks[i+2].value)})
                    i += 3
                    continue
        i += 1

    # ---------- 6) between → suppress inferred time_range ----------
    has_between = any(f.get("op") == "between" for f in filters)
    time_range_obj = None
    if rng and not has_between:
        time_range_obj = {"start": rng.start, "end": rng.end}

    # ---------- 7) order_by (rank) ----------
    order_by: List[OrderBy] = []
    if rank_dir and detected_metrics:
        order_by.append(OrderBy(field=detected_metrics[0], direction=Direction(rank_dir)))

    # ---------- 8) limit ----------
    if limit_val is None:
        limit_val = 100

    # ---------- 9) Detail-mode detection: 'show' / 'list' ----------
    proj_tokens_start = None
    for idx, t in enumerate(toks):
        if t.type == TokType.KEYWORD and t.value.lower() in ("show", "list"):
            proj_tokens_start = idx + 1
            if proj_tokens_start < len(toks) and toks[proj_tokens_start].type in (TokType.KEYWORD, TokType.IDENT) and toks[proj_tokens_start].value.lower() in ("me", "the"):
                proj_tokens_start += 1
            break

    projections: List[str] = []
    if proj_tokens_start is not None and base_entity_hint:
        seg, _ = collect_until(toks, proj_tokens_start, stop_show)
        parts: List[List[Any]] = [[]]
        for st in seg:
            if st.type == TokType.COMMA or (st.type == TokType.KEYWORD and st.value.lower() == "and"):
                if parts[-1]:
                    parts.append([])
                continue
            parts[-1].append(st)
        phrases = [" ".join([s.value for s in grp]).strip() for grp in parts if grp]

        # Map phrases to canonical alias names on the base entity using:
        entity_fields = [
            n for n, m in (fields_cfg or {}).items()
            if isinstance(m, str) and m.startswith(f"{base_entity_hint}.")
        ]
        entity_fields += [
            n for n, m in (dims_cfg or {}).items()
            if isinstance(m, str) and m.startswith(f"{base_entity_hint}.")
        ]
        entity_fields += [
            n for n in (lookup_proj_cfg or {}).keys()
            if isinstance(n, str) and n.startswith(f"{base_entity_hint}.")
        ]
        
        seen_e: Set[str] = set(); ef: List[str] = []
        for n in entity_fields:
            if n not in seen_e:
                seen_e.add(n); ef.append(n)
        def _match_alias(phrase: str) -> Optional[str]:
            key = phrase.lower().replace("-", " ").replace("_", " ")
            key = " ".join([k for k in key.split() if k not in {"the"}]).strip()
            key = key.replace(" ", "_")
            # exact leaf
            for alias in ef:
                if _leaf(alias) == key:
                    return alias
            # fuzzy contains either side
            for alias in ef:
                leaf = _leaf(alias)
                if key in leaf or leaf in key:
                    return alias
            return None

        for ph in phrases:
            a = _match_alias(ph)
            if a and a not in projections:
                projections.append(a)

    # ---------- 10) Build LQR (detail takes precedence if projections found) ----------
    if projections and base_entity_hint:
        lqr = LogicalQueryRequest(
            metrics=[],  # detail mode
            projections=projections,
            dimensions=[],
            filters=filters,
            resolutions=resolutions,
            order_by=[],  # explicit sort comes later if needed
            limit=limit_val,
            time_range=time_range_obj,
        )
        return {"lqr": lqr.model_dump(), "trace": trace}

    # If no projections, we require metrics (aggregate)
    if not detected_metrics:
        return {
            "error": "[MISSING_PARAMETER] Unable to identify metric or projections.",
            "trace": {**trace, "suggest": {"metrics": metric_names[:10]}},
        }

    # ---------- Aggregate LQR ----------
    lqr = LogicalQueryRequest(
        metrics=detected_metrics,
        dimensions=sel_dimensions,
        filters=filters,
        resolutions=resolutions,
        order_by=order_by,
        limit=limit_val,
        time_range=time_range_obj,
    )
    return {"lqr": lqr.model_dump(), "trace": trace}
