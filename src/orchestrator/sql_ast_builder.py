# src/orchestrator/sql_ast_builder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.expressions import Expression

from .errors import OrchestratorError, ErrorCode
from .models import LogicalQueryRequest


# -----------------------
# Small helpers
# -----------------------

def _table_part(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _col_part(dotted: str) -> str:
    return dotted.split(".", 1)[1]


def _col(dotted: str) -> exp.Column:
    t, c = _table_part(dotted), _col_part(dotted)
    return exp.column(c, table=t)


def _get(o: Any, key: str, default: Any = None) -> Any:
    """Access attribute or dict key uniformly."""
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _compile_metric(expr_sql: str, alias: str) -> exp.Expression:
    node = sqlglot.parse_one(expr_sql)
    return node.as_(alias)


def _lit(v: Any) -> Expression:
    if isinstance(v, bool):
        return exp.Boolean(this=v)
    if isinstance(v, (int, float)):
        return exp.Literal.number(v)
    return exp.Literal.string(str(v))


def _compile_filter(filt, field_map: Dict[str, str], dim_map: Dict[str, str]) -> Optional[Expression]:
    field = _get(filt, "field")
    op = _get(filt, "op")
    value = _get(filt, "value")
    dotted = dim_map.get(field) or field_map.get(field)
    if not dotted:
        return None
    col = _col(dotted)

    if op == "eq":
        return exp.EQ(this=col, expression=_lit(value))
    if op == "neq":
        return exp.NEQ(this=col, expression=_lit(value))
    if op == "gt":
        return exp.GT(this=col, expression=_lit(value))
    if op == "lt":
        return exp.LT(this=col, expression=_lit(value))
    if op == "gte":
        return exp.GTE(this=col, expression=_lit(value))
    if op == "lte":
        return exp.LTE(this=col, expression=_lit(value))
    if op == "between":
        a, b = value[0], value[1]
        return exp.Between(this=col, low=_lit(a), high=_lit(b))
    if op == "in":
        arr = value if isinstance(value, list) else [value]
        return exp.In(this=col, expressions=[_lit(v) for v in arr])
    if op == "like":
        # Compare LOWER(col) LIKE LOWER('value'); policy gate adds ESCAPE '\\'
        return exp.Like(this=exp.Lower(this=col), expression=exp.Lower(this=_lit(value)))
    return None


def _ensure_edge(join_graph: Dict[str, List[str]], a: str, b: str) -> None:
    neigh = set((join_graph or {}).get(a, []))
    if b not in neigh:
        raise OrchestratorError(
            ErrorCode.SCHEMA_MISMATCH,
            f"Join from '{a}' to '{b}' is not permitted by join_graph."
        )


# -----------------------
# Main compiler
# -----------------------

def build_select_sql(
    lqr: LogicalQueryRequest,
    config: Any,
    *,
    dialect: Optional[str] = None,
) -> str:
    """
    Build SELECT SQL from a LogicalQueryRequest.
    Supports:
      - Aggregate mode (metrics/dimensions)
      - Detail mode (projections) with safe lookup LEFT JOINs.
    """
    fields = getattr(config, "fields", {}) or {}
    dims = getattr(config, "dimensions", {}) or {}
    metrics_cfg = getattr(config, "metrics", {}) or {}
    resolvers = getattr(config, "resolvers", {}) or {}
    lookup_proj = getattr(config, "lookup_projections", {}) or {}
    join_graph = getattr(config, "join_graph", {}) or {}

    # ---------- Determine base table ----------
    base_table: Optional[str] = None

    if lqr.metrics:
        # derive base table(s) from metric expressions
        tables: Set[str] = set()
        for m in lqr.metrics:
            spec = metrics_cfg.get(m)
            if not spec:
                raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"Unknown metric '{m}'")
            try:
                expr_sql = spec.expression if hasattr(spec, "expression") else spec.get("expression")
                node = sqlglot.parse_one(expr_sql)
            except Exception as e:
                raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, f"Metric '{m}' expression parse error: {e}")
            for col in node.find_all(exp.Column):
                if not col.table:
                    raise OrchestratorError(
                        ErrorCode.SCHEMA_MISSING,
                        f"Metric '{m}' uses unqualified column '{col.name}'."
                    )
                tables.add(col.table)
        if not tables:
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, "Unable to derive base table from metrics.")
        if len(tables) > 1:
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Multiple base tables in metrics: {sorted(tables)}")
        base_table = next(iter(tables))
    else:
        # detail mode: infer base from first projection canonical name ("entity.field_alias")
        if not lqr.projections:
            raise OrchestratorError(ErrorCode.MISSING_PARAMETER, "No metrics or projections provided")
        first = lqr.projections[0]
        if "." not in first:
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, "Projection names must be <entity>.<alias>")
        base_table = _table_part(first)

    # ---------- FROM ----------
    select_exprs: List[Expression] = []
    where_expr: Optional[Expression] = None
    group_exprs: List[Expression] = []

    from_ = exp.Table(this=base_table)

    # ---------- Aggregate mode ----------
    if lqr.metrics:
        # metrics
        for m in lqr.metrics:
            spec = metrics_cfg[m]
            expr_sql = spec.expression if hasattr(spec, "expression") else spec.get("expression")
            select_exprs.append(_compile_metric(expr_sql, m))

        # dimensions
        for d in lqr.dimensions or []:
            dotted = dims.get(d) or fields.get(d)
            if not dotted:
                raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"Unknown dimension '{d}'")
            if _table_part(dotted) != base_table:
                raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Dimension '{d}' not on base table '{base_table}'")
            col = _col(dotted)
            select_exprs.append(col.as_(d))
            group_exprs.append(col)

    # ---------- Detail mode (projections) ----------
    else:
        for pname in lqr.projections:
            # same-table?
            dotted = dims.get(pname) or fields.get(pname)
            if dotted and _table_part(dotted) == base_table:
                select_exprs.append(_col(dotted).as_(pname))
                continue
            # lookup?
            spec = lookup_proj.get(pname)
            if spec:
                via = spec.via_table if hasattr(spec, "via_table") else spec.get("via_table")
                _ensure_edge(join_graph, base_table, via)
                sel = spec.select if hasattr(spec, "select") else spec.get("select")
                select_exprs.append(_col(sel).as_(pname))
                continue
            raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"Unknown projection '{pname}'")

    # ---------- Filters ----------
    for f in lqr.filters or []:
        pred = _compile_filter(f, fields, dims)
        if pred is None:
            raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, f"Unsupported filter: {_get(f,'field')} {_get(f,'op')}")
        # table guard
        mapped = dims.get(_get(f, "field")) or fields.get(_get(f, "field"))
        if not mapped or _table_part(mapped) != base_table:
            raise OrchestratorError(
                ErrorCode.SCHEMA_MISMATCH,
                f"Filter field '{_get(f,'field')}' not on base table '{base_table}' in Phase 1."
            )
        where_expr = pred if where_expr is None else exp.and_(where_expr, pred)

    # ---------- Time range ----------
    if lqr.time_range:
        start = lqr.time_range.get("start")
        end = lqr.time_range.get("end")
        date_candidates = [
            f"{base_table}.date_created",
            f"{base_table}.install_date",
            f"{base_table}.created_at",
        ]
        chosen = next((d for d in date_candidates if d in fields.values() or d in dims.values()), date_candidates[0])
        c = _col(chosen)
        rng_pred = exp.and_(
            exp.GTE(this=c, expression=exp.Literal.string(start)),
            exp.LT(this=c, expression=exp.Literal.string(end)),
        )
        where_expr = rng_pred if where_expr is None else exp.and_(where_expr, rng_pred)

    # ---------- Resolutions -> EXISTS subqueries ----------
    for res in lqr.resolutions or []:
        target_fk = _get(res, "target_fk")
        value = _get(res, "value")
        mode = _get(res, "mode")  # may be None; fallback to config
        if not target_fk:
            raise OrchestratorError(ErrorCode.MISSING_PARAMETER, "Resolution missing target_fk")

        rspec = resolvers.get(target_fk)
        if not rspec:
            raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"No resolver for '{target_fk}'")

        via = rspec.via_table if hasattr(rspec, "via_table") else rspec.get("via_table")
        return_col = rspec.return_column if hasattr(rspec, "return_column") else rspec.get("return_column")
        match = rspec.match if hasattr(rspec, "match") else rspec.get("match", {})
        match_field = match.get("field")
        cfg_mode = match.get("mode")
        # prefer LQR-specified mode if present; else config
        mode = mode or cfg_mode or "ilike_contains"

        _ensure_edge(join_graph, base_table, via)

        # Build name predicate according to mode
        val_str = str(value)
        if mode == "exact":
            name_pred = exp.EQ(
                this=exp.Lower(this=_col(match_field)),
                expression=exp.Lower(this=_lit(val_str)),
            )
        elif mode == "ilike_prefix":
            name_pred = exp.Like(
                this=exp.Lower(this=_col(match_field)),
                expression=exp.Lower(this=_lit(f"{val_str}%")),
            )
        else:  # ilike_contains (default)
            name_pred = exp.Like(
                this=exp.Lower(this=_col(match_field)),
                expression=exp.Lower(this=_lit(f"%{val_str}%")),
            )

        # Base-table FK column (use canonical mapping if present)
        base_fk_dotted = (
            fields.get(target_fk) or
            dims.get(target_fk) or
            f"{base_table}.{_col_part(target_fk)}"
        )

        subq_where = exp.and_(
            exp.EQ(this=_col(return_col), expression=_col(base_fk_dotted)),
            name_pred,
        )
        subq = exp.select(exp.Literal.number(1)).from_(exp.to_table(via)).where(subq_where)

        exists_pred = sqlglot.parse_one(f"EXISTS ({subq.sql()})")
        where_expr = exists_pred if where_expr is None else exp.and_(where_expr, exists_pred)

    # ---------- Build SELECT ----------
    select_stmt = exp.select(*select_exprs).from_(from_)

    # Apply LEFT JOINs for lookup projections (detail mode)
    if not lqr.metrics:
        added: Set[str] = set()
        for pname in lqr.projections:
            spec = lookup_proj.get(pname)
            if not spec:
                continue
            via = spec.via_table if hasattr(spec, "via_table") else spec.get("via_table")
            if via in added:
                continue
            _ensure_edge(join_graph, base_table, via)
            # Build ON: AND of all join_on clauses
            join_on = spec.join_on if hasattr(spec, "join_on") else spec.get("join_on", [])
            on_expr: Optional[Expression] = None
            for clause in join_on:
                left, right = [p.strip() for p in clause.split("=", 1)]
                cond = exp.EQ(this=_col(left), expression=_col(right))
                on_expr = cond if on_expr is None else exp.and_(on_expr, cond)
            select_stmt = select_stmt.join(exp.to_table(via), on=on_expr, join_type="left")
            added.add(via)

    # WHERE
    if where_expr is not None:
        select_stmt = select_stmt.where(where_expr)

    # GROUP BY / ORDER BY
    if lqr.metrics and group_exprs:
        select_stmt = select_stmt.group_by(*group_exprs)

    if lqr.order_by:
        for ob in lqr.order_by:
            select_stmt = select_stmt.order_by(exp.Identifier(this=ob.field), desc=(ob.direction.value == "desc"))

    # LIMIT
    if lqr.limit:
        select_stmt = select_stmt.limit(lqr.limit)

    sql = select_stmt.sql(dialect=dialect)
    return sql
