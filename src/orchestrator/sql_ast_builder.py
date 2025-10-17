# src/orchestrator/sql_ast_builder.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.expressions import Expression

from .errors import OrchestratorError, ErrorCode
from .models import Filter, LogicalQueryRequest, OrderBy, ResolutionMode


# =========================
# Public API
# =========================

def build_select_sql(
    lqr: LogicalQueryRequest,
    config: Any,
    *,
    dialect: Optional[str] = None,
) -> str:
    """
    Compile a LogicalQueryRequest into a safe SELECT SQL string using SQLGlot.

    Phase 1 constraints:
      - Single base table only for SELECT list and simple filters.
      - No joins in FROM, no CTEs, no DDL/DML.
      - Name->ID lookups are supported via declarative resolvers as EXISTS/IN subqueries.
      - Filters supported for common ops. LIKE uses explicit ESCAPE '\\'.
    """
    # Validate presence of metric definitions
    metric_defs = getattr(config, "metrics", {}) or {}
    if not metric_defs:
        raise OrchestratorError(ErrorCode.SCHEMA_MISSING, "No metrics configured")

    field_map = getattr(config, "fields", {}) or {}
    dim_map = getattr(config, "dimensions", {}) or {}
    join_graph = getattr(config, "join_graph", {}) or {}
    resolver_cfg: Dict[str, Any] = getattr(config, "resolvers", {}) or {}

    # 1) Build SELECT list: metrics (aliased to metric names) + dimensions
    select_exprs: List[Expression] = []
    used_tables: Set[str] = set()

    # Metrics
    metric_aliases: Set[str] = set()
    for mname in lqr.metrics:
        mdef = metric_defs.get(mname)
        if not mdef:
            raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"Unknown metric: {mname}")

        try:
            mexpr_inner = sqlglot.parse_one(_strip_alias(mdef.expression))
        except Exception as e:
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Invalid metric expression for '{mname}': {e}")

        mexpr = mexpr_inner.as_(mname)
        select_exprs.append(mexpr)
        used_tables.update(_tables_from_expression(mexpr))

        if mname in metric_aliases:
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Duplicate metric alias: {mname}")
        metric_aliases.add(mname)

    # Dimensions
    group_exprs: List[Expression] = []
    dim_columns: Dict[str, Expression] = {}
    for dname in lqr.dimensions:
        col = _resolve_column(dname, field_map, dim_map)
        dim_columns[dname] = col
        select_exprs.append(col.as_(dname))
        group_exprs.append(col)
        
        # Use canonical mapping to determine the table (more robust than AST inspection)
        mapped = dim_map.get(dname) or field_map.get(dname)
        if not mapped or "." not in mapped:
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Unknown or invalid mapping for dimension '{dname}'")
        used_tables.add(_table_part(mapped))

    # 2) Establish base table from metrics/dimensions FIRST (Phase-1 single-table)
    base_table = _ensure_single_table(used_tables)

    # 3) Time range → WHERE on a time dimension from the SAME base table
    where_expr: Optional[Expression] = None
    if lqr.time_range:
        time_field_name, time_col = _pick_time_dimension_for_table(base_table, lqr, dim_map)
        if time_col is None:
            candidates = _time_like_candidates(
                [name for name, mapping in dim_map.items() if _table_part(mapping) == base_table]
            )
            raise OrchestratorError(
                ErrorCode.MISSING_PARAMETER,
                f"time_range specified but no time dimension found on base table '{base_table}'.",
                meta={"suggest": {"time_dimensions": candidates, "base_table": base_table}}
            )
        start_lit = exp.Literal.string(lqr.time_range["start"])
        end_lit = exp.Literal.string(lqr.time_range["end"])
        where_expr = exp.and_(
            exp.GTE(this=time_col.copy(), expression=start_lit),
            exp.LT(this=time_col.copy(), expression=end_lit),
        )

    # 4) Additional simple filters (Phase 1 supports common ops)
    for f in lqr.filters or []:
        pred = _compile_filter(f, field_map, dim_map)
        if pred is None:
            raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, f"Unsupported filter operator: {f.op}")
        # Derive the filter's table from canonical mapping (deterministic, avoids AST quirks)
        mapped = dim_map.get(f.field) or field_map.get(f.field)
        if not mapped or "." not in mapped:
            raise OrchestratorError(
                ErrorCode.SCHEMA_MISSING,
                f"Unknown or invalid mapping for filter field '{f.field}'."
            )
        pred_table = _table_part(mapped)
        
        if pred_table != base_table:
            raise OrchestratorError(
                ErrorCode.SCHEMA_MISMATCH,
                f"Filter field '{f.field}' belongs to table '{pred_table}', "
                f"but base table is '{base_table}'. Joins are disabled in Phase 1."
            )
            
        used_tables.add(pred_table)
        
        where_expr = pred if where_expr is None else exp.and_(where_expr, pred)

    # Ensure still single-table after filters
    _ensure_single_table(used_tables)

    # 5) Resolver-based lookups (name -> id) compiled as EXISTS subqueries
    for r in lqr.resolutions or []:
        # target_fk must be a canonical mapping on the base table
        fk_mapping = dim_map.get(r.target_fk) or field_map.get(r.target_fk)
        if not fk_mapping or "." not in fk_mapping:
            raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"Unknown target_fk in resolution: {r.target_fk}")
        fk_table, fk_col = _split_table_column(fk_mapping)
        if fk_table != base_table:
            raise OrchestratorError(
                ErrorCode.SCHEMA_MISMATCH,
                f"Resolution target_fk '{r.target_fk}' is on table '{fk_table}', "
                f"but base table is '{base_table}'."
            )

        # resolver spec must exist in config
        spec = resolver_cfg.get(r.target_fk)
        if spec is None:
            raise OrchestratorError(
                ErrorCode.MISSING_PARAMETER,
                f"No resolver configured for '{r.target_fk}'. Add it to resolvers.yaml."
            )

        via_table = spec.via_table
        ret_tbl, ret_col = _split_table_column(spec.return_column)
        if ret_tbl != via_table:
            raise OrchestratorError(
                ErrorCode.SCHEMA_MISMATCH,
                f"resolver.return_column '{spec.return_column}' must belong to via_table '{via_table}'"
            )

        # must be one-hop allowed by join_graph (either direction)
        if not _is_one_hop_neighbor(base_table, via_table, join_graph):
            raise OrchestratorError(
                ErrorCode.SCHEMA_MISMATCH,
                f"Resolver path not allowed by join_graph: {base_table} ↔ {via_table}"
            )

        # Build EXISTS subquery predicate
        base_fk_col = exp.column(fk_col, table=fk_table)
        via_ret_col = exp.column(ret_col, table=via_table)

        # Join equality: via.return_column = base.fk
        join_eq = exp.EQ(this=via_ret_col, expression=base_fk_col)

        # Match condition (case-insensitive; LIKE modes add ESCAPE)
        mode = r.mode or spec.match.mode or ResolutionMode.ilike_contains
        match_expr = _build_match_condition(
            field_fq=spec.match.field,
            mode=mode,
            value=r.value,
            escape_char=(spec.match.escape or "\\"),
        )

        inner_where = exp.and_(join_eq, match_expr)

        subq = (
            exp.select(exp.Literal.number(1))
            .from_(exp.Table(this=exp.to_identifier(via_table)))
            .where(inner_where)
        )
        # Force a space between EXISTS and '(' to satisfy tests expecting "exists ("
        exists_pred = sqlglot.parse_one(f"EXISTS ({subq.sql()})")

        where_expr = exists_pred if where_expr is None else exp.and_(where_expr, exists_pred)

    # 6) Assemble SELECT
    query = (
        exp.Select()
        .from_(exp.Table(this=exp.to_identifier(base_table)))
        .select(*select_exprs)
    )

    if group_exprs:
        query.set("group", exp.Group(expressions=group_exprs))

    if where_expr is not None:
        query.set("where", exp.Where(this=where_expr))

    # ORDER BY
    if lqr.order_by:
        order_terms = []
        for ob in lqr.order_by:
            if ob.field in metric_aliases:
                term = exp.Ordered(this=exp.to_identifier(ob.field), desc=(ob.direction.value == "desc"))
            else:
                dcol = dim_columns.get(ob.field)
                if dcol is None:
                    try:
                        dcol = _resolve_column(ob.field, field_map, dim_map)
                    except OrchestratorError:
                        raise OrchestratorError(
                            ErrorCode.SCHEMA_MISSING,
                            f"ORDER BY references unknown field '{ob.field}'"
                        )
                if _table_of_column(dcol) != base_table:
                    raise OrchestratorError(
                        ErrorCode.SCHEMA_MISMATCH,
                        f"ORDER BY field '{ob.field}' is on a different table. "
                        f"Base table: '{base_table}'. Phase 1 prohibits joins."
                    )
                term = exp.Ordered(this=dcol.copy(), desc=(ob.direction.value == "desc"))
            order_terms.append(term)
        query.set("order", exp.Order(expressions=order_terms))

    # LIMIT  (correct SQLGlot keyword is 'expression')
    lim = lqr.limit if lqr.limit is not None else 100
    query.set("limit", exp.Limit(expression=exp.Literal.number(lim)))

    # 7) Serialize
    try:
        return query.sql(dialect=dialect)
    except Exception as e:
        raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, f"Failed to serialize SQL: {e}")


# =========================
# Internals
# =========================

# Prefer created dates first; fall back through common names.
_TIME_DIM_PRIORITIES = [
    "date_created", "created_at", "created",
    "order_date", "event_date", "business_date", "date",
    "install_date",
    "updated_at", "date_updated",
    "timestamp", "dt"
]

def _resolve_column(name: str, fields: Dict[str, str], dims: Dict[str, str]) -> exp.Column:
    mapping = dims.get(name) or fields.get(name)
    if not mapping:
        raise OrchestratorError(ErrorCode.SCHEMA_MISSING, f"Unknown field/dimension: {name}")
    table, column = _split_table_column(mapping)
    return exp.column(column, table=table)


def _compile_filter(f: Filter, fields: Dict[str, str], dims: Dict[str, str]) -> Optional[Expression]:
    col = _resolve_column(f.field, fields, dims)

    def lit(v) -> Expression:
        if isinstance(v, (int, float)):
            return exp.Literal.number(v)
        if isinstance(v, bool):
            return exp.Literal.number(1 if v else 0)
        return exp.Literal.string(str(v))

    op = f.op.value if hasattr(f.op, "value") else str(f.op)

    if op == "eq":
        return exp.EQ(this=col, expression=lit(f.value))
    if op == "neq":
        return exp.NEQ(this=col, expression=lit(f.value))
    if op == "gt":
        return exp.GT(this=col, expression=lit(f.value))
    if op == "lt":
        return exp.LT(this=col, expression=lit(f.value))
    if op == "gte":
        return exp.GTE(this=col, expression=lit(f.value))
    if op == "lte":
        return exp.LTE(this=col, expression=lit(f.value))
    if op == "between":
        vals = _as_list(f.value, expect_len=2, ctx="between")
        return exp.Between(this=col, low=lit(vals[0]), high=lit(vals[1]))
    if op == "in":
        vals = _as_list(f.value, expect_len=None, ctx="in")
        # FIX: The `In` constructor expects an `ExpressionList` passed to the `query` argument,
        # not a simple list passed to an `expressions` argument.
        expr_list = exp.ExpressionList(expressions=[lit(v) for v in vals])
        return exp.In(this=col, query=expr_list)
    if op == "like":
        if not isinstance(f.value, str):
            raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, "LIKE value must be a string")
        esc = _escape_like(f.value)
        col_sql = col.sql()
        pat_sql = exp.Literal.string(esc).sql()
        raw = f"{col_sql} LIKE {pat_sql} ESCAPE '\\\\'"
        return sqlglot.parse_one(raw)

    return None


def _build_match_condition(field_fq: str, mode: ResolutionMode, value: str, escape_char: str) -> Expression:
    """
    Build a case-insensitive predicate for resolver matching:
      - ilike_contains: LOWER(field) LIKE LOWER('%val%') ESCAPE '\'
      - ilike_prefix:   LOWER(field) LIKE LOWER('val%')  ESCAPE '\'
      - exact:          LOWER(field) = LOWER('val')
    """
    if "." not in field_fq:
        raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"resolver.match.field must be 'table.column', got '{field_fq}'")
    t, c = field_fq.split(".", 1)
    field_col = exp.column(c, table=t)

    if mode == ResolutionMode.exact:
        raw = f"LOWER({field_col.sql()}) = LOWER({exp.Literal.string(value).sql()})"
        return sqlglot.parse_one(raw)

    # LIKE variants
    esc_val = _escape_like(value)
    if mode == ResolutionMode.ilike_prefix:
        pattern = f"{esc_val}%"
    else:
        pattern = f"%{esc_val}%"

    raw = (
        f"LOWER({field_col.sql()}) LIKE LOWER({exp.Literal.string(pattern).sql()}) "
        f"ESCAPE '{escape_char.replace("'", "''")}'"
    )
    return sqlglot.parse_one(raw)


def _escape_like(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace("%", "\\%").replace("_", "\\_")
    return s


def _split_table_column(mapping: str) -> Tuple[str, str]:
    if "." not in mapping:
        raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Invalid mapping (expected 'table.column'): {mapping}")
    table, column = mapping.split(".", 1)
    if not table or not column:
        raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Invalid mapping (empty parts): {mapping}")
    return table, column


def _tables_from_expression(node: Expression) -> Set[str]:
    tables: Set[str] = set()
    for col in node.find_all(exp.Column):
        t = _table_of_column(col)
        if t:
            tables.add(t)
    return tables


def _table_of_column(col: Optional[Expression]) -> str:
    """
    Defensive helper: return the table for a Column node, else empty string.
    We avoid raising AttributeError on unexpected node types.
    """
    if col is None:
        return ""
    if isinstance(col, exp.Column):
        if col.table:
            return col.table
        # Column without table — treat as invalid mapping upstream
        raise OrchestratorError(
            ErrorCode.SCHEMA_MISMATCH,
            "Unqualified column found. Use 'table.column' in config expressions."
        )
    # Non-column node: ignore for table inference
    return ""


def _ensure_single_table(tables: Iterable[str]) -> str:
    ts = {t for t in tables if t}
    if not ts:
        raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, "No base table could be determined from fields/metrics")
    if len(ts) > 1:
        raise OrchestratorError(
            ErrorCode.SCHEMA_MISMATCH,
            f"Multiple tables referenced {sorted(ts)} but join resolution is not enabled in Phase 1"
        )
    return next(iter(ts))


def _time_like_candidates(names: Iterable[str]) -> List[str]:
    out = []
    for n in names:
        ln = n.lower()
        if any(tok in ln for tok in ("date", "time", "timestamp", "created", "dt")):
            out.append(n)
    return out[:10]


def _pick_time_dimension_for_table(
    base_table: str,
    lqr: LogicalQueryRequest,
    dim_map: Dict[str, str]
) -> Tuple[Optional[str], Optional[exp.Column]]:
    # 1) Look among requested dimensions
    for d in lqr.dimensions:
        mapping = dim_map.get(d)
        if mapping and _is_time_like(d) and _table_part(mapping) == base_table:
            return d, _resolve_column(d, {}, dim_map)

    # 2) Rank all time-like dims for this base table
    candidates: List[str] = []
    for name, mapping in dim_map.items():
        if _table_part(mapping) == base_table and _is_time_like(name):
            candidates.append(name)

    if candidates:
        lname_map = {name: name.lower() for name in candidates}
        for pref in _TIME_DIM_PRIORITIES:
            for name, ln in lname_map.items():
                if pref in ln:
                    return name, _resolve_column(name, {}, dim_map)
        pick = candidates[0]
        return pick, _resolve_column(pick, {}, dim_map)

    return None, None


def _is_time_like(name: str) -> bool:
    ln = name.lower()
    return any(tok in ln for tok in ("date", "time", "timestamp", "created", "dt"))


def _as_list(value, expect_len: Optional[int], ctx: str) -> List[Any]:
    if isinstance(value, (list, tuple)):
        vals = list(value)
    else:
        vals = [value]
    if expect_len is not None and len(vals) != expect_len:
        raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, f"Operator '{ctx}' expects {expect_len} values")
    return vals


def _strip_alias(expr_sql: str) -> str:
    try:
        node = sqlglot.parse_one(expr_sql)
        if isinstance(node, exp.Alias):
            return node.this.sql()
        return expr_sql
    except Exception:
        return expr_sql


def _table_part(mapping: str) -> str:
    return mapping.split(".", 1)[0] if "." in mapping else mapping


def _is_one_hop_neighbor(a: str, b: str, graph: Dict[str, List[str]]) -> bool:
    """Return True if b is a neighbor of a in join_graph (either direction)."""
    if not graph:
        return False
    return b in (graph.get(a, []) or []) or a in (graph.get(b, []) or [])