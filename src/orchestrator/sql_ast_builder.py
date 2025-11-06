from __future__ import annotations

"""
SQL AST Builder (Phase 1)

Compiles a LogicalQueryRequest (LQR) into a single-table SELECT using SQLGlot.
- Programmatic AST only (no string concatenation)
- Re-parses final SQL to verify the first token is SELECT
- MySQL dialect by default

Public API:
    compile_lqr_to_sql(lqr: LogicalQueryRequest, *, config_dir: Path|None=None, dialect: str="mysql") -> str
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp, parse_one
from .config_loader import load_all_configs
from .models import LogicalQueryRequest, Filter, Operator, OrderBy, Direction


# ---------------- Metric resolution ---------------- #

@dataclass
class MetricSpec:
    kind: str  # "count" | "sum" | "avg"
    column: Optional[str] = None  # for sum/avg


def _load_metric_specs(config_dir: Optional[Path]) -> Dict[str, MetricSpec]:
    bundle = load_all_configs((config_dir or Path("src/orchestrator/config")).resolve())
    raw = bundle.metrics or {}
    out: Dict[str, MetricSpec] = {}
    for key, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        agg = str(meta.get("agg", "")).lower()
        if agg == "count":
            out[key] = MetricSpec(kind="count")
        elif agg == "sum":
            col = meta.get("column")
            if isinstance(col, str) and col:
                out[key] = MetricSpec(kind="sum", column=col)
        elif agg == "avg":
            col = meta.get("column")
            if isinstance(col, str) and col:
                out[key] = MetricSpec(kind="avg", column=col)
        # ignore others for Phase 1
    return out


# ---------------- Helpers ---------------- #
_LOOKUP_CACHE = {}

def _lookup_for_base(base_table: str):
    # cache on base_table; default config dir used (Phase 1/2 default)
    if base_table in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[base_table]
    bundle = load_all_configs(Path("src/orchestrator/config").resolve())
    lp = getattr(bundle, "lookup_projections", {}) or {}
    _LOOKUP_CACHE.update(lp or {})
    return _LOOKUP_CACHE.get(base_table) or {}

def _is_base_field(base_table: str, qualified: str) -> bool:
    return qualified.startswith(base_table + ".")

def _col_ref(qualified: str) -> exp.Expression:
    if "." in qualified:
        t, c = qualified.split(".", 1)
        return exp.column(c, table=t)  # renders as `t`.`c` in MySQL
    return exp.column(qualified)

def _lit(v: Any) -> exp.Expression:
    if v is None:
        return exp.Null()
    if isinstance(v, bool):
        return exp.Boolean(this=v)
    if isinstance(v, (int, float)):
        return exp.Literal.number(v)
    # strings and dates as string literals; DB will cast appropriately
    return exp.Literal.string(str(v))

def _build_predicate(f) -> exp.Expression:
    op = f.op.value if hasattr(f.op, "value") else str(f.op)
    col = _col_ref(f.field) if isinstance(f.field, str) else f.field
    val = f.value

    if op == "eq":   return exp.EQ(this=col, expression=_lit(val))
    if op == "ne":   return exp.NEQ(this=col, expression=_lit(val))
    if op == "gt":   return exp.GT(this=col, expression=_lit(val))
    if op == "gte":  return exp.GTE(this=col, expression=_lit(val))
    if op == "lt":   return exp.LT(this=col, expression=_lit(val))
    if op == "lte":  return exp.LTE(this=col, expression=_lit(val))
    if op == "between":
        lo, hi = val
        return exp.Between(this=col, low=_lit(lo), high=_lit(hi))
    if op == "like":
        # ensure guardrail: LIKE ... ESCAPE '\'
        return exp.Like(this=col, expression=_lit(val), escape=exp.Literal.string("\\"))
    if op == "in":
        seq = list(val) if isinstance(val, (list, tuple, set)) else [val]
        return exp.In(this=col, expressions=[_lit(v) for v in seq])

    return exp.EQ(this=col, expression=_lit(val))

def _flatten_forbidden(forbidden) -> set[str]:
    out = set()
    if not forbidden:
        return out
    if isinstance(forbidden, list):
        out |= {str(x) for x in forbidden if isinstance(x, str)}
        return out
    if isinstance(forbidden, dict):
        if isinstance(forbidden.get("fields"), list):
            out |= {str(x) for x in forbidden["fields"] if isinstance(x, str)}
        for k in list(forbidden.keys()):
            if isinstance(k, str) and "." in k:
                out.add(k)
    return out

def _base_table_columns(base_table: str) -> list[str]:
    """All physical columns of base_table, excluding forbidden fields."""
    bundle = load_all_configs(Path("src/orchestrator/config").resolve())
    fields = (bundle.fields or {}).get(base_table) or {}
    forbidden = _flatten_forbidden(getattr(bundle, "forbidden", None))
    allowed = [c for c in fields.keys() if f"{base_table}.{c}" not in forbidden]
    return allowed

def _tbl_col(key: str) -> Tuple[str, str]:
    if "." not in key:
        raise ValueError(f"Expected qualified name 'table.column', got: {key}")
    t, c = key.split(".", 1)
    return t, c


def _col_expr(qualified_col: str) -> exp.Expression:
    t, c = _tbl_col(qualified_col)
    return exp.column(c, table=t)


def _compile_filter(f: Filter) -> exp.Expression:
    col = _col_expr(f.field)
    op = f.op
    v = f.value

    if op == Operator.IS_NULL:
        return exp.Is(this=col, expression=exp.Null())
    if op == Operator.NOT_NULL:
        return exp.Is(this=col, expression=exp.Not(this=exp.Null()))
    if op == Operator.EQ:
        return exp.EQ(this=col, expression=_lit(v))
    if op == Operator.NE:
        return exp.NEQ(this=col, expression=_lit(v))
    if op == Operator.LT:
        return exp.LT(this=col, expression=_lit(v))
    if op == Operator.LTE:
        return exp.LTE(this=col, expression=_lit(v))
    if op == Operator.GT:
        return exp.GT(this=col, expression=_lit(v))
    if op == Operator.GTE:
        return exp.GTE(this=col, expression=_lit(v))
    if op == Operator.LIKE:
        # Ensure ESCAPE '\\' (Phase 1 requirement)
        return exp.Like(this=col, expression=_lit(v), escape=exp.Literal.string("\\"))
    if op == Operator.IN:
        assert isinstance(v, (list, tuple)) and v, "IN requires a non-empty list"
        return exp.In(this=col, expressions=[_lit(x) for x in v])
    if op == Operator.BETWEEN:
        assert isinstance(v, (list, tuple)) and len(v) == 2, "BETWEEN requires [low, high]"
        return exp.Between(this=col, low=_lit(v[0]), high=_lit(v[1]))
    raise ValueError(f"Unsupported operator: {op}")


def _metric_expr(base_table: str, metric_key: str, spec: MetricSpec) -> Tuple[exp.Expression, str]:
    """Return (expr, alias) for SELECT list."""
    alias = metric_key.split(".", 1)[1] if "." in metric_key else metric_key
    if spec.kind == "count":
        expr = exp.Count(this=exp.Star())
    elif spec.kind == "sum":
        expr = exp.Sum(this=exp.column(spec.column, table=base_table))
    elif spec.kind == "avg":
        expr = exp.Avg(this=exp.column(spec.column, table=base_table))
    else:
        raise ValueError(f"Unsupported metric kind: {spec.kind}")
    return expr.as_(alias), alias


# ---------------- Main compiler ---------------- #

def compile_lqr_to_sql(
    lqr: LogicalQueryRequest,
    *,
    config_dir: Optional[Path] = None,
    dialect: str = "mysql",
) -> str:
    # Load metric specs for this build
    mets = _load_metric_specs(config_dir)

    # Plan LEFT JOINs for cross-table projections via lookup_projections.yaml
    join_plan = {}  # key: join table name -> {"table": str, "on": str, "type": "left"}
    lp = _lookup_for_base(lqr.base_table)
    
    base_table = lqr.base_table

    # detail/star mode detection
    proj_list = list(lqr.projections or [])
    star_requested = any(p == "*" or p == f"{lqr.base_table}.*" for p in proj_list)
    has_metrics = bool(lqr.metrics)
    use_star = (not has_metrics) and (star_requested or not proj_list)

    def _alias_for(qualified: str) -> str:
        # Use the column name as alias; if duplicates arise, prefix with table
        t, c = qualified.split(".", 1) if "." in qualified else (None, qualified)
        return c

    # SELECT projections
    select_items: List[exp.Expression] = []
    group_by_items: List[exp.Expression] = []

    # If star mode: emit all base-table columns (explicitly), excluding forbidden.
    # This behaves like `SELECT base_table.*` but with policy-friendly exclusions.
    if use_star:
        for c in _base_table_columns(lqr.base_table):
            select_items.append(exp.column(c, table=lqr.base_table).as_(c))
    
    # --- Dimensions (aggregate grouping keys) ---
    for d in lqr.dimensions:
        t, c = _tbl_col(d)
        if t != base_table:
            raise ValueError(f"Dimension '{d}' not on base_table '{base_table}'")
        col_e = exp.column(c, table=t)
        select_items.append(col_e.as_(c))  # alias to column name
        group_by_items.append(col_e)

    # --- Projections (detail columns or extra group keys when aggregating) ---
    projection_cols: List[Tuple[str, str]] = []      # base-table (t, c)
    cross_projection_cols: List[Tuple[str, str]] = [] # cross-table (t, c)

    for p in (lqr.projections or []):
        if p == "*" or p == f"{base_table}.*":
            continue  # already handled by star-mode expansion
        t, c = _tbl_col(p)
        if t == base_table:
            # base-table projection; we’ll add to SELECT later (keeping Phase-1 order)
            projection_cols.append((t, c))
        else:
            # cross-table projection: must be whitelisted via lookup_projections.yaml
            meta = lp.get(p)
            if meta and "join" in meta:
                j = meta["join"]
                jtable = j["table"]
                jon = j["on"]           # e.g., "advertisers.dealer_id = dealers.id"
                jtype = (j.get("type") or "left").lower()
                # de-dupe by table name
                join_plan[jtable] = {"table": jtable, "on": jon, "type": jtype}
                # add SELECT item now
                select_items.append(exp.column(c, table=t).as_(c))
                cross_projection_cols.append((t, c))
            else:
                # Defensive fallback (policy should prevent this)
                select_items.append(exp.column(c, table=t).as_(c))
                cross_projection_cols.append((t, c))

    # --- Metrics (aggregates) ---
    metric_aliases: Dict[str, str] = {}
    has_metrics = bool(lqr.metrics)
    if has_metrics:
        for mkey in lqr.metrics:
            spec = mets.get(mkey)
            if not spec:
                # Best-effort inference for common names
                if mkey.endswith(".count_rows"):
                    spec = MetricSpec(kind="count")
                elif ".sum_" in mkey:
                    spec = MetricSpec(kind="sum", column=mkey.split(".sum_", 1)[1])
                elif ".avg_" in mkey:
                    spec = MetricSpec(kind="avg", column=mkey.split(".avg_", 1)[1])
                else:
                    raise ValueError(f"Unknown metric: {mkey}")
            expr, alias = _metric_expr(base_table, mkey, spec)
            select_items.append(expr)
            metric_aliases[mkey] = alias

    # If aggregating and there are projections, include them and group by them
    if has_metrics:
        # base-table projections: add to SELECT and GROUP BY
        for t, c in projection_cols:
            col_e = exp.column(c, table=t)
            select_items.insert(0, col_e.as_(c))  # keep dims/projections before metrics
            group_by_items.append(col_e)
        # cross-table projections: only GROUP BY (already added to SELECT)
        for t, c in cross_projection_cols:
            group_by_items.append(exp.column(c, table=t))

    # If NOT aggregating (detail mode), select projections (and any dims act as projections)
    if not has_metrics:
        # Only add base-table projections; dimensions were already added above
        for t, c in projection_cols:
            select_items.append(exp.column(c, table=t).as_(c))
        group_by_items = []

    # FROM
    from_ = exp.Table(this=exp.to_identifier(base_table))

    # WHERE
    where_e: Optional[exp.Expression] = None
    for f in lqr.filters:
        cond = _compile_filter(f)
        where_e = cond if where_e is None else exp.And(this=where_e, expression=cond)

    # ORDER BY
    order_by_items: List[exp.Ordered] = []
    for ob in lqr.order_by:
        field = ob.field
        # Map metric keys to their aliases if needed
        if field in metric_aliases:
            order_expr: exp.Expression = exp.to_identifier(metric_aliases[field])
        else:
            # Could be a qualified dim/projection "table.col" or an alias; support both
            if "." in field:
                try:
                    _, c = _tbl_col(field)
                    order_expr = exp.to_identifier(c)
                except Exception:
                    order_expr = exp.to_identifier(field)
            else:
                order_expr = exp.to_identifier(field)
        order_by_items.append(exp.Ordered(this=order_expr, desc=(ob.direction == Direction.DESC)))

    # base SELECT
    select_ast = exp.select(*select_items).from_(from_)
    if lqr.distinct:
        select_ast.set("distinct", True)

    # apply WHERE (single pass)
    if where_e is not None:
        select_ast = select_ast.where(where_e)

    # Emit LEFT JOINs for cross-table projections
    for jt, spec in join_plan.items():
        # parse ON condition; if parse fails, build it programmatically
        try:
            on_expr = parse_one(spec["on"])
        except Exception:
            left, right = [s.strip() for s in spec["on"].split("=", 1)]
            on_expr = exp.EQ(this=_col_ref(left), expression=_col_ref(right))

        # Always emit a LEFT JOIN for lookup projections (sqlglot version-compatible)
        try:
            select_ast = select_ast.join(
                exp.Table(this=spec["table"]), on=on_expr, join_type="LEFT"
            )
        except TypeError:
            # fallback for versions that use `kind` instead of `join_type`
            select_ast = select_ast.join(
                exp.Table(this=spec["table"]), on=on_expr, kind="left"
            )

    # GROUP BY / ORDER BY / LIMIT
    if group_by_items:
        select_ast = select_ast.group_by(*group_by_items)
    if order_by_items:
        select_ast = select_ast.order_by(*order_by_items)
    if lqr.limit is not None:
        select_ast = select_ast.limit(lqr.limit)

    # emit + verify
    sql = select_ast.sql(dialect=dialect, pretty=False)
    node = sqlglot.parse_one(sql, read=dialect)
    if not isinstance(node, exp.Select):
        raise ValueError("Compiled SQL is not a SELECT")
    return sql
