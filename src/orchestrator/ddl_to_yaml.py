from __future__ import annotations

"""
SQL AST Builder (Phase 1 • Detail Mode enabled)

Compiles a LogicalQueryRequest (LQR) into a single-table SELECT using SQLGlot.
- Programmatic AST only (no string concatenation)
- Re-parses final SQL to verify the first token is SELECT
- MySQL dialect by default

Now supports two modes:
1) **Detail mode** – when `metrics` is empty and `projections` is non-empty.
   Emits a plain SELECT of the requested columns (no GROUP BY).
2) **Aggregate mode** – when `metrics` is non-empty. If `projections` are also
   present, they are included and **auto-GROUPed** along with any dimensions.

Public API:
    compile_lqr_to_sql(lqr: LogicalQueryRequest, *, config_dir: Path|None=None, dialect: str="mysql") -> str
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

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

def _tbl_col(key: str) -> Tuple[str, str]:
    if "." not in key:
        raise ValueError(f"Expected qualified name 'table.column', got: {key}")
    t, c = key.split(".", 1)
    return t, c


def _col_expr(qualified_col: str) -> exp.Expression:
    t, c = _tbl_col(qualified_col)
    return exp.column(c, table=t)


def _lit(v: Any) -> exp.Expression:
    if v is None:
        return exp.Null()
    if isinstance(v, bool):
        return exp.Boolean(this=v)
    if isinstance(v, (int, float)):
        return exp.Literal.number(v)
    # strings and dates as string literals; DB will cast appropriately
    return exp.Literal.string(str(v))


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
        # Ensure ESCAPE '\' (Phase 1 requirement)
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

    base_table = lqr.base_table

    # SELECT projections
    select_items: List[exp.Expression] = []
    group_by_items: List[exp.Expression] = []

    # --- Dimensions (aggregate grouping keys) ---
    for d in lqr.dimensions:
        t, c = _tbl_col(d)
        if t != base_table:
            raise ValueError(f"Dimension '{d}' not on base_table '{base_table}'")
        col_e = exp.column(c, table=t)
        select_items.append(col_e.as_(c))  # alias to column name
        group_by_items.append(col_e)

    # --- Projections (detail columns or extra group keys when aggregating) ---
    projection_cols: List[Tuple[str, str]] = []  # (table, col)
    for p in lqr.projections:
        t, c = _tbl_col(p)
        if t != base_table:
            raise ValueError(f"Projection '{p}' not on base_table '{base_table}'")
        projection_cols.append((t, c))

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
    if has_metrics and projection_cols:
        for t, c in projection_cols:
            col_e = exp.column(c, table=t)
            select_items.insert(0, col_e.as_(c))  # keep dims first-ish
            group_by_items.append(col_e)

    # If NOT aggregating (detail mode), select projections (and any dims act as projections)
    if not has_metrics:
        # Treat dimensions as additional projections when no metrics are present
        # (detail mode with explicit dims is equivalent to selecting those cols).
        dim_names = [d.split(".", 1)[1] for d in lqr.dimensions]
        # add projections first for stable order
        for t, c in projection_cols:
            select_items.append(exp.column(c, table=t).as_(c))
        for d in lqr.dimensions:
            t, c = _tbl_col(d)
            select_items.append(exp.column(c, table=t).as_(c))
        # Ensure we don't accidentally add a GROUP BY in detail mode
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

    # Build SELECT
    select = exp.select(*select_items).from_(from_)
    if lqr.distinct:
        select.set("distinct", True)
    if where_e is not None:
        select = select.where(where_e)
    if group_by_items:
        select = select.group_by(*group_by_items)
    if order_by_items:
        select = select.order_by(*order_by_items)
    if lqr.limit is not None:
        select = select.limit(lqr.limit)

    # Emit SQL and verify
    sql = select.sql(dialect=dialect, pretty=False)
    node = sqlglot.parse_one(sql, read=dialect)
    if not isinstance(node, exp.Select):
        raise ValueError("Compiled SQL is not a SELECT")

    return sql
