"""Pydantic models (LQR, Filter, OrderBy, enums).

This file will be provided in the next step.
"""
# Placeholder to satisfy imports in early scaffolding
from __future__ import annotations

"""
Pydantic v2 models for the QueryForge Phase 1 Logical Query Request (LQR).

Scope (Phase 1):
- Exactly one safe SELECT built from a single base table.
- Dimensions (group-by keys) and Metrics (aggregations) are named references validated elsewhere.
- No arbitrary SQL strings here—these are semantic intents that the AST builder will compile.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# -----------------------------
# Shared enums / constants
# -----------------------------

class Operator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    LIKE = "like"
    IN = "in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class Direction(str, Enum):
    ASC = "asc"
    DESC = "desc"


class Resolution(str, Enum):
    """Time bucket resolutions used by the router when normalizing date phrases."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


# Scalar values that can appear in filters.
Scalar = Union[str, int, float, bool, date, datetime]


DEFAULT_LIMIT: int = 100  # Policy gate may apply this if LQR.limit is None


# -----------------------------
# Leaf models
# -----------------------------

class Filter(BaseModel):
    """
    A semantic filter on a single field (dimension or metric-projected field).

    Constraints by operator:
      - IS_NULL / NOT_NULL: value must be None
      - LIKE: value must be a str (AST will emit `... LIKE ? ESCAPE '\\'`)
      - IN: value must be a non-empty sequence (list/tuple) of scalars
      - BETWEEN: value must be a 2-tuple/list (low, high)
      - All others: value must be a scalar
    """
    field: str = Field(..., min_length=1)
    op: Operator
    value: Optional[Union[Scalar, Sequence[Scalar], Tuple[Scalar, Scalar]]] = None

    @field_validator("value")
    @classmethod
    def validate_value_shape(cls, v: Any, info):
        op: Operator = info.data.get("op")
        if op in (Operator.IS_NULL, Operator.NOT_NULL):
            if v is not None:
                raise ValueError(f"value must be None for operator {op}")
            return None

        if op == Operator.LIKE:
            if not isinstance(v, str):
                raise ValueError("LIKE requires a string value")
            return v

        if op == Operator.IN:
            if not isinstance(v, (list, tuple)):
                raise ValueError("IN requires a list or tuple of values")
            if len(v) == 0:
                raise ValueError("IN requires at least one value")
            return list(v)

        if op == Operator.BETWEEN:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError("BETWEEN requires exactly two values [low, high]")
            low, high = v[0], v[1]
            # Basic sanity: both ends should be scalars
            for side in (low, high):
                if not isinstance(side, (str, int, float, bool, date, datetime)):
                    raise ValueError("BETWEEN bounds must be scalar values")
            return (low, high)

        # default: scalar required
        if isinstance(v, (list, tuple)):
            raise ValueError(f"{op} expects a single scalar value, not a list/tuple")
        if not isinstance(v, (str, int, float, bool, date, datetime)):
            raise ValueError("Filter value must be a scalar (str/int/float/bool/date/datetime)")
        return v


class OrderBy(BaseModel):
    """Ordering over a named projection (dimension or metric alias)."""
    field: str = Field(..., min_length=1)
    direction: Direction = Direction.DESC
    # The AST builder may support NULLS FIRST/LAST where dialect allows:
    nulls: Optional[str] = Field(default=None, pattern=r"^(first|last)$")


# -----------------------------
# Core LQR
# -----------------------------

class LogicalQueryRequest(BaseModel):
    """
    The router emits this structure; the AST builder compiles it into a SELECT.

    Important invariants for Phase 1:
      - Exactly one `base_table`.
      - Dimensions and metrics reference known names from the governed layer.
      - No free-form SQL expressions here.
    """
    base_table: str = Field(..., min_length=1)

    # Names are validated against config by the router/policy; models just hold them.
    dimensions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)

    # Optional additional projections (rare in Phase 1; prefer dimensions/metrics).
    projections: List[str] = Field(default_factory=list)

    filters: List[Filter] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)

    limit: Optional[int] = None
    distinct: bool = False

    # Optional routing context (not used by AST logic directly but passed through)
    resolution: Optional[Resolution] = None
    dialect: Optional[str] = None
    tz: Optional[str] = None
    locale: Optional[str] = None
    tenant: Optional[str] = None

    @field_validator("dimensions", "metrics", "projections")
    @classmethod
    def no_empty_names(cls, v: List[str]) -> List[str]:
        for name in v:
            if not isinstance(name, str) or name.strip() == "":
                raise ValueError("Names must be non-empty strings")
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("limit must be a positive integer")
        # Upper cap enforcement is handled by the policy gate; we only check >0 here.
        return v

    @model_validator(mode="after")
    def validate_single_base_table(self) -> "LogicalQueryRequest":
        # For Phase 1 we only store the base table string here.
        # Join allowlist & resolver wiring happen later in the AST/policy stages.
        if "." in self.base_table:
            # Allow schema-qualified names like "public.table"; still a single base table.
            pass
        return self

    # ------------ Convenience helpers (not serialized guards) ------------

    @property
    def select_order(self) -> List[str]:
        """
        Returns a deterministic select list ordering:
        - dimensions first (in provided order),
        - then metrics (in provided order),
        - then any extra projections (rare).
        """
        return list(dict.fromkeys(self.dimensions + self.metrics + self.projections))

    @property
    def has_aggregates(self) -> bool:
        """True if the query includes metrics (aggregations)."""
        return len(self.metrics) > 0

    def effective_limit(self, default_limit: int = DEFAULT_LIMIT) -> int:
        """Resolve the query limit with a default fallback (policy may call this)."""
        return self.limit if self.limit is not None else default_limit
