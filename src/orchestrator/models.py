# src/orchestrator/models.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


# -----------------------------
# Enums
# -----------------------------

class Operator(str, Enum):
    """Supported filter operators (string values match what the AST expects)."""
    eq = "eq"
    neq = "neq"
    gt = "gt"
    lt = "lt"
    gte = "gte"
    lte = "lte"
    between = "between"
    in_ = "in"       # note: python-safe name, string value is "in"
    like = "like"


class Direction(str, Enum):
    asc = "asc"
    desc = "desc"


class ResolutionMode(str, Enum):
    """How a human-readable name should be matched when resolving to an ID."""
    ilike_contains = "ilike_contains"   # %value%
    ilike_prefix = "ilike_prefix"       # value%
    exact = "exact"                     # = value


# -----------------------------
# Data models
# -----------------------------

class OrderBy(BaseModel):
    field: str
    direction: Direction

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("field")
    @classmethod
    def non_empty_field(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("order_by.field must be a non-empty string")
        return v.strip()


class Filter(BaseModel):
    field: str
    op: Operator
    value: Any

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("field")
    @classmethod
    def field_must_be_nonempty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("field must be a non-empty string")
        return v.strip()

    @field_validator("value")
    @classmethod
    def value_must_not_be_none(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("value must not be None")
        return v

    @model_validator(mode="after")
    def validate_shapes(self) -> "Filter":
        if self.op == Operator.between:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError("between expects a two-element array [start, end]")
        if self.op == Operator.in_ and not isinstance(self.value, (list, tuple)):
            raise ValueError("in expects an array of values")
        if self.op == Operator.like and not isinstance(self.value, str):
            raise ValueError("like expects a string value")
        return self


class Resolution(BaseModel):
    """
    Declarative name->id resolver instruction.
    Example:
      { "target_fk": "licenses.dealer_id", "value": "nebula", "mode": "ilike_contains" }
    """
    target_fk: str = Field(..., description="Canonical FK field on the base table (e.g., 'licenses.dealer_id').")
    value: str = Field(..., min_length=1, description="User-provided name or text to resolve.")
    mode: Optional[ResolutionMode] = Field(
        default=None,
        description="Match mode; if None, default is taken from resolvers.yaml."
    )

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("target_fk")
    @classmethod
    def fk_must_be_canonical(cls, v: str) -> str:
        v = (v or "").strip()
        if "." not in v:
            raise ValueError("target_fk must be 'table.column'")
        return v

    @field_validator("value")
    @classmethod
    def value_non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("resolution.value must be a non-empty string")
        return v


class LogicalQueryRequest(BaseModel):
    """
    Authoritative schema produced by the router and consumed by the AST builder.
    """
    metrics: List[str]
    dimensions: List[str] = Field(default_factory=list)
    filters: List[Filter] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: Optional[int] = None
    time_range: Optional[Dict[str, str]] = None  # {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}
    # NEW: optional resolver instructions for name->id lookups (single-statement via EXISTS/IN in AST)
    resolutions: List[Resolution] = Field(default_factory=list)

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("metrics")
    @classmethod
    def at_least_one_metric(cls, v: List[str]) -> List[str]:
        if not v or not isinstance(v, list):
            raise ValueError("metrics must be a non-empty array of canonical metric names")
        cleaned = [m.strip() for m in v if isinstance(m, str) and m.strip()]
        if not cleaned:
            raise ValueError("metrics must contain at least one non-empty string")
        return cleaned

    @field_validator("dimensions")
    @classmethod
    def clean_dimensions(cls, v: List[str]) -> List[str]:
        return [d.strip() for d in (v or []) if isinstance(d, str) and d.strip()]

    @field_validator("limit")
    @classmethod
    def positive_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not isinstance(v, int) or v <= 0:
            raise ValueError("limit must be a positive integer")
        return v

    @model_validator(mode="after")
    def validate_time_range_shape(self) -> "LogicalQueryRequest":
        if self.time_range is None:
            return self
        if not isinstance(self.time_range, dict):
            raise ValueError("time_range must be an object with 'start' and 'end'")
        start = self.time_range.get("start")
        end = self.time_range.get("end")
        if not start or not end or not isinstance(start, str) or not isinstance(end, str):
            raise ValueError("time_range requires string 'start' and 'end' keys")
        return self


__all__ = [
    "Operator",
    "Direction",
    "ResolutionMode",
    "OrderBy",
    "Filter",
    "Resolution",
    "LogicalQueryRequest",
]
