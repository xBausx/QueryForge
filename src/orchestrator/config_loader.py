# src/orchestrator/config_loader.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# =========================
# Pydantic config specs
# =========================

class MetricSpec(BaseModel):
    expression: str


class Synonyms(BaseModel):
    metrics: Dict[str, List[str]] = Field(default_factory=dict)
    dimensions: Dict[str, List[str]] = Field(default_factory=dict)


class PatternsConfig(BaseModel):
    shapes: Dict[str, Any] = Field(default_factory=dict)


class ResolverSpec(BaseModel):
    """
    Declarative name→id resolver for a base-table FK.

    Example YAML (either flat or nested under `resolvers:`):
      licenses.dealer_id:
        via_table: dealers
        return_column: dealers.dealer_id
        match:
          field: dealers.dealer_name
          mode: ilike_contains
          escape: "\\"
    """
    via_table: str
    return_column: str  # dotted: table.column
    match: Dict[str, Any]  # requires "field" (dotted); optional "mode", "escape"

    @model_validator(mode="after")
    def _validate(self) -> "ResolverSpec":
        if "." not in self.return_column:
            raise ValueError("return_column must be table.column")
        mf = self.match.get("field")
        if not mf or "." not in mf:
            raise ValueError("match.field must be table.column")
        return self


class LookupProjectionSpec(BaseModel):
    """
    Safe lookup projection that requires a deterministic LEFT JOIN.
    """
    via_table: str                 # table to join
    select: str                    # dotted: table.column to project
    join_on: List[str]             # ["licenses.dealer_id = dealers.dealer_id", ...]

    @model_validator(mode="after")
    def _validate(self) -> "LookupProjectionSpec":
        if "." not in self.select:
            raise ValueError("select must be table.column")
        if not self.join_on:
            raise ValueError("join_on cannot be empty")
        for j in self.join_on:
            if "=" not in j:
                raise ValueError(f"join_on clause must contain '=': {j}")
            left, right = [p.strip() for p in j.split("=", 1)]
            if "." not in left or "." not in right:
                raise ValueError(f"join_on sides must be table.column: {j}")
        return self


class ConfigBundle(BaseModel):
    entities: Dict[str, str] = Field(default_factory=dict)
    fields: Dict[str, str] = Field(default_factory=dict)
    dimensions: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, MetricSpec] = Field(default_factory=dict)
    join_graph: Dict[str, List[str]] = Field(default_factory=dict)

    synonyms: Optional[Synonyms] = None
    patterns: Optional[PatternsConfig] = None

    # Maps like {"licenses.dealer_id": ResolverSpec(...), ...}
    resolvers: Dict[str, ResolverSpec] = Field(default_factory=dict)

    # Maps like {"licenses.host_name": LookupProjectionSpec(...), ...}
    lookup_projections: Dict[str, LookupProjectionSpec] = Field(default_factory=dict)

    @field_validator("metrics", mode="before")
    @classmethod
    def _metrics_coerce(cls, v):
        """
        Accept either:
          metrics:
            licenses.count: { expression: "COUNT(...)" }
        or:
          metrics:
            licenses.count: "COUNT(...)"
        """
        v = v or {}
        out = {}
        for k, val in v.items():
            if isinstance(val, dict):
                out[k] = val
            else:
                out[k] = {"expression": str(val)}
        return out

    @field_validator("dimensions", "fields", mode="before")
    @classmethod
    def _coerce_flat_maps(cls, v):
        return v or {}

    @field_validator("join_graph", mode="before")
    @classmethod
    def _coerce_join_graph(cls, v):
        return v or {}

    @field_validator("resolvers", mode="before")
    @classmethod
    def _normalize_resolvers(cls, v):
        """
        Accept either:
          { resolvers: { <fk>: {...} } }   # nested
        or:
          { <fk>: {...} }                  # flat
        """
        if not v:
            return {}
        if isinstance(v, dict) and "resolvers" in v and isinstance(v["resolvers"], dict):
            return v["resolvers"]
        return v

    @field_validator("lookup_projections", mode="before")
    @classmethod
    def _normalize_lookup_projections(cls, v):
        """
        Accept either:
          { projections: { <alias>: {...} } }   # nested
        or:
          { <alias>: {...} }                    # flat
        """
        if not v:
            return {}
        if isinstance(v, dict) and "projections" in v and isinstance(v["projections"], dict):
            return v["projections"]
        return v


# =========================
# YAML helpers
# =========================

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


# =========================
# Public loader
# =========================

def load_config(config_dir: str) -> ConfigBundle:
    """
    Load all YAML configuration into a validated ConfigBundle.
    Robust to flat vs nested styles for resolvers and lookup_projections.
    """
    entities = _load_yaml(os.path.join(config_dir, "entities.yaml"))
    fields = _load_yaml(os.path.join(config_dir, "fields.yaml"))
    dimensions = _load_yaml(os.path.join(config_dir, "dimensions.yaml"))
    metrics = _load_yaml(os.path.join(config_dir, "metrics.yaml"))
    join_graph = _load_yaml(os.path.join(config_dir, "join_graph.yaml"))

    synonyms_raw = _load_yaml(os.path.join(config_dir, "synonyms.yaml"))
    patterns_raw = _load_yaml(os.path.join(config_dir, "patterns.yaml"))

    resolvers_raw = _load_yaml(os.path.join(config_dir, "resolvers.yaml"))
    lookup_proj_raw = _load_yaml(os.path.join(config_dir, "lookup_projections.yaml"))

    # Build bundle with Pydantic coercion/validation
    cfg = ConfigBundle(
        entities=entities or {},
        fields=fields or {},
        dimensions=dimensions or {},
        metrics=metrics or {},
        join_graph=join_graph or {},
        synonyms=Synonyms(**(synonyms_raw or {})) if synonyms_raw else None,
        patterns=PatternsConfig(**(patterns_raw or {})) if patterns_raw else None,
        resolvers=(resolvers_raw or {}),
        lookup_projections=(lookup_proj_raw or {}),
    )

    # Explicitly ensure dict[str, ResolverSpec] typing (safer than relying on model coercion only)
    if cfg.resolvers:
        cfg.resolvers = {k: (v if isinstance(v, ResolverSpec) else ResolverSpec(**v))
                         for k, v in cfg.resolvers.items()}

    if cfg.lookup_projections:
        cfg.lookup_projections = {k: (v if isinstance(v, LookupProjectionSpec) else LookupProjectionSpec(**v))
                                  for k, v in cfg.lookup_projections.items()}

    # Metrics already coerced by the field_validator above
    if cfg.metrics:
        cfg.metrics = {k: (v if isinstance(v, MetricSpec) else MetricSpec(**v))
                       for k, v in cfg.metrics.items()}

    return cfg
