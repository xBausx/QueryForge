# src/orchestrator/config_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from .models import ResolutionMode


# ===========================
# Pydantic config models
# ===========================

class MetricDef(BaseModel):
    expression: str

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("expression")
    @classmethod
    def expr_nonempty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("metric.expression must be a non-empty string")
        return v.strip()


class Synonyms(BaseModel):
    metrics: Dict[str, List[str]] = Field(default_factory=dict)
    dimensions: Dict[str, List[str]] = Field(default_factory=dict)
    operators: Dict[str, List[str]] = Field(default_factory=dict)


class ResolverMatch(BaseModel):
    field: str = Field(..., description="Fully-qualified name field (e.g., dealers.dealer_name)")
    mode: Optional[ResolutionMode] = Field(default=None, description="ilike_contains | ilike_prefix | exact")
    escape: Optional[str] = Field(default="\\", description="LIKE escape character, default '\\'")

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("field")
    @classmethod
    def fq_field(cls, v: str) -> str:
        v = (v or "").strip()
        if "." not in v:
            raise ValueError("match.field must be 'table.column'")
        return v


class ResolverSpec(BaseModel):
    """
    Declarative resolver used by AST to compile single-statement name->id filters.
    Keyed by target_fk in the config bundle, e.g.:

    resolvers:
      licenses.dealer_id:
        via_table: dealers
        return_column: dealers.dealer_id
        match:
          field: dealers.dealer_name
          mode: ilike_contains
          escape: "\\"
    """
    via_table: str = Field(..., description="Lookup table to search (must be in join_graph relative to base table)")
    return_column: str = Field(..., description="Fully-qualified column returned by the lookup (e.g., dealers.dealer_id)")
    match: ResolverMatch

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("via_table")
    @classmethod
    def nonempty_table(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("via_table must be a non-empty string")
        return v

    @field_validator("return_column")
    @classmethod
    def fq_return_column(cls, v: str) -> str:
        v = (v or "").strip()
        if "." not in v:
            raise ValueError("return_column must be 'table.column'")
        return v


class PatternsConfig(BaseModel):
    """
    Generic holder for router shapes. We don't validate structure here to keep
    it data-driven; the router will validate presence/semantics of fields it uses.
    """
    shapes: Dict[str, Any] = Field(default_factory=dict)


class ConfigBundle(BaseModel):
    """
    Source-of-truth configuration loaded from YAML files.
    Most fields are optional so the loader can run with partial configs during development.
    """
    # Core
    entities: Dict[str, Any] = Field(default_factory=dict)
    fields: Dict[str, str] = Field(default_factory=dict)
    dimensions: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, MetricDef] = Field(default_factory=dict)

    # Joins and safety
    join_graph: Dict[str, List[str]] = Field(default_factory=dict)
    forbidden: List[str] = Field(default_factory=list)

    # Optional, data-driven router helpers
    synonyms: Optional[Synonyms] = None
    resolvers: Dict[str, ResolverSpec] = Field(default_factory=dict)
    patterns: Optional[PatternsConfig] = None

    model_config = ConfigDict(str_strip_whitespace=True)

    # --------- Validators (Pydantic v2) ---------

    @field_validator("fields", "dimensions")
    @classmethod
    def validate_mappings_are_fq(cls, mapping: Dict[str, str]) -> Dict[str, str]:
        """
        Ensure all field/dimension mappings are 'table.column'.
        """
        out: Dict[str, str] = {}
        for k, v in (mapping or {}).items():
            k_s = str(k).strip()
            v_s = str(v).strip()
            if "." not in v_s:
                raise ValueError(f"Mapping for '{k_s}' must be 'table.column', got '{v_s}'")
            out[k_s] = v_s
        return out

    @field_validator("metrics")
    @classmethod
    def ensure_metric_defs(cls, mdefs: Dict[str, MetricDef]) -> Dict[str, MetricDef]:
        """
        Normalize metric keys to str (strip) and keep MetricDef objects.
        """
        out: Dict[str, MetricDef] = {}
        for k, v in (mdefs or {}).items():
            if isinstance(v, dict):
                v = MetricDef(**v)
            if not isinstance(v, MetricDef):
                raise ValueError(f"Invalid metric entry for '{k}': expected dict or MetricDef")
            out[str(k).strip()] = v
        return out

    @field_validator("join_graph")
    @classmethod
    def normalize_join_graph(cls, g: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Accept either:
          table: [neighbor1, neighbor2]
        or
          table: { neighbor1: "id=id", neighbor2: "..." }   (we ignore expressions in Phase 1)
        We only keep the neighbor names (deterministic paths are enforced elsewhere).
        """
        out: Dict[str, List[str]] = {}
        for t, neighbors in (g or {}).items():
            t_s = str(t).strip()
            if isinstance(neighbors, dict):
                out[t_s] = sorted([str(n).strip() for n in neighbors.keys()])
            elif isinstance(neighbors, list):
                out[t_s] = sorted([str(n).strip() for n in neighbors])
            else:
                raise ValueError(f"join_graph entry for '{t_s}' must be list or dict")
        return out


# ===========================
# Loader
# ===========================

_YAML_FILES = {
    "entities": "entities.yaml",
    "fields": "fields.yaml",
    "dimensions": "dimensions.yaml",
    "metrics": "metrics.yaml",
    "join_graph": "join_graph.yaml",
    "forbidden": "forbidden.yaml",
    # Optional router helpers:
    "synonyms": "synonyms.yaml",
    "resolvers": "resolvers.yaml",
    "patterns": "patterns.yaml",
}


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str) -> ConfigBundle:
    """
    Load all config from the provided directory, validating with Pydantic.

    Required (for non-trivial operation):
      - fields.yaml
      - dimensions.yaml
      - metrics.yaml

    Optional:
      - entities.yaml
      - join_graph.yaml
      - forbidden.yaml
      - synonyms.yaml
      - resolvers.yaml
      - patterns.yaml

    On validation error we raise SystemExit with a descriptive message (keeps current orchestrator behavior).
    """
    base = Path(config_dir).resolve()
    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Config directory not found: {base}")

    raw: Dict[str, Any] = {}

    # Load present files; missing optional files are fine.
    for key, fname in _YAML_FILES.items():
        p = base / fname
        if not p.exists():
            continue
        try:
            raw[key] = _load_yaml(p)
        except Exception as e:
            raise SystemExit(f"Failed to parse {fname}: {e}")

    # Coerce models for optional helpers
    if "synonyms" in raw and raw["synonyms"] is not None:
        try:
            raw["synonyms"] = Synonyms(**raw["synonyms"])
        except Exception as e:
            raise SystemExit(f"Invalid synonyms.yaml: {e}")

    if "patterns" in raw and raw["patterns"] is not None:
        try:
            raw["patterns"] = PatternsConfig(**raw["patterns"])
        except Exception as e:
            raise SystemExit(f"Invalid patterns.yaml: {e}")

    if "resolvers" in raw and raw["resolvers"] is not None:
        try:
            # keep dict[str, ResolverSpec]
            raw["resolvers"] = {str(k): ResolverSpec(**v) for k, v in raw["resolvers"].items()}
        except Exception as e:
            raise SystemExit(f"Invalid resolvers.yaml: {e}")

    try:
        bundle = ConfigBundle(**raw)
    except Exception as e:
        # Pretty-print nested validation errors
        try:
            msg = json.loads(e.json())
        except Exception:
            msg = str(e)
        raise SystemExit(f"Config validation error: {msg}")

    return bundle
