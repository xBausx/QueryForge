from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


CONFIG_FILES: dict[str, str] = {
    "entities": "entities.yaml",
    "fields": "fields.yaml",
    "dimensions": "dimensions.yaml",
    "metrics": "metrics.yaml",
    "join_graph": "join_graph.yaml",
    "forbidden": "forbidden.yaml",
    "resolvers": "resolvers.yaml",
    "lookup_projections": "lookup_projections.yaml",
}


class ConfigBundle(BaseModel):
    """Container model for all YAML configs, validated as mappings.

    For M1 we only validate that each YAML is a mapping (YAML map/dict).
    More detailed schemas will be added in later milestones.
    """
    entities: Dict[str, Any] = Field(default_factory=dict)
    fields: Dict[str, Any] = Field(default_factory=dict)
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    join_graph: Dict[str, Any] = Field(default_factory=dict)
    forbidden: Dict[str, Any] = Field(default_factory=dict)
    resolvers: Dict[str, Any] = Field(default_factory=dict)
    lookup_projections: Dict[str, Any] = Field(default_factory=dict)


class SchemaTypeError(ValueError):
    """Raised when a YAML file is not a mapping."""
    code = "SCHEMA_MISSING"


def _load_yaml_map(path: Path) -> Dict[str, Any]:
    """Load a YAML file and ensure its root is a mapping.

    Returns an empty dict for empty files.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SchemaTypeError(f"{path.name} must be a YAML mapping (got {type(data).__name__})")
    return data


def load_all_configs(config_dir: Path) -> ConfigBundle:
    """Load and validate all required YAML config files from `config_dir`.

    Parameters
    ----------
    config_dir : Path
        Path to the directory containing the YAML config files.

    Returns
    -------
    ConfigBundle
        An instance containing all loaded mappings.

    Raises
    ------
    FileNotFoundError
        If any required YAML file is missing.
    SchemaTypeError
        If any YAML root is not a mapping.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for key, filename in CONFIG_FILES.items():
        fp = config_dir / filename
        if not fp.exists():
            raise FileNotFoundError(f"Missing config file: {fp}")
        results[key] = _load_yaml_map(fp)

    # Pydantic validation ensures mapping types
    return ConfigBundle(**results)


if __name__ == "__main__":
    # Allow quick manual check: python src/orchestrator/config_loader.py
    here = Path(__file__).parent
    bundle = load_all_configs(here / "config")
    print("Loaded config files successfully:")
    for k in CONFIG_FILES:
        print(f"  - {k}")
