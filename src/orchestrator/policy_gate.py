from __future__ import annotations

"""
Phase-1 Policy Gate (detail-mode enabled)

Validates and minimally amends a LogicalQueryRequest (LQR) before SQL compile.
- Enforces single-table semantics
- Verifies dimensions/metrics/projections exist and live on the base table
- Allows **detail mode** (no metrics) if projections are present
- Applies default LIMIT=100 when not provided
- Blocks forbidden fields from `forbidden.yaml`

Errors use Phase-1 codes:
  [MISSING_PARAMETER] [SCHEMA_MISSING] [SCHEMA_MISMATCH] [FORBIDDEN_FIELD]

Usage (PowerShell one‑liner):
  python -c "from orchestrator.router_runtime import DeterministicRouter; from orchestrator.policy_gate import apply_policy; from orchestrator.sql_ast_builder import compile_lqr_to_sql; r=DeterministicRouter(); lqr=r.route('latest advertisers'); lqr=apply_policy(lqr); print(compile_lqr_to_sql(lqr))"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .config_loader import load_all_configs
from .models import LogicalQueryRequest, DEFAULT_LIMIT


@dataclass
class PolicyError(Exception):
    code: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.code}] {self.reason}"


# ---------------- Surfaces ---------------- #

class _Surfaces:
    def __init__(self, *, entities: Dict[str, Any], fields: Dict[str, Any], dimensions: Dict[str, Any], metrics: Dict[str, Any], forbidden: Any) -> None:
        self.tables: Set[str] = set((entities or {}).keys())
        self.dimensions: Dict[str, Any] = dimensions or {}
        self.metrics: Dict[str, Any] = metrics or {}
        self.fields: Dict[str, Dict[str, Any]] = fields or {}
        self.forbidden_fields: Set[str] = _extract_forbidden_fields(forbidden)

    def has_field(self, qualified: str) -> bool:
        if "." not in qualified:
            return False
        t, c = qualified.split(".", 1)
        return t in self.fields and isinstance(self.fields[t], dict) and c in self.fields[t]

def _extract_forbidden_fields(forbidden: Any) -> set[str]:
    out: set[str] = set()
    if forbidden is None:
        return out
    if isinstance(forbidden, list):
        out.update(str(x) for x in forbidden if isinstance(x, str))
        return out
    if isinstance(forbidden, dict):
        if isinstance(forbidden.get("fields"), list):
            out.update(str(x) for x in forbidden["fields"] if isinstance(x, str))
        for k in list(forbidden.keys()):
            if isinstance(k, str) and "." in k:
                out.add(k)
        return out
    return out

def _build_surfaces(config_dir: Optional[Path]) -> _Surfaces:
    bundle = load_all_configs((config_dir or Path("src/orchestrator/config")).resolve())
    return _Surfaces(
        entities=bundle.entities,
        fields=bundle.fields,
        dimensions=bundle.dimensions,
        metrics=bundle.metrics,
        forbidden=bundle.forbidden,
    )


# ---------------- Core checks ---------------- #

def _ensure_base_table(lqr: LogicalQueryRequest, s: _Surfaces) -> None:
    if not lqr.base_table:
        raise PolicyError("MISSING_PARAMETER", "base_table is required")
    if lqr.base_table not in s.tables:
        raise PolicyError("SCHEMA_MISSING", f"Unknown base_table '{lqr.base_table}'")


def _ensure_dims(lqr: LogicalQueryRequest, s: _Surfaces) -> None:
    for d in lqr.dimensions:
        if d not in s.dimensions:
            raise PolicyError("SCHEMA_MISSING", f"Unknown dimension '{d}'")
        if not d.startswith(lqr.base_table + "."):
            raise PolicyError("SCHEMA_MISMATCH", f"Dimension '{d}' is not on base_table '{lqr.base_table}'")
        if d in s.forbidden_fields:
            raise PolicyError("FORBIDDEN_FIELD", f"Dimension '{d}' is forbidden by policy")


def _ensure_metrics_or_projections(lqr: LogicalQueryRequest, s: _Surfaces) -> None:
    has_metrics = bool(lqr.metrics)
    has_projections = bool(lqr.projections)
    if not has_metrics and not has_projections:
        raise PolicyError("MISSING_PARAMETER", "Provide at least one metric or projection")

    # Metrics checks (if present)
    for m in lqr.metrics:
        if m not in s.metrics:
            raise PolicyError("SCHEMA_MISSING", f"Unknown metric '{m}'")
        if not m.startswith(lqr.base_table + "."):
            raise PolicyError("SCHEMA_MISMATCH", f"Metric '{m}' is not on base_table '{lqr.base_table}'")
        if m in s.forbidden_fields:
            raise PolicyError("FORBIDDEN_FIELD", f"Metric '{m}' is forbidden by policy")

    # Projections checks (if present)
    for p in lqr.projections:
        if not s.has_field(p):
            raise PolicyError("SCHEMA_MISSING", f"Unknown projection field '{p}'")
        if not p.startswith(lqr.base_table + "."):
            raise PolicyError("SCHEMA_MISMATCH", f"Projection '{p}' is not on base_table '{lqr.base_table}'")
        if p in s.forbidden_fields:
            raise PolicyError("FORBIDDEN_FIELD", f"Projection '{p}' is forbidden by policy")


def _ensure_filters(lqr: LogicalQueryRequest, s: _Surfaces) -> None:
    for f in lqr.filters:
        field = f.field if isinstance(f.field, str) else None
        if not field:
            continue
        # If fully-qualified, enforce same-table
        if "." in field and not field.startswith(lqr.base_table + "."):
            raise PolicyError("SCHEMA_MISMATCH", f"Filter field '{field}' is not on base_table '{lqr.base_table}'")
        if field in s.forbidden_fields:
            raise PolicyError("FORBIDDEN_FIELD", f"Filter field '{field}' is forbidden by policy")


def _apply_defaults(lqr: LogicalQueryRequest) -> LogicalQueryRequest:
    if lqr.limit is None:
        lqr = lqr.model_copy(update={"limit": lqr.effective_limit(DEFAULT_LIMIT)})
    return lqr


# ---------------- Public API ---------------- #

def apply_policy(lqr: LogicalQueryRequest, *, config_dir: Optional[Path] = None) -> LogicalQueryRequest:
    """Validate and minimally amend the LQR.

    Returns a (possibly updated) LQR or raises PolicyError with a coded message.
    """
    s = _build_surfaces(config_dir)
    _ensure_base_table(lqr, s)
    _ensure_dims(lqr, s)
    _ensure_metrics_or_projections(lqr, s)
    _ensure_filters(lqr, s)
    lqr = _apply_defaults(lqr)
    return lqr
