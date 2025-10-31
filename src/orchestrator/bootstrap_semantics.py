from __future__ import annotations

"""
Bootstrap dimensions.yaml and metrics.yaml from fields.yaml.

Usage (from project root):
  python -m orchestrator.bootstrap_semantics --config src/orchestrator/config --write

Notes:
- Keeps shapes simple for Phase 1. Router only needs enum surfaces.
- You can safely re-run; it overwrites dimensions.yaml and metrics.yaml.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Any

import yaml

STRING_TYPES = {
    "char", "varchar", "text", "tinytext", "mediumtext", "longtext", "enum", "set",
}
DATETIME_TYPES = {
    "datetime", "timestamp", "date", "time", "year",
}
NUMERIC_TYPES = {
    "tinyint", "smallint", "mediumint", "int", "integer", "bigint",
    "decimal", "dec", "numeric", "fixed",
    "float", "double", "double precision", "real",
}

def _base_type(mysql_type: str) -> str:
    t = mysql_type.strip().lower()
    # strip length/precision, unsigned, etc.
    t = re.sub(r"\s+unsigned\b", "", t)
    t = re.sub(r"\s*\(.*\)", "", t)
    return t

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a mapping")
    return data

def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def build_dimensions(fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    dims: Dict[str, Any] = {}
    for table, cols in fields.items():
        if not isinstance(cols, dict):
            continue
        for col, meta in cols.items():
            if not isinstance(meta, dict):
                continue
            dtype = _base_type(str(meta.get("data_type", "")))
            name = f"{table}.{col}"

            if dtype in STRING_TYPES:
                dims[name] = {"table": table, "column": col, "type": "string"}
            elif dtype in DATETIME_TYPES:
                # Provide common time grains for router to target
                dims[name] = {"table": table, "column": col, "type": "datetime", "grains": ["day", "month", "year"]}
            else:
                # numeric and other types can still be groupable if needed,
                # but we keep it conservative to avoid exploding the surface.
                pass
    return dims

def build_metrics(fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mets: Dict[str, Any] = {}
    for table, cols in fields.items():
        if not isinstance(cols, dict):
            continue

        # Always offer count_rows
        mets[f"{table}.count_rows"] = {"table": table, "agg": "count", "expr": "*"}

        for col, meta in cols.items():
            if not isinstance(meta, dict):
                continue
            dtype = _base_type(str(meta.get("data_type", "")))
            if dtype in NUMERIC_TYPES:
                mets[f"{table}.sum_{col}"] = {"table": table, "agg": "sum", "column": col}
                mets[f"{table}.avg_{col}"] = {"table": table, "agg": "avg", "column": col}
    return mets

def main():
    ap = argparse.ArgumentParser(description="Bootstrap dimensions/metrics from fields.yaml")
    ap.add_argument("--config", type=Path, default=Path("src/orchestrator/config"), help="Config directory")
    ap.add_argument("--write", action="store_true", help="Write files (otherwise just show counts)")
    args = ap.parse_args()

    fields = load_yaml(args.config / "fields.yaml")

    dims = build_dimensions(fields)
    mets = build_metrics(fields)

    print(f"Will generate: {len(dims)} dimensions, {len(mets)} metrics")

    if args.write:
        write_yaml(args.config / "dimensions.yaml", dims)
        write_yaml(args.config / "metrics.yaml", mets)
        print(f"Wrote dimensions.yaml and metrics.yaml in {args.config}")

if __name__ == "__main__":
    main()
