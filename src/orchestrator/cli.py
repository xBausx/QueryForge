from __future__ import annotations

"""
QueryForge Orchestrator CLI (Phase 1 utilities)

PowerShell/Wave-friendly one‑liners:
  python -m orchestrator.cli --validate-config
  python -m orchestrator.cli --route "top 5 activities last 30 days" --json
  python -m orchestrator.cli --compile "top 5 advertisers this month by name"

Exit codes:
  0 = success, 1 = router/policy/compile failure, 2 = schema/config failure
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from .config_loader import load_all_configs
from .router_runtime import DeterministicRouter, RouterError
from .policy_gate import apply_policy, PolicyError
from .sql_ast_builder import compile_lqr_to_sql


# ---------------- summarize config ---------------- #

def summarize_bundle(bundle) -> Dict[str, Any]:
    entities = bundle.entities or {}
    fields = bundle.fields or {}
    join_graph = bundle.join_graph or {}
    dims = bundle.dimensions or {}
    mets = bundle.metrics or {}
    forb = bundle.forbidden or {}
    resolv = bundle.resolvers or {}
    lookups = bundle.lookup_projections or {}

    entity_only = sorted(set(entities) - set(fields))
    fields_only = sorted(set(fields) - set(entities))

    pk_missing = [
        t for t, meta in (entities or {}).items()
        if not isinstance(meta, dict) or not meta.get("pk")
    ]

    col_counts = {
        t: (len(cols) if isinstance(cols, dict) else 0)
        for t, cols in (fields or {}).items()
    }

    return {
        "counts": {
            "entities": len(entities),
            "tables_with_fields": len(fields),
            "total_columns": sum(col_counts.values()),
            "join_edges": len(join_graph),
            "dimensions": len(dims),
            "metrics": len(mets),
            "forbidden_rules": len(forb),
            "resolvers": len(resolv),
            "lookup_projections": len(lookups),
        },
        "warnings": {
            "entities_without_fields": entity_only,
            "fields_without_entities": fields_only,
            "entities_missing_pk": pk_missing,
        },
        "column_counts_per_table": col_counts,
    }


def cmd_validate_config(config_dir: Path, as_json: bool) -> int:
    try:
        bundle = load_all_configs(config_dir)
    except FileNotFoundError as e:
        print(f"[SCHEMA_MISSING] {e}")
        return 2
    except Exception as e:
        print(f"[SCHEMA_MISSING] {e}")
        return 2

    summary = summarize_bundle(bundle)
    if as_json:
        print(json.dumps(summary, indent=2))
    else:
        counts = summary["counts"]
        warns = summary["warnings"]
        print("Config summary")
        print("--------------")
        for k, v in counts.items():
            print(f"{k:>20}: {v}")
        if any(warns.values()):
            print("Warnings")
            print("--------")
            if warns["entities_without_fields"]:
                print("entities_without_fields:", ", ".join(warns["entities_without_fields"]))
            if warns["fields_without_entities"]:
                print("fields_without_entities:", ", ".join(warns["fields_without_entities"]))
            if warns["entities_missing_pk"]:
                print("entities_missing_pk:", ", ".join(warns["entities_missing_pk"]))
    return 0


# ---------------- routing ---------------- #

def cmd_route(question: str, config_dir: Path, as_json: bool) -> int:
    try:
        r = DeterministicRouter(config_dir=config_dir)
        lqr = r.route(question)
    except RouterError as e:
        print(json.dumps({"error": f"[{e.code}] {e.reason}"}))
        return 1
    except Exception as e:
        print(json.dumps({"error": f"[UNEXPECTED] {e}"}))
        return 1

    payload = {"lqr": lqr.model_dump()}
    print(json.dumps(payload, indent=2) if as_json else json.dumps(payload))
    return 0


# ---------------- compile ---------------- #

def cmd_compile(question: str, config_dir: Path, as_json: bool, dialect: str) -> int:
    try:
        r = DeterministicRouter(config_dir=config_dir)
        lqr = r.route(question)
        lqr = apply_policy(lqr, config_dir=config_dir)
        sql = compile_lqr_to_sql(lqr, config_dir=config_dir, dialect=dialect)
    except (RouterError, PolicyError) as e:
        print(json.dumps({"error": str(e)}))
        return 1
    except Exception as e:
        print(json.dumps({"error": f"[UNEXPECTED] {e}"}))
        return 1

    # Per contract: Success → {"query":"<SELECT …>"}
    print(json.dumps({"query": sql}))
    return 0


# ---------------- main ---------------- #

def main():
    ap = argparse.ArgumentParser(description="QueryForge Orchestrator CLI")
    ap.add_argument("--validate-config", action="store_true", help="Load YAML configs and print a summary")
    ap.add_argument("--route", type=str, default=None, help="Natural language question to route to an LQR")
    ap.add_argument("--compile", type=str, default=None, help="Natural language question to compile to SQL")
    ap.add_argument("--dialect", type=str, default="mysql", help="SQL dialect (default: mysql)")
    ap.add_argument("--config", type=Path, default=Path("src/orchestrator/config"), help="Config directory path")
    ap.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output with indentation where applicable")
    args = ap.parse_args()

    if args.validate_config:
        raise SystemExit(cmd_validate_config(args.config, args.as_json))

    if args.route:
        raise SystemExit(cmd_route(args.route, args.config, args.as_json))

    if args.compile:
        raise SystemExit(cmd_compile(args.compile, args.config, args.as_json, args.dialect))

    ap.print_help()


if __name__ == "__main__":
    main()
