# src/orchestrator/cli.py
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .orchestrator import orchestrate


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="queryforge",
        description="AST-Centric Orchestrator: NL → LogicalQueryRequest → Safe SELECT SQL",
    )
    p.add_argument("question", nargs="+", help="Natural language question, e.g. \"top 5 licenses.count by licenses.dealer_id in October 2025\"")
    p.add_argument("--dialect", default=None, help="SQL dialect for serialization (postgres|bigquery|snowflake|...); default: generic")
    p.add_argument("--tz", default="Asia/Manila", help="Timezone for date inference; default: Asia/Manila")
    p.add_argument("--config", default=None, help="Path to YAML config directory (defaults to package config)")
    p.add_argument("--no-trace", action="store_true", help="Do not include trace in output JSON")
    p.add_argument("--limit", type=int, default=100, help="Default LIMIT if none present; default: 100")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    nl = " ".join(args.question).strip()

    out = orchestrate(
        nl,
        dialect=args.dialect or None,
        tz=args.tz,
        config_path=args.config,
        include_trace=not args.no_trace,
        default_limit=args.limit,
    )

    # I/O contract: print JSON to stdout; non-zero exit on error.
    if "error" in out:
        payload = {"error": out["error"]}
        if "trace" in out:
            payload["trace"] = out["trace"]
        sys.stderr.write(json.dumps(payload, indent=2) + "\n")
        return 2

    payload = {"query": out["query"]}
    if "trace" in out:
        payload["trace"] = out["trace"]
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
