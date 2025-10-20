# src/orchestrator/cli.py
from __future__ import annotations
import argparse
from .orchestrator import orchestrate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", type=str, help="Natural language question")
    p.add_argument("--dialect", type=str, default=None)
    p.add_argument("--tz", type=str, default="Asia/Manila")
    p.add_argument("--config-dir", type=str, default="src/orchestrator/config")
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()

    out = orchestrate(
        args.question,
        dialect=args.dialect,
        tz=args.tz,
        config_dir=args.config_dir,
        ai_fallback=(not args.no_ai),
        include_trace=args.trace,
    )
    if "query" in out:
        print(out["query"])
    else:
        print(out.get("error", "unknown error"))

if __name__ == "__main__":
    main()
