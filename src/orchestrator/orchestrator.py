# src/orchestrator/orchestrator.py
from __future__ import annotations

import os

from typing import Any, Dict, Optional

from .config_loader import load_config
from .sql_ast_builder import build_select_sql
from .policy_gate import enforce_policies
from .errors import OrchestratorError, ErrorCode
from .models import LogicalQueryRequest  # <-- build a model, not a dict

# Routers
from .router_runtime import route as route_runtime
from .router_rules import route as route_rules  # safety fallback
# Optional AI router (stub unless you implement it)
try:
    from .router_ai import translate as route_ai  # type: ignore
except Exception:
    route_ai = None


def orchestrate(
    nl_text: str,
    *,
    dialect: Optional[str] = None,
    tz: str = "Asia/Manila",
    include_trace: bool = False,
    config_dir: str = "src/orchestrator/config",
    ai_fallback: bool = True,
) -> Dict[str, Any]:
    """
    NL → LQR → AST → SQL (+ policy). Deterministic runtime router first; AI router optional; rules router fallback.
    Returns either {"query": "<SELECT …>"} or {"error": "[CODE] …"}.
    """
    try:
        cfg = load_config(config_dir)
    except SystemExit as e:
        return {"error": f"[SCHEMA_MISSING] {e}"}

    traces: Dict[str, Any] = {}

    # 1) Runtime router (tokenizer + boundaries)
    r = route_runtime(nl_text, cfg, tz=tz)
    if "error" in r:
        traces["runtime"] = r.get("trace")
        # 2) AI fallback (optional)
        if ai_fallback and route_ai is not None:
            provider = os.getenv("ROUTER_PROVIDER", "openai")
            ar = route_ai(nl_text, cfg, tz=tz, provider=provider)  # <-- pass provider
            if "lqr" in ar:
                r = ar
            else:
                traces["ai"] = ar.get("trace")
        # 3) Legacy rules router as last resort
        if "error" in r:
            rr = route_rules(nl_text, cfg, tz=tz)
            if "lqr" in rr:
                r = rr
            else:
                traces["rules"] = rr.get("trace")
                return {
                    "error": rr.get("error", r.get("error", "[AMBIGUOUS_REQUEST] Unable to route")),
                    "stage": "router",
                    "trace": traces if include_trace else None,
                }

    try:
        lqr_payload = r["lqr"]
        lqr = lqr_payload if hasattr(lqr_payload, "metrics") else LogicalQueryRequest(**lqr_payload)
        sql = build_select_sql(lqr, cfg, dialect=dialect)

        # Support either enforce_policies(sql, cfg) or enforce_policies(sql)
        try:
            safe_sql = enforce_policies(sql, cfg)  # older signature
        except TypeError:
            safe_sql = enforce_policies(sql)       # current signature

        out: Dict[str, Any] = {"query": safe_sql}
        if include_trace:
            out["trace"] = {**traces, "router": r.get("trace")}
        return out
    except OrchestratorError as e:
        return {"error": f"[{e.code}] {e}", "stage": "policy"}
    except Exception as e:
        return {"error": f"[UNSUPPORTED_OPERATION] AST build error: {e}", "stage": "ast"}
