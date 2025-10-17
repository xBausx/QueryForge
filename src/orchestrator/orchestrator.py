# src/orchestrator/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .config_loader import load_config
from .errors import OrchestratorError, ErrorCode
from .models import LogicalQueryRequest
from .policy_gate import enforce_policies
from .router_rules import route as route_rules
from .sql_ast_builder import build_select_sql


# Resolve the default config directory relative to this file, not CWD
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def orchestrate(
    nl_input: str,
    *,
    dialect: Optional[str] = None,
    tz: str = "Asia/Manila",
    locale: Optional[str] = None,
    tenant: Optional[str] = None,
    config_path: Optional[str] = None,
    include_trace: bool = True,
    default_limit: int = 100,
) -> Dict[str, Any]:
    """
    Deterministic pipeline:
      NL -> LogicalQueryRequest (router) -> AST SQL (sqlglot) -> Policy Gate
      -> {"query": "..."} | {"error": "[CODE] ..."}
    """
    trace: Dict[str, Any] = {"pipeline": {}, "ctx": {"tz": tz, "dialect": dialect}}

    # -------- Stage 0: Load config (fail fast by design) --------
    try:
        cfg_dir = Path(config_path) if config_path else _DEFAULT_CONFIG_DIR
        config = load_config(str(cfg_dir))
        trace["pipeline"]["config"] = "ok"
    except SystemExit as e:
        msg = str(e).strip()
        payload = {"error": f"[{ErrorCode.SCHEMA_MISSING.value}] {msg}"}
        if include_trace:
            payload["trace"] = {**trace, "stage": "config"}
        return payload
    except Exception as e:
        payload = {"error": f"[{ErrorCode.UNSUPPORTED_OPERATION.value}] Config load error: {e}"}
        if include_trace:
            payload["trace"] = {**trace, "stage": "config"}
        return payload

    # -------- Stage 1: Route NL -> LQR --------
    try:
        routed = route_rules(nl_input or "", config, tz=tz)
        trace["pipeline"]["router"] = routed.get("trace", {"router": "rules"})
        if "error" in routed:
            payload = {"error": routed["error"]}
            if include_trace:
                payload["trace"] = {**trace, "stage": "router"}
            return payload
        lqr_dict = routed.get("lqr") or {}
        lqr = LogicalQueryRequest(**lqr_dict)
    except OrchestratorError as e:
        payload = e.to_public_json()
        if include_trace:
            payload["trace"] = {**trace, "stage": "router"}
        return payload
    except Exception as e:
        payload = {"error": f"[{ErrorCode.UNSUPPORTED_OPERATION.value}] Router error: {e}"}
        if include_trace:
            payload["trace"] = {**trace, "stage": "router"}
        return payload

    # -------- Stage 2: Build AST SQL --------
    try:
        sql = build_select_sql(lqr, config, dialect=dialect)
        trace["pipeline"]["ast"] = "ok"
    except OrchestratorError as e:
        payload = e.to_public_json()
        if include_trace:
            payload["trace"] = {**trace, "stage": "ast", "lqr": lqr.model_dump()}
        return payload
    except Exception as e:
        payload = {"error": f"[{ErrorCode.UNSUPPORTED_OPERATION.value}] AST build error: {e}"}
        if include_trace:
            payload["trace"] = {**trace, "stage": "ast", "lqr": lqr.model_dump()}
        return payload

    # -------- Stage 3: Policy Gate --------
    try:
        final_sql = enforce_policies(
            sql,
            forbidden=getattr(config, "forbidden", []) or [],
            default_limit=default_limit,
            dialect=dialect,
        )
        trace["pipeline"]["policy"] = "ok"
    except OrchestratorError as e:
        payload = e.to_public_json()
        if include_trace:
            payload["trace"] = {**trace, "stage": "policy", "sql_pre_policy": sql}
        return payload
    except Exception as e:
        payload = {"error": f"[{ErrorCode.UNSUPPORTED_OPERATION.value}] Policy error: {e}"}
        if include_trace:
            payload["trace"] = {**trace, "stage": "policy", "sql_pre_policy": sql}
        return payload

    # -------- Success --------
    result = {"query": final_sql}
    if include_trace:
        result["trace"] = trace
    return result


# Optional convenience alias
compile_query = orchestrate


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "top 5 licenses.count by licenses.dealer_id in October 2025"
    out = orchestrate(question, dialect=None, tz="Asia/Manila", include_trace=True)
    if "query" in out:
        print(out["query"])
    else:
        print(out["error"])
