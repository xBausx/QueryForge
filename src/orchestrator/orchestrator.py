from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .router_runtime import DeterministicRouter
from .policy_gate import apply_policy, PolicyError
from .sql_ast_builder import compile_lqr_to_sql


def orchestrate(
    nl: str,
    *,
    dialect: Optional[str] = None,
    tz: Optional[str] = None,
    locale: Optional[str] = None,
    tenant: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase-1 orchestrator: NL -> LQR (router) -> policy -> SQL (AST).
    Returns {"query": "..."} on success or {"error": "[CODE] reason"} on failure.
    """
    try:
        router = DeterministicRouter(config_dir=config_dir)
        lqr = router.route(nl, dialect=dialect, tz=tz, locale=locale, tenant=tenant)
        lqr = apply_policy(lqr, config_dir=config_dir)
        sql = compile_lqr_to_sql(lqr, dialect=dialect)
        return {"query": sql}
    except PolicyError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"[UNEXPECTED] {e}"}
