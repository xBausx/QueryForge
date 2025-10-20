# src/orchestrator/tests/test_policy.py
from __future__ import annotations

import re

from orchestrator.orchestrator import orchestrate


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def test_select_only__no_cte_no_multistmt_no_comments_and_default_limit():
    """
    Policy: generated SQL must be a single SELECT statement (no CTE),
    contain no comments, and apply default LIMIT 100 when not specified.
    """
    nl = "licenses.count by licenses.dealer_id"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql_raw = out["query"]
    sql = _norm(sql_raw)

    # Must begin with SELECT (single-statement)
    assert re.search(r"^\s*select\b", sql, flags=re.I), sql_raw

    # No CTEs (no leading WITH)
    assert not re.search(r"^\s*with\b", sql, flags=re.I), sql_raw

    # No multi-statement trailing SQL
    assert not re.search(r";\s*\S", sql_raw), sql_raw

    # No SQL comments
    assert "--" not in sql_raw and "/*" not in sql_raw and "*/" not in sql_raw, sql_raw

    # Default limit applied
    assert "limit 100" in sql, sql_raw


def test_like_has_escape_enforced__policy_guard():
    """
    Policy: LIKE must include ESCAPE '\\' so %/_ are safely handled.
    """
    nl = "hosts.count by hosts.state where hosts.state like 'CA%'"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert "from hosts" in sql
    assert "group by hosts.state" in sql
    assert "like" in sql and "escape '\\'" in sql, out["query"]


def test_unknown_projection_rejected__schema_missing():
    """
    Guardrail: projections must exist in fields.yaml or lookup_projections.yaml.
    An unknown projection should return [SCHEMA_MISSING].
    """
    nl = "licenses. Show me pineapple"
    out = orchestrate(nl, include_trace=True)

    assert "error" in out, "Expected SCHEMA_MISSING for unknown projection"
    assert "[SCHEMA_MISSING]" in out["error"], out.get("error", "no error text")
