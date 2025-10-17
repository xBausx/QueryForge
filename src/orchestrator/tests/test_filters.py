# src/orchestrator/tests/test_filters.py
from __future__ import annotations

import re

from orchestrator.orchestrator import orchestrate


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def test_like_filter_hosts_state():
    """
    hosts.count by hosts.state with a LIKE filter.
    Expect:
      - FROM hosts
      - GROUP BY hosts.state
      - WHERE hosts.state LIKE 'CA\%' ESCAPE '\'
      - LIMIT present (100 if not specified)
    """
    nl = "hosts.count by hosts.state where hosts.state like 'CA%'"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert "from hosts" in sql
    assert "group by hosts.state" in sql
    assert "where hosts.state like 'ca\\%'" in sql or "where hosts.state like 'ca%'" in sql
    assert "escape" in sql  # ensure ESCAPE clause is present
    assert "limit 100" in sql


def test_contains_synonym_for_like():
    """
    'contains' should map to LIKE safely.
    """
    nl = "hosts.count by hosts.state hosts.state contains 'CA'"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert "from hosts" in sql
    assert "group by hosts.state" in sql
    assert "like" in sql
    assert "escape" in sql
    assert "limit 100" in sql


def test_in_filter_numbers_on_licenses():
    """
    licenses.count grouped by dealer with an IN list of numeric values.
    Expect IN (1, 2, 3) and LIMIT 100 by default.
    """
    nl = "licenses.count by licenses.dealer_id where licenses.dealer_id in (1,2,3)"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert "from licenses" in sql
    assert "group by licenses.dealer_id" in sql
    assert re.search(r"licenses\.dealer_id\s+in\s*\(\s*1\s*,\s*2\s*,\s*3\s*\)", sql)
    assert "limit 100" in sql


def test_in_filter_strings_on_licenses():
    """
    String IN list; ensure quotes are preserved.
    """
    nl = "licenses.count by licenses.dealer_id licenses.player_status in ('online','idle')"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert "from licenses" in sql
    # Depending on serializer, quotes may be single; be flexible with spacing
    assert "licenses.player_status in ('online','idle')" in sql or \
           "licenses.player_status in ('online', 'idle')" in sql
    assert "limit 100" in sql


def test_between_dates_on_licenses():
    """
    BETWEEN range on a date field; no explicit time_range needed.
    Expect WHERE licenses.install_date BETWEEN '2025-10-01' AND '2025-10-15'
    and LIMIT 100.
    """
    nl = "licenses.count licenses.install_date between 2025-10-01 and 2025-10-15"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert "from licenses" in sql
    assert "where licenses.install_date between '2025-10-01' and '2025-10-15'" in sql
    assert "limit 100" in sql


def test_cross_table_filter_is_blocked():
    """
    Phase-1 single-table guard: metric on licenses + filter on hosts.* → error.
    """
    nl = "licenses.count by licenses.dealer_id hosts.state = 'CA'"
    out = orchestrate(nl, include_trace=True)
    assert "error" in out, "Expected SCHEMA_MISMATCH for cross-table filter"
    assert "[SCHEMA_MISMATCH]" in out["error"]
