# src/orchestrator/tests/test_filters.py
from __future__ import annotations

import re
import pytest
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

def test_eq_and_ne_numeric_on_dealer_id():
    """
    Numeric equality/inequality on the same table.
    """
    nl_eq = "licenses.count where licenses.dealer_id = 42"
    out_eq = orchestrate(nl_eq, include_trace=False)
    assert "query" in out_eq, out_eq.get("error", "no error")
    sql_eq = _norm(out_eq["query"])
    assert "from licenses" in sql_eq
    assert "where licenses.dealer_id = 42" in sql_eq
    assert "limit 100" in sql_eq

    nl_ne = "licenses.count where licenses.dealer_id != 42"
    out_ne = orchestrate(nl_ne, include_trace=False)
    assert "query" in out_ne, out_ne.get("error", "no error")
    sql_ne = _norm(out_ne["query"])
    assert "from licenses" in sql_ne
    # Accept either SQL inequality form
    assert ("where licenses.dealer_id != 42" in sql_ne) or ("where licenses.dealer_id <> 42" in sql_ne)
    assert "limit 100" in sql_ne


def test_gt_numeric_on_license_id():
    nl = "licenses.count where licenses.license_id > 1000"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])
    assert "from licenses" in sql
    assert "where licenses.license_id > 1000" in sql
    assert "limit 100" in sql


def test_gte_numeric_on_license_id():
    nl = "licenses.count where licenses.license_id >= 1000"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])
    assert "from licenses" in sql
    assert "where licenses.license_id >= 1000" in sql
    assert "limit 100" in sql


def test_lt_numeric_on_license_id():
    nl = "licenses.count where licenses.license_id < 1000"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])
    assert "from licenses" in sql
    assert "where licenses.license_id < 1000" in sql
    assert "limit 100" in sql


def test_lte_numeric_on_license_id():
    nl = "licenses.count where licenses.license_id <= 1000"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])
    assert "from licenses" in sql
    assert "where licenses.license_id <= 1000" in sql
    assert "limit 100" in sql


def test_string_eq_and_ne_on_player_status():
    """
    String equality/inequality with quoted literals.
    """
    nl_eq = "licenses.count where licenses.player_status = 'online'"
    out_eq = orchestrate(nl_eq, include_trace=False)
    assert "query" in out_eq, out_eq.get("error", "no error")
    sql_eq = _norm(out_eq["query"])
    assert "from licenses" in sql_eq
    assert "where licenses.player_status = 'online'" in sql_eq
    assert "limit 100" in sql_eq

    nl_ne = "licenses.count where licenses.player_status != 'offline'"
    out_ne = orchestrate(nl_ne, include_trace=False)
    assert "query" in out_ne, out_ne.get("error", "no error")
    sql_ne = _norm(out_ne["query"])
    assert "from licenses" in sql_ne
    assert ("where licenses.player_status != 'offline'" in sql_ne) or \
        ("where licenses.player_status <> 'offline'" in sql_ne)
    assert "limit 100" in sql_ne


def test_date_comparisons_ge_and_lt_on_install_date():
    """
    Date comparisons without BETWEEN: >= start AND < end (half-open window pattern).
    """
    nl = "licenses.count where licenses.install_date >= 2025-10-01 and licenses.install_date < 2025-10-15"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])
    assert "from licenses" in sql
    assert "where licenses.install_date >=" in sql and "'2025-10-01'" in sql
    assert "and licenses.install_date <" in sql and "'2025-10-15'" in sql
    assert "limit 100" in sql
