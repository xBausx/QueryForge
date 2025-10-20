# src/orchestrator/tests/test_golden.py
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from orchestrator.orchestrator import orchestrate


def _normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def test_top5_licenses_by_dealer_oct_2025():
    """
    Expect:
      - FROM licenses
      - metric COUNT(licenses.license_id) aliased as licenses.count
      - GROUP BY licenses.dealer_id
      - WHERE licenses.date_created >= '2025-10-01' AND < '2025-11-01'
      - ORDER BY licenses.count DESC
      - LIMIT 5
    """
    nl = "top 5 licenses.count by licenses.dealer_id in October 2025"
    out = orchestrate(nl, include_trace=True)
    assert "query" in out, out.get("error", "no error")
    sql = _normalize(out["query"])

    assert "from licenses" in sql
    # alias with dot is typically quoted; accept with or without quotes (dialect-agnostic)
    assert re.search(r"sum\(licenses\.amount\)|count\(licenses\.license_id\)", sql), sql
    assert "group by licenses.dealer_id" in sql
    assert "where licenses.date_created >=" in sql and "'2025-10-01'" in sql
    assert "and licenses.date_created <" in sql and "'2025-11-01'" in sql
    # order by the metric alias (quoted or not)
    assert re.search(r'order by ("?licenses\.count"?|count\(licenses\.license_id\)) desc', sql), sql
    assert "limit 5" in sql


def test_top_hyphen_accepted():
    """
    Accept 'top-10' and 'top10' as rankers.
    """
    for phr in ["top-10", "top10"]:
        nl = f"{phr} licenses.count by licenses.dealer_id in October 2025"
        out = orchestrate(nl, include_trace=False)
        assert "query" in out, out.get("error", "no error")
        assert "limit 10" in _normalize(out["query"])


def test_default_limit_when_not_specified():
    """
    No explicit rank/limit -> LIMIT 100 (policy or AST default).
    """
    nl = "licenses.count by licenses.dealer_id"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    assert "limit 100" in _normalize(out["query"])


def test_cross_table_mix_is_blocked():
    """
    Mixing tables in Phase 1 should yield SCHEMA_MISMATCH before AST.
    """
    nl = "licenses.count by hosts.state in October 2025"
    out = orchestrate(nl, include_trace=True)
    assert "error" in out, "Expected an error for cross-table mix"
    assert "[SCHEMA_MISMATCH]" in out["error"]


def test_october_1_without_year_uses_current_year_asia_manila():
    """
    'October 1' should infer current year in Asia/Manila, producing a one-day window.
    """
    now = datetime.now(ZoneInfo("Asia/Manila"))
    y = now.year
    nl = "licenses.count by licenses.dealer_id on October 1"
    out = orchestrate(nl, tz="Asia/Manila", include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _normalize(out["query"])
    assert f"'{y}-10-01'" in sql
    assert f"'{y}-10-02'" in sql
    assert "where licenses.date_created >=" in sql
    assert "and licenses.date_created <" in sql


def test_detail__licenses_of_dealer_nebula__show_host_license_hours():
    """
    Golden test (detail mode + resolvers)

    NL: "licenses of dealer Nebula. Show me the host name, license id and business hours"

    Expect:
        - FROM licenses
        - LEFT JOIN hosts (via host_id), LEFT JOIN dealers (via dealer_id)
        - EXISTS resolver on dealers with LOWER(...) LIKE LOWER('%Nebula%') ESCAPE '\\'
        - Projections include licenses.host_name, licenses.license_id, licenses.business_hours
        - LIMIT 100 (default)
    """
    nl = "licenses of dealer Nebula. Show me the host name, license id and business hours"
    out = orchestrate(nl, include_trace=True)
    assert "query" in out, out.get("error", "no error")
    sql = _normalize(out["query"])

    # Base table
    assert "from licenses" in sql

    # Safe lookup joins (detail mode expects deterministic LEFT JOINs)
    assert "left join hosts" in sql
    assert "left join dealers" in sql

    # Resolver EXISTS on dealer name with ESCAPE '\\'
    exists_pattern = (
        r"exists\s*\(\s*select\s+1\s+from\s+dealers.*?"
        r"lower\(dealers\.dealer_name\)\s+like\s+lower\('.*nebula.*'\)\s+escape\s+'\\\\'"
    )
    assert re.search(exists_pattern, sql, flags=re.I | re.S), out["query"]

    # Projections (aliases under base table namespace)
    assert re.search(r'("licenses\.host_name"|licenses\.host_name)', sql), out["query"]
    assert re.search(r'("licenses\.license_id"|licenses\.license_id)', sql), out["query"]
    assert re.search(r'("licenses\.business_hours"|licenses\.business_hours)', sql), out["query"]

    # Default LIMIT 100
    assert "limit 100" in sql


def test_aggregate__top5_licenses_by_dealer_nebula_in_oct_2025():
    """
    Golden test (aggregate + time window + resolver)

    NL: "top 5 licenses by dealer Nebula in October 2025"

    Expect:
        - FROM licenses
        - Metric licenses.count (or COUNT(licenses.license_id))
        - GROUP BY licenses.dealer_id
        - Date filter: >= '2025-10-01' AND < '2025-11-01'
            (uses licenses.date_created if available, else licenses.install_date)
        - ORDER BY licenses.count DESC
        - EXISTS (…) resolver on dealer name
        - LIMIT 5
    """
    nl = "top 5 licenses by dealer Nebula in October 2025"
    out = orchestrate(nl, include_trace=True)
    assert "query" in out, out.get("error", "no error")
    sql = _normalize(out["query"])

    assert "from licenses" in sql
    # metric alias or raw COUNT(..)
    assert re.search(r'("licenses\.count"|count\(licenses\.license_id\))', sql), out["query"]

    # group by dealer_id
    assert "group by licenses.dealer_id" in sql

    # date filter on date_created (preferred) OR install_date (fallback), half-open window
    date_created_ok = (
        "where licenses.date_created >=" in sql and "'2025-10-01'" in sql and
        "and licenses.date_created <" in sql and "'2025-11-01'" in sql
    )
    install_date_ok = (
        "where licenses.install_date >=" in sql and "'2025-10-01'" in sql and
        "and licenses.install_date <" in sql and "'2025-11-01'" in sql
    )
    assert date_created_ok or install_date_ok, out["query"]

    # resolver EXISTS with LIKE ... ESCAPE '\\'
    exists_pattern = (
        r"exists\s*\(\s*select\s+1\s+from\s+dealers.*?"
        r"lower\(dealers\.dealer_name\)\s+like\s+lower\('.*nebula.*'\)\s+escape\s+'\\\\'"
    )
    assert re.search(exists_pattern, sql, flags=re.I | re.S), out["query"]

    # order + limit
    assert re.search(r'order by ("?licenses\.count"?|count\(licenses\.license_id\)) desc', sql), out["query"]
    assert "limit 5" in sql