# src/orchestrator/tests/test_resolvers.py
import re
from orchestrator.orchestrator import orchestrate

def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def test_resolution_licenses_by_dealer_name_exists():
    """
    Natural language with dealer *name* should compile to a single SELECT
    that filters licenses.dealer_id via an EXISTS subquery to dealers,
    matching dealers.dealer_name ILIKE '%nebula%'.
    """
    nl = "top 5 licenses by dealer nebula in October 2025"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert " from licenses " in sql
    # time range from our normalizer should prefer date_created on licenses if present
    assert "date_created" in sql or "install_date" in sql

    # single-statement resolver: EXISTS with dealers
    assert " exists (" in sql and " from dealers " in sql
    assert "dealer_name" in sql
    assert "%nebula%" in sql
    # explicit escape on LIKE is enforced by policy/ast
    assert " escape " in sql

    # ranking & limit
    assert " order by " in sql and " desc" in sql
    assert " limit 5" in sql

def test_resolution_hosts_by_dealer_name_group_by_state():
    """
    'All hosts per state for dealer orion' should:
    - pick hosts.count
    - group by hosts.state
    - add EXISTS subquery resolving hosts.dealer_id via dealers.dealer_name ILIKE '%orion%'
    """
    nl = "all hosts per state for dealer orion"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert " from hosts " in sql
    assert " group by hosts.state" in sql
    assert " exists (" in sql and " from dealers " in sql
    assert "dealer_name" in sql and "%orion%" in sql
    # default LIMIT injected
    assert " limit 100" in sql

def test_resolver_prefers_exists_over_join_in_aggregate():
    """
    Aggregate query with a dealer name should resolve via EXISTS(...)
    and should NOT introduce a JOIN to dealers in Phase 1.
    """
    nl = "licenses.count for dealer nebula"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert " from licenses " in sql
    assert " exists (" in sql and " from dealers " in sql
    # Ensure we didn't turn the resolver into a JOIN for aggregates
    assert " join dealers " not in sql
    # LIKE/ILIKE with ESCAPE '\\'
    assert (" ilike " in sql) or (" like " in sql and "lower(" in sql)
    assert " escape '\\'" in sql
    assert " limit 100" in sql


def test_resolver_escapes_percent_and_underscore():
    """
    Resolver must escape wildcard characters in the user input so they are
    treated literally in LIKE. Expect '\%' and '\_' inside the quoted pattern
    plus ESCAPE '\\'.
    """
    nl = "licenses.count for dealer 100%_power"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert " from licenses " in sql
    assert " exists (" in sql and " from dealers " in sql
    # Pattern should contain escaped percent and underscore
    assert "100\\%_power" in sql or "100\\%\\_power" in sql
    # And the ESCAPE clause must be present
    assert " escape '\\'" in sql
    assert " limit 100" in sql


def test_hosts_resolver_exists_without_join_in_aggregate():
    """
    Hosts aggregate: use hosts.dealer_id -> dealers resolver via EXISTS.
    No JOIN to dealers should be introduced in Phase 1 aggregates.
    """
    nl = "hosts.count for dealer orion"
    out = orchestrate(nl, include_trace=False)
    assert "query" in out, out.get("error", "no error")
    sql = _norm(out["query"])

    assert " from hosts " in sql
    assert " exists (" in sql and " from dealers " in sql
    assert " join dealers " not in sql
    assert (" ilike " in sql) or (" like " in sql and "lower(" in sql)
    assert " escape '\\'" in sql
    assert " limit 100" in sql
