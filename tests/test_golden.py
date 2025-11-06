# tests/test_golden.py
import re
import pytest

from orchestrator.router_runtime import DeterministicRouter
from orchestrator.policy_gate import apply_policy, PolicyError
from orchestrator.sql_ast_builder import compile_lqr_to_sql

def _compile(nl: str) -> str:
    r = DeterministicRouter()
    lqr = r.route(nl)
    lqr = apply_policy(lqr)
    return compile_lqr_to_sql(lqr)

def _norm(s: str) -> str:
    # collapse whitespace to make assertions stable
    return re.sub(r"\s+", " ", s.strip())

def test_contains_like_detail():
    sql = _compile("list advertisers where name contains shop limit 20")
    s = _norm(sql.lower())
    assert s.startswith("select ")
    assert " from advertisers " in s
    assert "where advertisers.name like '%shop%'" in s
    assert " group by " not in s
    assert " limit 20" in s

def test_starts_with_detail_ordered():
    sql = _compile("list advertisers where name starts with Acme order by name asc limit 10")
    s = _norm(sql.lower())
    assert "where advertisers.name like 'acme%'" in s
    assert " group by " not in s  # detail mode, no aggregate
    assert " order by " in s and " name asc " in s
    assert " limit 10" in s

def test_state_in_list_and_order():
    sql = _compile("list advertisers in TX, CA order by state asc limit 50")
    s = _norm(sql)
    # keep case for state abbreviations
    assert "WHERE advertisers.state IN ('TX', 'CA')" in sql or "where advertisers.state in ('TX','CA')" in s
    assert "ORDER BY" in sql and (" state ASC" in sql or " state asc" in s)
    assert s.endswith(" limit 50")

def test_top_order_by_city_is_aggregate():
    sql = _compile("top 10 advertisers order by city")
    s = _norm(sql.lower())
    assert "count(*) as count_rows" in s
    assert " group by advertisers.city" in s
    assert " order by count_rows desc" in s
    assert " limit 10" in s

def test_latest_detail_default_columns():
    sql = _compile("latest 10 advertisers")
    s = _norm(sql.lower())
    assert " from advertisers " in s
    assert " order by " in s and (" date_created desc" in s or " date_updated desc" in s)
    assert " limit 10" in s
