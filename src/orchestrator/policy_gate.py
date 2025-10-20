# src/orchestrator/policy_gate.py
from __future__ import annotations

import re
from typing import Iterable, Optional

import sqlglot
from sqlglot import exp

from .errors import OrchestratorError, ErrorCode


def enforce_policies(
    sql: str,
    *,
    forbidden: Optional[Iterable[str]] = None,
    default_limit: int = 100,
    dialect: Optional[str] = None,
) -> str:
    """
    Safety gate applied to the final SQL string.

    Policies enforced:
      1) Strip comments; first token must be SELECT (no CTE/DDL/DML/multi-statement).
      2) No forbidden fields/patterns (simple case-insensitive substring match).
      3) Ensure all LIKE/ILIKE have ESCAPE '\\' (inject if absent).
      4) LIMIT is required; inject default LIMIT if none present.
      5) Re-parse and serialize canonically.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, "Empty SQL provided to policy gate")

    # 1) Block comments outright, then strip to normalize
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise OrchestratorError(
            ErrorCode.UNSUPPORTED_OPERATION,
            "Comments are not allowed in generated SQL",
        )
    stripped = _strip_comments(sql).lstrip()

    # Must start with SELECT (not WITH/INSERT/UPDATE/etc.)
    if not stripped[:6].upper().startswith("SELECT"):
        raise OrchestratorError(
            ErrorCode.UNSUPPORTED_OPERATION,
            "Only SELECT statements are allowed (first token must be SELECT)",
        )

    # Parse; must be exactly one statement; normalize to a Select node
    try:
        stmts = sqlglot.parse(stripped)
    except Exception as e:
        raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, f"Failed to parse SQL: {e}")

    if not stmts:
        raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, "No parseable statement found")
    if len(stmts) != 1:
        raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, "Multiple statements are not allowed")

    node = stmts[0]
    if isinstance(node, (exp.With, exp.Union, exp.Except, exp.Intersect)):
        raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, "Only a single plain SELECT is allowed")
    if not isinstance(node, exp.Select):
        inner = node
        if isinstance(node, exp.Paren) and isinstance(node.this, exp.Select):
            inner = node.this
        if not isinstance(inner, exp.Select):
            raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, "Statement must be a SELECT")
        node = inner  # normalize

    # 2) (we'll compute canonical SQL after mutations) — Forbidden pattern scan happens later

    # 3) Ensure LIKE/ILIKE all have ESCAPE '\\'
    #    (do not alter the pattern; just attach ESCAPE if it's missing)
    for like_node in list(node.find_all(exp.Like)):
        if like_node.args.get("escape") is None:
            like_node.set("escape", exp.Literal.string("\\"))

    ILike = getattr(exp, "ILike", None)
    if ILike is not None:
        for ilike_node in list(node.find_all(ILike)):
            if ilike_node.args.get("escape") is None:
                ilike_node.set("escape", exp.Literal.string("\\"))

    # 4) Inject default LIMIT if absent
    if not _has_limit(node):
        node.set("limit", exp.Limit(expression=exp.Literal.number(int(default_limit))))

    # Canonicalize and check forbidden patterns on the final string
    canonical_sql = node.sql(dialect=dialect).strip()
    if forbidden:
        hit = _find_forbidden(canonical_sql, forbidden)
        if hit:
            raise OrchestratorError(
                ErrorCode.FORBIDDEN_FIELD,
                f"Use of forbidden field/pattern detected: '{hit}'",
            )

    return canonical_sql


# ---------------------------
# Internals
# ---------------------------

_LINE_COMMENT = re.compile(r"--[^\n]*")
_HASH_COMMENT = re.compile(r"#[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", flags=re.DOTALL)


def _strip_comments(sql: str) -> str:
    s = _BLOCK_COMMENT.sub("", sql)
    s = _LINE_COMMENT.sub("", s)
    s = _HASH_COMMENT.sub("", s)
    return s


def _find_forbidden(sql: str, forbidden: Iterable[str]) -> Optional[str]:
    low = sql.lower()
    for pat in forbidden:
        if not pat:
            continue
        if pat.lower() in low:
            return pat
    return None


def _has_limit(node: exp.Expression) -> bool:
    if isinstance(node, exp.Select):
        return node.args.get("limit") is not None
    sel = node.find(exp.Select)
    return bool(sel and sel.args.get("limit"))
