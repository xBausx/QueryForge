# src/orchestrator/tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class TokType(str, Enum):
    IDENT = "IDENT"      # letters, digits, underscores, dots (e.g., licenses.dealer_id)
    STRING = "STRING"    # '...'/ "..." (supports simple escaping of quotes)
    NUMBER = "NUMBER"    # 123 or 45.67
    OP = "OP"            # = != <> > >= < <=
    COMMA = "COMMA"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    HYPHEN = "HYPHEN"    # -
    KEYWORD = "KEYWORD"  # top, bottom, limit, by, per, for, between, in, like, contains, where, on, and, or
    EOF = "EOF"


_KEYWORDS = {
    "top", "bottom", "limit",
    "by", "per", "for",
    "between", "and", "or",
    "in", "like", "contains",
    "where", "on",
}


@dataclass
class Token:
    type: TokType
    value: str
    pos: int


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch in "_"


def _is_ident_part(ch: str) -> bool:
    return ch.isalnum() or ch in "._"


def _is_op_start(ch: str) -> bool:
    return ch in "=!<>"


def _read_op(s: str, i: int) -> tuple[str, int]:
    # two-char ops first
    if i + 1 < len(s):
        two = s[i:i + 2]
        if two in ("!=", "<=", ">="):
            return two, i + 2
    return s[i], i + 1


def tokenize(text: str) -> list[Token]:
    s = text or ""
    i = 0
    out: list[Token] = []
    n = len(s)

    while i < n:
        ch = s[i]

        # whitespace
        if ch.isspace():
            i += 1
            continue

        # strings
        if ch in ("'", '"'):
            quote = ch
            i += 1
            start = i
            buf: list[str] = []
            while i < n:
                c = s[i]
                if c == "\\" and i + 1 < n:
                    buf.append(c)
                    buf.append(s[i + 1])
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                buf.append(c)
                i += 1
            out.append(Token(TokType.STRING, "".join(buf), start - 1))
            continue

        # number (simple)
        if ch.isdigit():
            start = i
            i += 1
            dot_used = False
            while i < n and (s[i].isdigit() or (s[i] == "." and not dot_used)):
                if s[i] == ".":
                    dot_used = True
                i += 1
            out.append(Token(TokType.NUMBER, s[start:i], start))
            continue

        # identifier / keyword
        if _is_ident_start(ch):
            start = i
            i += 1
            while i < n and _is_ident_part(s[i]):
                i += 1
            val = s[start:i]
            low = val.lower()
            # Special-case: "top10" / "bottom25" → split into keyword + number
            if low.startswith("top") and len(low) > 3 and low[3:].isdigit():
                out.append(Token(TokType.KEYWORD, "top", start))
                out.append(Token(TokType.NUMBER, low[3:], start + 3))
                continue
            if low.startswith("bottom") and len(low) > 6 and low[6:].isdigit():
                out.append(Token(TokType.KEYWORD, "bottom", start))
                out.append(Token(TokType.NUMBER, low[6:], start + 6))
                continue
            ttype = TokType.KEYWORD if low in _KEYWORDS else TokType.IDENT
            out.append(Token(ttype, low if ttype == TokType.KEYWORD else val, start))
            continue

        # operators
        if _is_op_start(ch):
            val, i2 = _read_op(s, i)
            out.append(Token(TokType.OP, val, i))
            i = i2
            continue

        # punctuation
        if ch == ",":
            out.append(Token(TokType.COMMA, ch, i)); i += 1; continue
        if ch == "(":
            out.append(Token(TokType.LPAREN, ch, i)); i += 1; continue
        if ch == ")":
            out.append(Token(TokType.RPAREN, ch, i)); i += 1; continue
        if ch == "-":
            out.append(Token(TokType.HYPHEN, ch, i)); i += 1; continue

        # fallback: treat as identifier char to be permissive
        start = i
        i += 1
        out.append(Token(TokType.IDENT, s[start:i], start))

    out.append(Token(TokType.EOF, "", n))
    return out


def collect_until(tokens: list[Token], start_idx: int, stop_keywords: set[str]) -> tuple[list[Token], int]:
    """
    Collect tokens from start_idx (inclusive) until we see a KEYWORD whose value is in stop_keywords,
    or EOF. Returns (collected, next_index).
    """
    i = start_idx
    out: list[Token] = []
    while i < len(tokens):
        t = tokens[i]
        if t.type == TokType.KEYWORD and t.value in stop_keywords:
            break
        if t.type == TokType.EOF:
            break
        out.append(t)
        i += 1
    return out, i


def tokens_to_text(ts: list[Token]) -> str:
    """
    Reconstruct a space-joined string of identifiers / numbers / strings (strings without quotes).
    """
    buf: list[str] = []
    for t in ts:
        if t.type in (TokType.IDENT, TokType.NUMBER):
            buf.append(t.value)
        elif t.type == TokType.STRING:
            buf.append(t.value)
    return " ".join([b for b in buf if b])
