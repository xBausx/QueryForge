# src/orchestrator/date_normalizer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


# ---------------------------
# Public API
# ---------------------------

@dataclass(frozen=True)
class NormalizedRange:
    """
    Closed-open date range [start, end), expressed as ISO dates.
    The orchestrator's LogicalQueryRequest.TimeRange prefers an exclusive upper bound.
    """
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"


def infer_time_range(
    text: str,
    *,
    tz: str = "Asia/Manila",
    now: Optional[datetime] = None,
) -> Optional[NormalizedRange]:
    """
    Deterministically infer a monthly or daily time window from short, partial date phrases.

    Supported phrases (case-insensitive, English month names/abbrevs):
        - "October 2025"  -> [2025-10-01, 2025-11-01)
        - "Oct 2025"      -> [2025-10-01, 2025-11-01)
        - "2025-10"       -> [2025-10-01, 2025-11-01)   (ISO year-month)
        - "October 1"     -> uses CURRENT YEAR in the provided timezone
        - "Oct 1, 2025"   -> [2025-10-01, 2025-10-02)
        - "2025-10-01"    -> [2025-10-01, 2025-10-02)   (ISO date)

    Rules you specified:
        - Month + Year -> first day of that month to first day of next month (exclusive upper bound).
        - Month + Day (no year) -> assume current year in the given timezone.

    Returns:
        NormalizedRange(start="YYYY-MM-DD", end="YYYY-MM-DD") if a supported phrase is found,
        otherwise None.

    Notes:
        - This function intentionally avoids fuzzy or locale-wide NLP; it’s a deterministic
        ruleset to keep the router predictable and portable.
        - The caller can plug the result into LogicalQueryRequest.time_range directly.
    """
    if not text or not isinstance(text, str):
        return None

    text_norm = text.strip()

    # Establish "now" in the target tz for current-year assumptions
    if now is None:
        now = datetime.now(ZoneInfo(tz))
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo(tz))
    else:
        # Normalize to target tz (keeping instant)
        now = now.astimezone(ZoneInfo(tz))

    # 1) Full ISO date: YYYY-MM-DD
    m = _RE_ISO_DATE.search(text_norm)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            day = date(y, mo, d)
        except ValueError:
            return None
        return NormalizedRange(start=day.isoformat(), end=(day + timedelta(days=1)).isoformat())

    # 2) ISO year-month: YYYY-MM
    m = _RE_ISO_YM.search(text_norm)
    if m:
        y, mo = map(int, m.groups())
        try:
            first = date(y, mo, 1)
        except ValueError:
            return None
        return NormalizedRange(start=first.isoformat(), end=_first_of_next_month(first).isoformat())

    # 3) "MonthName Year" (e.g., "October 2025", "Oct 2025")
    m = _RE_MONTHNAME_YEAR.search(text_norm)
    if m:
        month_name, y = m.group("month"), int(m.group("year"))
        mo = _month_name_to_num(month_name)
        if mo is None:
            return None
        first = date(y, mo, 1)
        return NormalizedRange(start=first.isoformat(), end=_first_of_next_month(first).isoformat())

    # 4) "MonthName Day[, Year]?" (e.g., "October 1", "Oct 1, 2025")
    m = _RE_MONTHNAME_DAY_OPT_YEAR.search(text_norm)
    if m:
        month_name = m.group("month")
        day = int(m.group("day"))
        year = m.group("year")
        y = int(year) if year else now.year
        mo = _month_name_to_num(month_name)
        if mo is None:
            return None
        try:
            d = date(y, mo, day)
        except ValueError:
            return None
        return NormalizedRange(start=d.isoformat(), end=(d + timedelta(days=1)).isoformat())

    # No supported phrase found
    return None


# ---------------------------
# Internal helpers
# ---------------------------

# Month name patterns (English), 3-letter abbrevs allowed (case-insensitive)
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def _month_name_to_num(s: str) -> Optional[int]:
    return _MONTHS.get(s.strip().lower())


def _first_of_next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


# Regexes
# Using word boundaries and permissive whitespace. Case-insensitive search.
_RE_ISO_DATE = re.compile(r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b")
_RE_ISO_YM   = re.compile(r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])\b")

# e.g., "October 2025", "Oct 2025"
_RE_MONTHNAME_YEAR = re.compile(
    r"\b(?P<month>[A-Za-z]{3,9})\s+(?P<year>(?:19|20)\d{2})\b",
    flags=re.IGNORECASE
)

# e.g., "October 1", "Oct 1, 2025", "Oct 01 2025"
_RE_MONTHNAME_DAY_OPT_YEAR = re.compile(
    r"\b(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:[,\s]+(?P<year>(?:19|20)\d{2}))?\b",
    flags=re.IGNORECASE
)
