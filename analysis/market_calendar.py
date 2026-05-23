"""
US market closure calendar.

The pipeline crons fire regardless of whether markets are open. On holidays
when NYSE/NASDAQ are closed, picks would generate against stale Friday data,
producing Discord messages with unreliable strikes/premiums. This module
provides a simple lookup so market-dependent stages can skip cleanly.

Maintained as a hardcoded dict because:
  - Market closures != federal holidays (Good Friday is a market closure but
    not federal; Columbus Day and Veterans Day are federal but markets are open)
  - Lightweight — no external calendar library dependency
  - Easy to audit against the official NYSE calendar at
    https://www.nyse.com/markets/hours-calendars
  - Easy to extend annually

UPDATE EVERY DECEMBER for the upcoming year.
"""

from datetime import datetime, date, timedelta


US_MARKET_CLOSURES = {
    # === 2026 ===
    "2026-01-01": "New Year's Day",
    "2026-01-19": "MLK Day",
    "2026-02-16": "Presidents Day",
    "2026-04-03": "Good Friday",
    "2026-05-25": "Memorial Day",
    "2026-06-19": "Juneteenth",
    "2026-07-03": "Independence Day (observed)",  # July 4, 2026 = Saturday
    "2026-09-07": "Labor Day",
    "2026-11-26": "Thanksgiving",
    "2026-12-25": "Christmas Day",
    # === 2027 ===
    "2027-01-01": "New Year's Day",
    "2027-01-18": "MLK Day",
    "2027-02-15": "Presidents Day",
    "2027-03-26": "Good Friday",
    "2027-05-31": "Memorial Day",
    "2027-06-18": "Juneteenth (observed)",  # June 19, 2027 = Saturday
    "2027-07-05": "Independence Day (observed)",  # July 4, 2027 = Sunday
    "2027-09-06": "Labor Day",
    "2027-11-25": "Thanksgiving",
    "2027-12-24": "Christmas Day (observed)",  # Dec 25, 2027 = Saturday
    # === 2028 ===
    "2028-01-17": "MLK Day",
    "2028-02-21": "Presidents Day",
    "2028-04-14": "Good Friday",
    "2028-05-29": "Memorial Day",
    "2028-06-19": "Juneteenth",
    "2028-07-04": "Independence Day",
    "2028-09-04": "Labor Day",
    "2028-11-23": "Thanksgiving",
    "2028-12-25": "Christmas Day",
}


# Stages that depend on live market data — skip these on closures.
# friday_close, reflect, and scorecard are deliberately omitted because they
# operate on historical data (grading prior week, generating reflections);
# they're safe to run on holidays.
MARKET_DEPENDENT_STAGES = frozenset({
    "monday", "monday_picks",
    "confirm", "monday_entry_confirmation",
    "monitor", "position_monitor",
    "final_exit",
    "wednesday", "wednesday_scan",
    "friday", "friday_refresh",
})


def is_market_closed(date_str: str = None) -> "str | None":
    """Return the holiday name if `date_str` (YYYY-MM-DD) is a US market closure.

    Parameters
    ----------
    date_str : str, optional
        Date to check in YYYY-MM-DD format. Defaults to today.

    Returns
    -------
    str or None
        Holiday name (e.g. "Memorial Day") if markets are closed, None otherwise.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return US_MARKET_CLOSURES.get(date_str)


def is_market_dependent(stage_name: str) -> bool:
    """Return True if the given stage requires live market data and should
    therefore be skipped on market-closed days.
    """
    return stage_name.lower() in MARKET_DEPENDENT_STAGES


# How many days ahead each stage looks for its "pick week."
# Wed scan + Fri refresh PREP for the FOLLOWING Monday's picks; Mon picks
# and intraday monitor stages operate WITHIN the current week.
_STAGE_PICK_WEEK_OFFSET = {
    "wednesday": 5,         # Wed → Mon = 5 days
    "wednesday_scan": 5,
    "friday": 3,            # Fri → Mon = 3 days
    "friday_refresh": 3,
}


def get_pick_week_start(today: date, stage: str) -> date:
    """Return the Monday that begins the pick week associated with `stage`
    running on `today`.

    Wed scan and Fri refresh build the candidate pool for the NEXT Monday's
    picks, so their pick week starts ahead. Monday picks, monitor stages, and
    final_exit all operate within the CURRENT week (today's Monday).
    """
    offset = _STAGE_PICK_WEEK_OFFSET.get(stage.lower())
    if offset is not None:
        return today + timedelta(days=offset)
    # weekday() returns 0=Mon, 1=Tue, ..., 6=Sun
    return today - timedelta(days=today.weekday())


def is_pick_week_holiday_affected(
    today: "date | None" = None,
    stage: str = "monday",
) -> "tuple[str, str, str] | None":
    """Check whether the pick week associated with `stage` on `today` contains
    any US market closure.

    A "pick week" is the Mon-Fri block of trading days for a given week's
    options cycle. If ANY day in that block is a closure, the week is
    "shortened" and the entire cycle (Wed scan + Fri refresh + Mon picks +
    monitors + final_exit) should be skipped. Saves the user from getting
    Discord noise on weeks that won't be traded anyway.

    Parameters
    ----------
    today : date, optional
        Defaults to today.
    stage : str
        The pipeline stage being considered. Disambiguates the pick week
        (wed/fri prep stages reach FORWARD; in-week stages stay current).

    Returns
    -------
    tuple of (holiday_name, holiday_date_iso, pick_week_start_iso) or None
        Tuple if the pick week is shortened by a closure; None if all 5
        days are tradeable.
    """
    if today is None:
        today = datetime.now().date()
    week_start = get_pick_week_start(today, stage)
    for i in range(5):  # Mon through Fri
        check_date = week_start + timedelta(days=i)
        date_str = check_date.strftime("%Y-%m-%d")
        if date_str in US_MARKET_CLOSURES:
            return (
                US_MARKET_CLOSURES[date_str],
                date_str,
                week_start.strftime("%Y-%m-%d"),
            )
    return None
