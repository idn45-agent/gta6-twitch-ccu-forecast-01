"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads every ``*.in`` case file in $MAKEFUN_INPUT_DIR (default /input) and
writes one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR
(default /output): five quantiles (5/25/50/75/95%) of the Twitch category
viewers for hours 2..24 of the release day.

Method (stdlib only, deterministic, no network):

1.  Anchor on the case's hour-1 print ``p1``.
2.  For every comparable past launch in the case, take its release-day
    hourly curve and compute the log-ratio ``ln(viewers(h) / viewers(1))``
    for hours 2..24.  Launch-day curves share a shape: a first-hour print
    taken at 00:00-01:00 UTC (only New Zealand / Asia unlocked), then a
    climb as Europe wakes, a jump at the US East (05:00 UTC) and US West
    (08:00 UTC) unlocks, a prime-time peak, and a late-night decay.
3.  Per hour, the cross-launch distribution of those log-ratios is the
    forecast distribution: its empirical quantiles at the five levels,
    exponentiated and multiplied by ``p1``, are the five quantiles.
    Launches disagree more later in the day, so the spread widens with h
    on its own -- no hand-set widening is needed.
4.  A small spread floor keeps the 5%/95% band honest when launches
    happen to agree, and a persistence-style fallback covers cases with
    too few usable comparables or a degenerate hour-1 print.

Everything is computed from the case file itself, so the same program
works on the practice cases and on the final case.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOURS = range(2, 25)
MAX_VIEWERS = 1_000_000_000.0
MIN_VIEWERS = 1.0
# Minimum number of usable comparable launches before trusting the
# empirical distribution; below this we fall back to persistence.
MIN_COMPARABLES = 5
# Spread floor (in log units) for the 5%/95% half-width, so the band
# never collapses when the launches happen to agree closely.
SPREAD_FLOOR = 0.35


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _quantile(sorted_values, level):
    """Linear-interpolation quantile of a sorted list of floats."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("quantile of empty list")
    if n == 1:
        return sorted_values[0]
    pos = level * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _release_day_ratios(comparable):
    """Log-ratios ln(v(h)/v(1)) for h=2..24 of one comparable launch.

    Returns a dict hour -> log-ratio, or None when the launch has no
    usable release-day curve (missing/zero hour-1 print, or too few
    printed hours).
    """
    try:
        day_start = _parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comparable.get("hourly") or []:
        try:
            t = _parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        delta_hours = (t - day_start).total_seconds() / 3600.0
        hour = int(delta_hours) + 1
        if 1 <= hour <= 24 and 0 <= delta_hours < 24:
            # Keep the first row seen for an hour; rows are in order.
            by_hour.setdefault(hour, viewers)
    v1 = by_hour.get(1)
    if v1 is None or v1 <= 0:
        return None
    ratios = {}
    for hour in HOURS:
        v = by_hour.get(hour)
        if v is not None and v > 0:
            ratios[hour] = math.log(v / v1)
    # Require most of the day to be present; a launch with a holey
    # release day would bias the per-hour samples unevenly.
    if len(ratios) < 18:
        return None
    return ratios


def _forecast_from_ratios(p1, ratio_lists):
    """Build the 23 hourly quantile rows from per-hour log-ratio lists."""
    hours = []
    for hour in HOURS:
        values = sorted(ratio_lists[hour])
        qs = [_quantile(values, tau) for tau in LEVELS]
        # Spread floor: keep the 5%/95% band at least SPREAD_FLOOR wide
        # in log units on each side of the median.
        median = qs[2]
        half = max(qs[4] - median, median - qs[0], 0.0)
        if half < SPREAD_FLOOR:
            qs[0] = median - SPREAD_FLOOR
            qs[4] = median + SPREAD_FLOOR
        # Re-derive the inner quantiles from the widened band only if
        # they would cross the median after widening (they cannot, the
        # empirical quantiles are ordered, but stay safe).
        row = []
        prev = 0.0
        for q in qs:
            val = p1 * math.exp(q)
            if val < MIN_VIEWERS:
                val = MIN_VIEWERS
            elif val > MAX_VIEWERS:
                val = MAX_VIEWERS
            if val < prev:
                val = prev
            row.append(round(val, 2))
            prev = val
        hours.append({"hour": hour, "q": row})
    return {"hours": hours}


def _fallback_forecast(p1):
    """Persistence forecast with a widening spread (house-baseline shape).

    Used only when the case carries too few usable comparables.
    """
    z = (-1.645, -0.674, 0.0, 0.674, 1.645)
    hours = []
    for hour in HOURS:
        width = 0.15 * math.sqrt(hour - 1)
        row = []
        prev = 0.0
        for zi in z:
            val = p1 * math.exp(zi * width)
            if val < MIN_VIEWERS:
                val = MIN_VIEWERS
            elif val > MAX_VIEWERS:
                val = MAX_VIEWERS
            if val < prev:
                val = prev
            row.append(round(val, 2))
            prev = val
        hours.append({"hour": hour, "q": row})
    return {"hours": hours}


def forecast_case(case):
    """Forecast one case dict, returning the output object."""
    try:
        p1 = float(case["known"][0]["viewers"])
    except (KeyError, IndexError, TypeError, ValueError):
        p1 = 1.0
    if not (p1 > 0) or math.isnan(p1) or math.isinf(p1):
        p1 = 1.0

    ratio_lists = {hour: [] for hour in HOURS}
    n_used = 0
    for comparable in case.get("comparables") or []:
        ratios = _release_day_ratios(comparable)
        if ratios is None:
            continue
        n_used += 1
        for hour, value in ratios.items():
            ratio_lists[hour].append(value)

    usable_hours = sum(1 for hour in HOURS if len(ratio_lists[hour]) >= MIN_COMPARABLES)
    if n_used < MIN_COMPARABLES or usable_hours < 18:
        return _fallback_forecast(p1)
    return _forecast_from_ratios(p1, ratio_lists)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception:
            # Never crash a case: fall back to a safe persistence
            # forecast built from the hour-1 print alone.
            try:
                p1 = float(case["known"][0]["viewers"])
            except (KeyError, IndexError, TypeError, ValueError):
                p1 = 1.0
            if not (p1 > 0) or math.isnan(p1) or math.isinf(p1):
                p1 = 1.0
            result = _fallback_forecast(p1)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
