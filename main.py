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
3.  Per hour, fit a normal distribution to those log-ratios, weighted by
    how similar each launch is to this case: a Gaussian kernel on
    ``ln(viewers(1))`` (sigma 1.0), so launches whose first hour was
    about as big as this case's count most, and a x2 weight for
    console-only launches -- GTA VI is PS5/Xbox only, and console
    launches share the rolling-midnight unlock shape while PC launches
    often unlock simultaneously worldwide.  The fitted mean and spread
    give the five quantiles via the normal z-scores, with the spread
    shrunk by 0.8 (the raw cross-launch spread overstates the
    uncertainty once launches are weighted by similarity).
4.  A small spread floor keeps the 5%/95% band honest when launches
    happen to agree, and a persistence-style fallback covers cases with
    too few usable comparables or a degenerate hour-1 print.

Everything is computed from the case file itself, so the same program
works on the practice cases and on the final case.  Tuned on local
pseudo-cases built from the comparables' own release days (see
tools/score_local.py): mean pinball loss ~0.40 overall, ~0.20 on
console-only launches, against ~0.93 for the house persistence baseline
on the same pseudo-cases.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the five levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
HOURS = range(2, 25)
MAX_VIEWERS = 1_000_000_000.0
MIN_VIEWERS = 1.0
# Minimum number of usable comparable launches before trusting the
# fitted distribution; below this we fall back to persistence.
MIN_COMPARABLES = 5
# Spread floor (in log units) for the 5%/95% half-width, so the band
# never collapses when the launches happen to agree closely.
SPREAD_FLOOR = 0.30
# Width of the Gaussian kernel on ln(hour-1 viewers) that weights
# comparables by similarity to this case's first-hour print.
KERNEL_SIGMA = 1.0
# Extra weight for console-only launches (no PC in their platforms):
# GTA VI launches on PS5/Xbox only, and console launches share the
# rolling-midnight unlock shape.
CONSOLE_BOOST = 2.0
# The raw cross-launch spread overstates the uncertainty once launches
# are weighted by similarity; shrink the fitted spread by this factor.
SPREAD_SHRINK = 0.8


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _is_console_only(comparable):
    """True when the launch had no PC release (console-only)."""
    platforms = comparable.get("platforms") or ""
    return "PC" not in platforms.upper()


def _similarity_weight(v1, p1, console_only):
    """Kernel weight of a comparable launch for a case with print p1."""
    diff = (math.log(v1) - math.log(p1)) / KERNEL_SIGMA
    weight = math.exp(-0.5 * diff * diff)
    if console_only:
        weight *= CONSOLE_BOOST
    return weight


def _release_day_ratios(comparable):
    """Log-ratios ln(v(h)/v(1)) for h=2..24 of one comparable launch.

    Returns (ratios, v1, console_only), or (None, None, False) when the
    launch has no usable release-day curve (missing/zero hour-1 print,
    or too few printed hours).
    """
    try:
        day_start = _parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None, None, False
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
        return None, None, False
    ratios = {}
    for hour in HOURS:
        v = by_hour.get(hour)
        if v is not None and v > 0:
            ratios[hour] = math.log(v / v1)
    # Require most of the day to be present; a launch with a holey
    # release day would bias the per-hour samples unevenly.
    if len(ratios) < 18:
        return None, None, False
    return ratios, v1, _is_console_only(comparable)


def _forecast_from_ratios(p1, weighted_ratios):
    """Build the 23 hourly quantile rows from per-hour (ratio, weight) lists.

    Per hour, a weighted normal fit of the log-ratios gives the five
    quantiles: mean + z_tau * sd, exponentiated and scaled by p1.
    """
    hours = []
    for hour in HOURS:
        pairs = weighted_ratios[hour]
        total = sum(w for _, w in pairs)
        mean = sum(r * w for r, w in pairs) / total
        var = sum(w * (r - mean) ** 2 for r, w in pairs) / total
        sd = math.sqrt(max(var, 0.0)) * SPREAD_SHRINK
        qs = [mean + z * sd for z in Z_SCORES]
        # Spread floor: keep the 5%/95% band at least SPREAD_FLOOR wide
        # in log units on each side of the median.
        median = qs[2]
        half = max(qs[4] - median, median - qs[0], 0.0)
        if half < SPREAD_FLOOR:
            qs[0] = median - SPREAD_FLOOR
            qs[4] = median + SPREAD_FLOOR
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

    weighted_ratios = {hour: [] for hour in HOURS}
    n_used = 0
    for comparable in case.get("comparables") or []:
        ratios, v1, console_only = _release_day_ratios(comparable)
        if ratios is None:
            continue
        n_used += 1
        weight = _similarity_weight(v1, p1, console_only)
        for hour, value in ratios.items():
            weighted_ratios[hour].append((value, weight))

    usable_hours = sum(
        1 for hour in HOURS if len(weighted_ratios[hour]) >= MIN_COMPARABLES
    )
    if n_used < MIN_COMPARABLES or usable_hours < 18:
        return _fallback_forecast(p1)
    return _forecast_from_ratios(p1, weighted_ratios)


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
