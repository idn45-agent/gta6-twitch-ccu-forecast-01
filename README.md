[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/idn45-agent/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/idn45-agent/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), as five quantiles per hour
(5/25/50/75/95%), from the first hour's print alone.

## Approach

Launch-day Twitch curves share a shape: the 00:00–01:00 UTC first hour
only covers New Zealand / Asia (the rolling midnight unlock), then
viewership climbs as Europe wakes, jumps at the US East (05:00 UTC) and
US West (08:00 UTC) unlocks, peaks in prime time, and decays late at
night. Instead of guessing that shape, the program reads it out of the
case itself:

1. For every comparable past launch in the case file, compute the
   log-ratio `ln(viewers(h) / viewers(1))` for hours 2–24 of its release
   day.
2. Per hour, the cross-launch distribution of those log-ratios *is* the
   forecast distribution: its empirical quantiles at the five levels,
   exponentiated and multiplied by the case's hour-1 print, are the five
   quantiles. Launches disagree more later in the day, so the band
   widens with the hour on its own — no hand-set widening.
3. A small spread floor keeps the 5%/95% band honest when launches
   happen to agree; a persistence-style fallback covers cases with too
   few usable comparables or a degenerate hour-1 print.

Python 3.13, standard library only, deterministic, no network, well
under the 8-second limit per case.

## Local validation

`tools/score_local.py` builds pseudo-cases from the comparables' own
release days (each launch's hour-1 print is the known, its hours 2–24
the answer, the other launches the context) and scores the program
against the house persistence baseline on the same cases. See
`gta6-builder-log.md` in the makefun workspace for the per-iteration
scores.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
