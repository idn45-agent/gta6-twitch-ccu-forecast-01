"""Local scorer for the GTA VI Twitch CCU forecast entry.

Builds pseudo-cases from the comparables' own release days (each
comparable's hour-1 print is the known, its hours 2..24 the answer, the
other comparables the context), runs a program on them, and reports the
mean pinball loss on the log scale -- the competition's score -- plus
typical miss and 90% coverage.

Usage:
    python3 tools/score_local.py [--program PATH] [--kit KIT_DIR]
                                 [--out DIR]

Default program is this repo's main.py; pass the kit starter's main.py
to score the house baseline on the same pseudo-cases.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
KIT_DEFAULT = os.path.expanduser(
    "~/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/"
    "gta6-twitch-ccu-forecast-01-kit"
)


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_hours(comparable):
    """Hour -> viewers for hours 1..24 of a comparable's release day."""
    day_start = _parse_time(comparable["day_start"])
    by_hour = {}
    for row in comparable.get("hourly") or []:
        try:
            t = _parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        delta = (t - day_start).total_seconds() / 3600.0
        hour = int(delta) + 1
        if 1 <= hour <= 24 and 0 <= delta < 24:
            by_hour.setdefault(hour, viewers)
    return by_hour


def build_pseudo_cases(kit_dir):
    """Return (pseudo_cases, answers): case dicts and {id: {hour: viewers}}."""
    with open(os.path.join(kit_dir, "input", "p00.in"), encoding="utf-8") as fh:
        practice = json.load(fh)
    comparables = practice["comparables"]
    usable = []
    for comp in comparables:
        hours = release_day_hours(comp)
        if hours.get(1) and hours[1] > 0 and len(hours) >= 23:
            usable.append((comp, hours))
    cases = []
    answers = {}
    for idx, (target, hours) in enumerate(usable):
        case_id = "pseudo-%02d" % idx
        others = [c for c, _ in usable if c is not target]
        case = {
            "id": case_id,
            "game": {"name": target["name"], "twitch_id": "0"},
            "day_start": target["day_start"],
            "checkpoint": target["day_start"],
            "levels": list(LEVELS),
            "known": [{
                "hour": 1,
                "from": target["day_start"],
                "until": target["day_start"],
                "viewers": hours[1],
                "channels": None,
                "reads": 1,
            }],
            "forecast": [{"hour": h} for h in range(2, 25)],
            "history": [],
            "history_hourly": [],
            "streams": None,
            "categories_hourly": None,
            "comparables": others,
            "notes": "pseudo-case built from %s" % target["name"],
        }
        cases.append((case_id, case))
        answers[case_id] = {h: hours[h] for h in range(2, 25)}
    return cases, answers


def pinball(y, q, tau):
    d = math.log(max(y, 1.0)) - math.log(max(q, 1.0))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_outputs(out_dir, answers):
    losses = []
    abs_log_err = []
    covered = 0
    total_hours = 0
    for case_id, truth in sorted(answers.items()):
        path = os.path.join(out_dir, case_id + ".out")
        with open(path, encoding="utf-8") as fh:
            forecast = json.load(fh)
        rows = {row["hour"]: row["q"] for row in forecast["hours"]}
        for hour, y in sorted(truth.items()):
            qs = rows[hour]
            for tau, q in zip(LEVELS, qs):
                losses.append(pinball(y, q, tau))
            abs_log_err.append(abs(math.log(max(y, 1.0)) - math.log(max(qs[2], 1.0))))
            total_hours += 1
            if qs[0] <= y <= qs[4]:
                covered += 1
    mean_loss = sum(losses) / len(losses)
    typical_miss = 100.0 * (math.exp(sum(abs_log_err) / len(abs_log_err)) - 1.0)
    coverage = covered / total_hours
    return mean_loss, typical_miss, coverage, len(answers)


def run_program(program, cases, workdir):
    in_dir = os.path.join(workdir, "in")
    out_dir = os.path.join(workdir, "out")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    for case_id, case in cases:
        with open(os.path.join(in_dir, case_id + ".in"), "w", encoding="utf-8") as fh:
            json.dump(case, fh)
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = in_dir
    env["MAKEFUN_OUTPUT_DIR"] = out_dir
    import time
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-P", "-s", program],
        env=env, capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        print("PROGRAM FAILED (rc=%d)" % proc.returncode)
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        sys.exit(1)
    return out_dir, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", default=None)
    parser.add_argument("--kit", default=KIT_DEFAULT)
    args = parser.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    program = args.program or os.path.join(repo_root, "main.py")

    cases, answers = build_pseudo_cases(args.kit)
    print("pseudo-cases: %d (one per usable comparable launch)" % len(cases))
    workdir = tempfile.mkdtemp(prefix="gta6-score-")
    try:
        out_dir, elapsed = run_program(program, cases, workdir)
        loss, miss, cov, n = score_outputs(out_dir, answers)
        print("program: %s" % program)
        print("mean pinball loss : %.6f" % loss)
        print("typical miss      : %.2f%%" % miss)
        print("90%% coverage      : %.3f" % cov)
        print("cases scored      : %d, wall time %.2fs (%.3fs/case)"
              % (n, elapsed, elapsed / n))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
