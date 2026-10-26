"""Day against night (Phase 53 step 2): the safety checks measured on the same videos by day, as infrared and as dark
colour (tools/make_night_videos.py), with tools/incident_check.py and the same settings as the README's day numbers.

    python tools/night_check.py --config <test_config.yaml> [--sets falls,fights,fire,floor,behaviour,everyday]
                                [--conditions day,ir,dark] [--summary]

Each set runs day, then ir, then dark (so a stopped run still has whole comparisons); each result is kept in
data/datasets/night_made/results/<set>_<condition>.json and a finished one is not run again. Every kind is reported on
every set, so a false alarm that only comes at night (fire from a bright IR lamp, a dark picture taken for a covered
camera) shows wherever it happens. --summary only prints the table from the kept results. Needs Ollama and the
videos (data/datasets, not in git); use a test config from tools/make_test_config.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "datasets" / "night_made" / "results"
CONDITIONS = ("day", "ir", "dark")
KINDS = "person_down,fall,fire,fight,weapon,child,floor,exit,equipment,crowd,suspicious"
# set -> (truth file under the base, areas file or None, incident_check options): the README's day settings.
SETS = {
    "falls": ("urfd/incidents_truth.csv", None,   # the report half: falls 16-30, everyday clips 21-40
              ["--fps", "30", "--hold-s", "30", "--match", r"(fall-(1[6-9]|2[0-9]|30)|adl-(2[1-9]|3[0-9]|40))-cam0"]),
    "fights": ("survfight/incidents_truth.csv", None, ["--fps", "25", "--match", r"[13579]\.mp4$"]),   # odd clips
    "fire": ("dfire/videos_truth.csv", None, []),
    "floor": ("floor_made/truth.csv", "floor_made/areas.json", ["--match", "report_"]),
    "behaviour": ("behaviour_made/truth.csv", "behaviour_made/areas.json", []),
    "everyday": ("behaviour_made/everyday_truth.csv", "behaviour_made/areas.json", []),
}


def base(condition: str) -> str:
    return "data/datasets" if condition == "day" else f"data/datasets/night_made/{condition}"


def run(config: str, name: str, condition: str) -> None:
    out = RESULTS / f"{name}_{condition}.json"
    if out.exists():
        return
    truth, areas, extra = SETS[name]
    cmd = [sys.executable, str(ROOT / "tools" / "incident_check.py"), "--config", config,
           "--truth", f"{base(condition)}/{truth}", "--kinds", KINDS, "--json", str(out.with_suffix(".part")), *extra]
    if areas:
        cmd += ["--areas", f"{base(condition)}/{areas}"]
    print(f"== {name} {condition}", flush=True)
    log = (RESULTS / f"{name}_{condition}.log").open("w", encoding="utf-8")
    if subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
        sys.exit(f"{name} {condition} failed: see {log.name}")
    out.with_suffix(".part").replace(out)


def summary() -> None:
    """Per set and kind: found of the true spans, and false alerts (confirmed outside every span), per condition."""
    print("| set | kind | " + " | ".join(f"{c}: found / false (raised)" for c in CONDITIONS) + " |")
    print("|---|---|" + "---|" * len(CONDITIONS))
    for name in SETS:
        got = {c: json.loads((RESULTS / f"{name}_{c}.json").read_text(encoding="utf-8"))
               for c in CONDITIONS if (RESULTS / f"{name}_{c}.json").exists()}
        if not got:
            continue
        rows = {c: {r["kind"]: r for r in g["rows"]} for c, g in got.items()}
        for kind in KINDS.split(","):
            seen = [rows[c].get(kind) for c in got]
            if not any(r and (r["spans"] or r["false_raised"] or r["false_found"]) for r in seen):
                continue
            cells = []
            for c in CONDITIONS:
                r = rows.get(c, {}).get(kind)
                cells.append("-" if r is None else
                             (f"{r['found']} of {r['spans']} / " if r["spans"] else "")
                             + f"{r['false_found']} ({r['false_raised']})")
            print(f"| {name} | {kind} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.summary:
        if not args.config:
            ap.error("--config is needed (tools/make_test_config.py)")
        for name in args.sets.split(","):
            for condition in args.conditions.split(","):
                run(args.config, name, condition)
    summary()


if __name__ == "__main__":
    main()
