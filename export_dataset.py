"""Export human-verified labels as a training dataset.

    python export_dataset.py                 # human-verified labels only
    python export_dataset.py --include-auto  # also confident automatic labels

Writes data/export/<timestamp>/images/*.jpg and labels.jsonl, one line per person:
    {"image": "images/12.jpg", "track_id": 12, "labels": {"upper_color": "red", ...}, "verified": [...]}
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime

from sqlalchemy import select

from labeler.config import load_config, resolve
from labeler.db import Database, Track, resolve as resolve_attrs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--include-auto", action="store_true",
                    help="include unreviewed labels at or above the review threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db = Database(cfg["database"]["url"])
    threshold = cfg["review"]["confidence_threshold"]
    crop_dir = resolve(cfg["storage"]["crop_dir"])
    out = resolve("data/export") / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "images").mkdir(parents=True)

    n = 0
    with db.Session() as s, open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        tracks = s.scalars(select(Track).where(Track.crop_path.is_not(None))).all()
        for t in tracks:
            resolved = resolve_attrs(t.attributes)
            labels, verified = {}, []
            for name, a in resolved.items():
                if a["value"] in ("unknown",):
                    continue
                if a["source"] == "human":
                    labels[name] = a["value"]
                    verified.append(name)
                elif args.include_auto and not a["needs_review"] and a["confidence"] >= threshold:
                    labels[name] = a["value"]
            if not labels or (not verified and not args.include_auto):
                continue
            src = crop_dir / t.crop_path
            if not src.exists():
                continue
            shutil.copy2(src, out / "images" / t.crop_path)
            f.write(json.dumps({"image": f"images/{t.crop_path}", "track_id": t.id,
                                "labels": labels, "verified": verified}) + "\n")
            n += 1
    print(f"exported {n} people to {out}")


if __name__ == "__main__":
    main()
