"""Where was this person? Their look-alikes from every camera, in time order, impossible ones removed.

Candidates are people whose appearance (embedding, mostly clothing) is close enough (min_similarity) and who
were seen within window_h hours. One person cannot be in two places at once, so candidates are accepted from
the most to the least similar, and one whose time in view overlaps an already accepted sighting is left out
(on another camera it would be a second place; on the same camera, a second person).
This is a lead, not proof: people in the same clothes look alike to it.
"""
from __future__ import annotations

from datetime import datetime, timedelta


def _span(t: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(t["started_at"])
    end = datetime.fromisoformat(t["ended_at"]) if t.get("ended_at") else datetime.now()
    return start, max(start, end)


def plausible(anchor: dict, candidates: list[dict], window_h: float) -> list[dict]:
    """anchor + accepted candidates (each with "similarity"), sorted by time. Each gets "gap_s": seconds
    since the previous sighting ended (None for the first)."""
    a_start, _ = _span(anchor)
    window = timedelta(hours=window_h)
    accepted = [dict(anchor, similarity=1.0, anchor=True)]
    spans = [_span(anchor)]
    for c in sorted(candidates, key=lambda c: -c["similarity"]):
        start, end = _span(c)
        if abs(start - a_start) > window:
            continue
        if any(start < e and s < end for s, e in spans):      # seen somewhere else at the same time
            continue
        accepted.append(dict(c, anchor=False))
        spans.append((start, end))
    accepted.sort(key=lambda t: t["started_at"])
    prev_end = None
    for t in accepted:
        start, end = _span(t)
        t["gap_s"] = round((start - prev_end).total_seconds()) if prev_end else None
        prev_end = end
    return accepted
