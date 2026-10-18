"""What's new (Phase 36): after an update, the Live page says what the new version added, in plain words.

ENTRIES is a short list, one entry per phase in the order they were finished (not by phase number: Phase 16 came
after 27 and 28). The app remembers the newest entry it ran (`whats_new.state_file`, data/last_version.json):
  - the same as now: nothing changes (a restart keeps showing what the last update added);
  - older: the entries after it are new in this version ("before" moves on to it);
  - no file yet: the first start with this feature. The user's app may jump from before Phase 13, so the database
    tells how old the last version was: tables and columns made at this start (Database.created_tables /
    added_columns) belong to phases that never ran here. A new database (people's table made now) is a new install:
    nothing is "new" then.
The list is ordered, so no git is needed (an install from the offline package has none); the git commit is only
shown. The banner can be closed per browser (session storage on the page, not here).
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import ROOT

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entry:
    key: str           # stable name, stored in the state file
    phase: int
    commit: str        # the commit that finished it (shown on the page; "" for the newest until the next commit)
    title: str
    text: str          # plain words: what it does and where to find it
    page: str          # where to look ("" = nowhere in particular)
    who: str = "all"   # "all" or "admin": viewers see only "all"
    marker: str = ""   # a table or "table.column" this phase added: made at start = this phase never ran here


ENTRIES: list[Entry] = [
    Entry("p13", 13, "ca573e4", "Recording",
          "The cameras record video while people are in view (kept 3 days, at most 100 GB). Watch it on the "
          "Recordings page; people's pages and alerts jump to the moment. Viewers see faces blurred.",
          "/recordings", marker="recordings"),
    Entry("p15", 15, "0817e40", "A camera sent from a phone or another PC",
          "Settings, \"Add camera sent from a phone or PC\": the phone (Larix Broadcaster) or the PC sends its camera "
          "to this PC over Tailscale, so it works through firewalls.", "/settings", "admin"),
    Entry("p17", 17, "3b7ffd5", "Mobility aids and fall alerts",
          "People are labeled with a wheelchair, walking stick, crutches or walker. A camera that sees a room from "
          "the side notices a fall: a red pop-up with a beep on every page until someone presses OK. Mark beds and "
          "sofas \"Lying is normal here\" with Zones on the Live page.", "/alerts", marker="zones.lying_ok"),
    Entry("p18", 18, "9a03ef4", "Vehicles",
          "Security cameras find cars, motorbikes, bicycles, buses and lorries, with type, color, make and model. "
          "They are listed on the Live page and can have alert rules.", "/", marker="vehicles"),
    Entry("p19", 19, "0bc7d41", "Home / Away",
          "A Home / Away button at the top of every page, and a weekly Away schedule on the Settings page. Alert "
          "rules can work only when Away or only when Home.", "/", marker="arming"),
    Entry("p20", 20, "7f7dbc7", "Loitering alerts",
          "Alert rule \"stayed in an area\" (30 s to 1 hour) for people and vehicles; the time inside shows on the "
          "live picture.", "/alerts", "admin"),
    Entry("p21", 21, "d6b98fe", "Animals",
          "Dogs, cats and birds on security cameras: \"Animals seen\" on the Live page, the Ask box, Reports and alert "
          "rules.", "/", marker="animals"),
    Entry("p22", 22, "938f2c3", "Scene questions",
          "Ask a camera a yes/no question, e.g. \"Is the gate open?\", every 5 minutes; an alert after two yes in a "
          "row. On the Alerts page.", "/alerts", "admin", marker="scene_questions"),
    Entry("p24", 24, "194cfc8", "Was this right?",
          "The Check page shows vehicles, animals, alerts and scene answers one by one: answer yes or no, and the "
          "app shows how often it is right on your cameras.", "/check", "admin", marker="checks"),
    Entry("p25", 25, "903f453", "What happened (Summary)",
          "The Summary page: what happened while you were away, last night or today. After switching to Home, a "
          "\"Welcome back\" note says it in one line.", "/summary"),
    Entry("p27", 27, "806e0ae", "Scene questions about one area, at some hours",
          "A scene question can look at one area drawn on the Live page, ask only at some hours, or only when Away.",
          "/alerts", "admin", marker="scene_questions.zone_id"),
    Entry("p28", 28, "5304118", "Newer PyTorch",
          "PyTorch 2.14.1 with CUDA 13.0: the NVIDIA driver must be version 580 or newer.", "/health", "admin"),
    Entry("p16", 16, "591615a", "Search recordings by words",
          "Recordings page, \"Find in recordings\": type e.g. \"man in a red shirt yesterday\" and play the moments "
          "found, one by one or all.", "/recordings"),
    Entry("p30", 30, "03ec5b3", "Camera picture check",
          "A covered, too dark, washed-out or blurry picture, or light from behind people, shows on the Live and "
          "Health pages. Settings, \"Check the picture\" checks at once.", "/health"),
    Entry("p31", 31, "f5349fd", "No movement alert",
          "The built-in rule \"No movement\": nobody seen on a kiosk camera for 4 hours between 08:00 and 22:00 "
          "while the house is set to Home. Change it on the Alerts page.", "/alerts"),
    Entry("p32", 32, "55f7758", "Help sign",
          "Hold one hand above your head for 3 seconds in front of a camera: a red help alert with a beep on every "
          "page.", "/alerts"),
    Entry("p33", 33, "8a0f48d", "Search recordings by photo",
          "Recordings page, \"or with a photo\": the moments with the person most alike come first.", "/recordings"),
    Entry("p34", 34, "c2bb71c", "Time-lapse of a day",
          "Recordings page: the day's recordings in a few minutes; \"Open this moment\" plays the real video there. "
          "Viewers get a copy with faces blurred.", "/recordings"),
    Entry("p35", 35, "b15c94f", "Update button",
          "Settings, \"Updates\": the app checks GitHub once a day; \"Update now\" backs up the database, gets the "
          "new version and restarts.", "/settings", "admin"),
    Entry("p36", 36, "ad9670e", "What's new and the first-run check",
          "This list, and the first-run check: after an update, one page says whether everything works (health, "
          "database, alert rules, recording, each camera's picture) and lets you try the help sign without an alert.",
          "/first-run", "admin"),
    Entry("p37", 37, "c8e1fe3", "Faster playback",
          "Recordings page: play at 2x, 4x or 8x and skip 10 seconds back or forward, with buttons under the picture "
          "or the keys space, arrows and 1 2 4 8. Also for the blurred recordings viewers see.", "/recordings"),
    Entry("p38", 38, "", "PC watch",
          "The Health page also watches the PC: the graphics card's temperature, each disk's health, and times the app "
          "or the PC stopped without warning (a power cut, the PC turned off without stop.bat). Problems show like "
          "camera problems.", "/health"),
]
FRESH_MARK = "tracks"   # made at this start: a new database, so a new install


def guess_before(created: set[str], added: set[str], entries: list[Entry] = ENTRIES) -> str:
    """The newest entry the last version had, judged by what this start had to add to the database."""
    newest = entries[-1].key
    if FRESH_MARK in created:
        return newest
    for i, e in enumerate(entries):
        if e.marker and (e.marker in created or e.marker in added):
            return entries[i - 1].key if i else ""
    return newest


def version() -> str | None:
    """The running version as git sees it: "b15c94f 2026-10-15" (None without git)."""
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%h %cs"], cwd=ROOT, capture_output=True, text=True,
                           timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class WhatsNew:
    def __init__(self, before: str, ran: str | None = None, version: str | None = None,
                 entries: list[Entry] = ENTRIES):
        self.entries = entries
        self.before = before        # the newest entry of the version that ran before this one
        self.ran = ran              # when this version first started ("2026-10-17 09:12")
        self.version = version
        keys = [e.key for e in entries]
        # An unknown key (an entry taken out of the list): nothing new rather than everything.
        self.first_new = keys.index(before) + 1 if before in keys else 0 if before == "" else len(keys)

    @classmethod
    def start(cls, path: Path, created: set[str] = frozenset(), added: set[str] = frozenset(),
              entries: list[Entry] = ENTRIES, now: datetime | None = None) -> "WhatsNew":
        """Read the state file, move it on when the version changed, write it back."""
        newest = entries[-1].key
        stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
        state = None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("last"), str):
                state = None
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            log.warning("what's new: could not read %s (%s); starting again", path, e)
        if state is None:
            state = {"last": newest, "before": guess_before(set(created), set(added), entries), "ran": stamp}
        elif state["last"] != newest:
            state = {"last": newest, "before": state["last"], "ran": stamp}
        state["version"] = version()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        except OSError as e:
            log.warning("what's new: could not write %s (%s)", path, e)
        w = cls(state.get("before", newest), state.get("ran"), state.get("version"), entries)
        if w.new_items(True):
            log.info("new in this version: %s", ", ".join(e["title"] for e in w.new_items(True)))
        return w

    def items(self, admin: bool) -> list[dict]:
        """Newest first, each with "new": added since the version that ran before."""
        out = [{**asdict(e), "new": i >= self.first_new} for i, e in enumerate(self.entries)
               if admin or e.who == "all"]
        for item in out:
            item.pop("marker")
        return out[::-1]

    def new_items(self, admin: bool) -> list[dict]:
        return [i for i in self.items(admin) if i["new"]]

    def summary(self, admin: bool) -> dict:
        items = self.items(admin)
        return {"version": self.version, "since": self.ran, "newest": self.entries[-1].key,
                "new": sum(i["new"] for i in items), "items": items}
