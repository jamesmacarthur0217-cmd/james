"""Update button (Phase 35): bring this copy up to date from its GitHub repository, from the Settings page.

The app's folder is a git copy of the user's repository (`origin`). Once a day (and on "Check now") it fetches and
lists what is new (the commit titles). "Update now" (admins, asks first, audited) does, in this order, stopping at
the first problem with the running app untouched:
  1. checks that nothing was changed by hand (tracked files) and that this copy has no commits of its own, so the
     update can only be a fast-forward;
  2. backs up the database (PostgreSQL; labeler/backups.py), as stop.bat does;
  3. `git pull --ff-only`;
  4. `pip install -r requirements.txt` only when requirements.txt changed; if that fails, the files go back to the
     version that was running (`git reset --hard` to it: safe, as step 1 found no changes by hand);
  5. the caller restarts the app (exit 4: the supervisor starts it again at once, like "Restart now").
Git uses the sign-in the user already has (Git Credential Manager); it is never asked to prompt from here.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT

log = logging.getLogger(__name__)

CHECK_EVERY_S = 24 * 3600
MAX_TITLES = 30


def find_git() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    for p in (r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files\Git\bin\git.exe"):
        if Path(p).is_file():
            return p
    return None


class Updater:
    def __init__(self, cfg: dict, root: Path = ROOT, git: str | None = None, run=subprocess.run, backup=None,
                 pip=None, clock=time.time):
        self.cfg, self.root, self.run, self.clock = cfg, Path(root), run, clock
        self.git_exe = git or find_git()
        self.backup = backup          # () -> bool, the database backup before updating (None: none)
        self.pip = pip                # () -> (ok, message); default: pip install -r requirements.txt
        self.state: dict = {"checked_at": None}
        self._lock = threading.Lock()
        self.busy = False

    # --- git -------------------------------------------------------------------------------------------------------

    def git(self, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "LC_ALL": "C"}
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        return self.run([self.git_exe, *args], cwd=self.root, capture_output=True, text=True, timeout=timeout,
                        env=env, creationflags=flags)

    def _out(self, *args: str) -> str:
        r = self.git(*args)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "git failed").strip().splitlines()[-1][:300])
        return r.stdout.strip()

    # --- checking --------------------------------------------------------------------------------------------------

    def check(self, fetch: bool = True) -> dict:
        """What is new on GitHub: {"available", "behind", "changes", "hand_changes", "own_commits", "error", ...}."""
        state = {"checked_at": datetime.now().isoformat(timespec="seconds"), "available": False, "behind": 0,
                 "changes": [], "hand_changes": [], "own_commits": 0, "error": None}
        try:
            if not self.git_exe:
                raise RuntimeError("Git is not installed on this PC")
            if self._out("rev-parse", "--is-inside-work-tree") != "true":
                raise RuntimeError("this copy is not a git copy (installed from a package)")
            state["branch"] = self._out("rev-parse", "--abbrev-ref", "HEAD")
            upstream = self._out("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
            if fetch:
                self._out("fetch", "--quiet", upstream.split("/", 1)[0])
            state["version"] = self._out("log", "-1", "--format=%h %s")[:120]
            state["behind"] = int(self._out("rev-list", "--count", "HEAD..@{u}"))
            state["own_commits"] = int(self._out("rev-list", "--count", "@{u}..HEAD"))
            if state["behind"]:
                state["changes"] = self._out("log", f"-{MAX_TITLES}", "--format=%s", "HEAD..@{u}").splitlines()
            r = self.git("status", "--porcelain", "--untracked-files=no")   # " M file": not stripped (the space counts)
            state["hand_changes"] = [line[3:] for line in r.stdout.splitlines() if line.strip()]
            state["available"] = True
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as e:
            state["error"] = str(e)
        with self._lock:
            self.state = state
        return state

    def problem(self, state: dict) -> str | None:
        """Why updating cannot go ahead, in plain words, or None."""
        if not state.get("available"):
            return state.get("error") or "not checked yet"
        if state["hand_changes"]:
            return ("files were changed by hand (" + ", ".join(state["hand_changes"][:5]) + "): undo or keep them "
                    "with git first; the update button only adds the new version on top")
        if state["own_commits"]:
            return "this copy has changes of its own that are not on GitHub; update it with git by hand"
        if not state["behind"]:
            return "already up to date"
        return None

    # --- updating --------------------------------------------------------------------------------------------------

    def _pip(self) -> tuple[bool, str]:
        if self.pip is not None:
            return self.pip()
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        r = self.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=self.root,
                     capture_output=True, text=True, timeout=1800, creationflags=flags)
        return r.returncode == 0, (r.stderr or r.stdout or "").strip()[-300:]

    def update(self) -> dict:
        """Update to the newest version (steps 1-4 above). {"ok", "message", "changes", "requirements"}."""
        with self._lock:
            if self.busy:
                return {"ok": False, "message": "an update is already running"}
            self.busy = True
        try:
            state = self.check(fetch=True)
            why = self.problem(state)
            if why:
                return {"ok": False, "message": why}
            if self.backup is not None and not self.backup():
                return {"ok": False, "message": "the database backup failed, so nothing was changed (see the log)"}
            old = self._out("rev-parse", "HEAD")
            r = self.git("pull", "--ff-only", "--quiet", timeout=300)
            if r.returncode != 0:
                return {"ok": False, "message": "git could not update: " + (r.stderr or r.stdout).strip()[-300:]}
            new = self._out("rev-parse", "HEAD")
            reqs = bool(self._out("diff", "--name-only", old, new, "--", "requirements.txt"))
            if reqs:
                ok, msg = self._pip()
                if not ok:
                    self.git("reset", "--hard", old)
                    return {"ok": False, "message": "installing the new Python packages failed, so the old version "
                                                    f"was put back: {msg}"}
            log.info("updated from %s to %s (%d changes)", old[:7], new[:7], len(state["changes"]))
            self.check(fetch=False)
            return {"ok": True, "message": f"updated: {len(state['changes'])} changes", "changes": state["changes"],
                    "requirements": reqs, "from": old[:7], "to": new[:7]}
        finally:
            with self._lock:
                self.busy = False

    # --- once a day ------------------------------------------------------------------------------------------------

    def loop(self, stop: threading.Event, first_after_s: float = 120) -> None:
        if stop.wait(first_after_s):
            return
        while True:
            try:
                self.check(fetch=True)
            except Exception:
                log.exception("checking for updates failed")
            if stop.wait(CHECK_EVERY_S):
                return
