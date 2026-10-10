"""User accounts: password hashing, what each role may do, and slowing down password guessing.

Two roles. "admin" may do everything. "viewer" may watch the live view and search, nothing else: viewers get an
explicit list of allowed requests, so a new page or API is admin-only until it is added here on purpose.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

ROLES = ("admin", "viewer")
MIN_PASSWORD = 10

# scrypt (built into Python): a slow, salted hash, so a stolen database doesn't give away the passwords.
_N, _R, _P = 2 ** 14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(got, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


# Checked against when the user name doesn't exist, so a wrong name takes as long as a wrong password.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"the password needs at least {MIN_PASSWORD} characters")


# --- what a viewer may do --------------------------------------------------------------------------

# Open to anyone, logged in or not: the login page itself.
PUBLIC = {("GET", "/login"), ("POST", "/api/login"), ("GET", "/static/style.css"), ("GET", "/static/session.js")}

# Viewer: the Live page, search and questions, alert pop-ups, the health check. Exact paths, or prefixes ending in "/".
VIEWER = {
    "GET": ["/", "/stream", "/api/live", "/api/live/thumb/", "/api/status", "/api/schema", "/api/cameras",
            "/api/tracks", "/api/tracks/", "/api/search", "/api/alerts/events", "/api/me", "/crops/", "/static/",
            "/health", "/api/health", "/api/today"],
    "POST": ["/api/ask", "/api/logout", "/api/search/photo"],
}


def _match(path: str, rules: list[str]) -> bool:
    return any(path == r or (r.endswith("/") and r != "/" and path.startswith(r)) for r in rules)


def is_public(method: str, path: str) -> bool:
    return (method, path) in PUBLIC


def allowed(role: str, method: str, path: str) -> bool:
    if role == "admin":
        return True
    if role == "viewer":
        if method == "GET" and path.startswith("/api/tracks/") and not path.endswith(("/similar", "/timeline")):
            return False   # only look-alikes and the timeline below a person, nothing else about one person
        return _match(path, VIEWER.get(method, []))
    return False


# --- slowing down password guessing ----------------------------------------------------------------

class LoginLimiter:
    """After `max_failures` wrong passwords from one address within `window_s`, refuse logins from it
    until the window has passed."""

    def __init__(self, max_failures: int = 5, window_s: float = 300):
        self.max_failures, self.window_s = max_failures, window_s
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, address: str, now: float) -> deque[float]:
        q = self._failures[address]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def blocked_for(self, address: str, now: float | None = None) -> float:
        """Seconds until this address may try again (0 = allowed now)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._recent(address, now)
            return self.window_s - (now - q[0]) if len(q) >= self.max_failures else 0.0

    def failed(self, address: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._recent(address, now).append(now)

    def succeeded(self, address: str) -> None:
        with self._lock:
            self._failures.pop(address, None)
