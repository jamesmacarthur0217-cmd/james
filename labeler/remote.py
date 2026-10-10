"""Self-checks for access from other devices and from another site over Tailscale (shown on the Health page).

Everything here only reads: whether login accounts and HTTPS are set up, this PC's Tailscale address and how many
other devices are in the Tailscale network (`tailscale status --json`), whether a Windows firewall rule lets them
reach the HTTPS port (`netsh advfirewall firewall show rule`, which works without administrator rights), and
when someone last logged in from a Tailscale address (the audit log). Nothing is changed: installing Tailscale
and adding the firewall rule are the user's steps (docs/SECOND-SITE.md).

The Tailscale and firewall answers take about a second to get, so they are kept for 5 minutes.
"""
from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .config import resolve

TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")   # every Tailscale address is in this range
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
GUIDE = "docs\\SECOND-SITE.md"
CACHE_S = 300


def _check(name: str, level: str, detail: str) -> dict:
    return {"name": name, "level": level, "detail": detail}


def tailscale_exe() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe"
    return str(candidate) if candidate.is_file() else None


def tailscale_status(run=subprocess.run) -> dict | None:
    """{"state", "ip", "peers", "online"}; None when Tailscale is not installed."""
    exe = tailscale_exe()
    if exe is None:
        return None
    try:
        r = run([exe, "status", "--json"], capture_output=True, text=True, timeout=10)
        data = json.loads(r.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"state": "not answering", "ip": None, "peers": 0, "online": 0}
    ips = [ip for ip in (data.get("Self") or {}).get("TailscaleIPs") or [] if "." in ip]
    peers = list((data.get("Peer") or {}).values())
    return {"state": data.get("BackendState") or "unknown", "ip": ips[0] if ips else None,
            "peers": len(peers), "online": sum(1 for p in peers if p.get("Online"))}


def parse_netsh(text: str) -> list[dict]:
    """`netsh advfirewall firewall show rule name=all dir=in` → one dict per rule ("Rule Name", "LocalPort"...)."""
    rules, cur = [], None
    for line in text.splitlines():
        if ":" not in line or line.startswith("-"):
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Rule Name":
            cur = {"Rule Name": value}
            rules.append(cur)
        elif cur is not None:
            cur[key] = value
    return rules


def _port_in(port: int, spec: str) -> bool:
    if spec.lower() == "any":
        return False   # "any port" rules are for one program; this check looks for the port rule from the guide
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.isdigit() and hi.isdigit() and int(lo) <= port <= int(hi):
                return True
        elif part.isdigit() and int(part) == port:
            return True
    return False


def _covers_tailscale(spec: str) -> bool:
    if spec.lower() == "any":
        return True
    for part in spec.split(","):
        try:
            if "-" in part:
                lo, hi = (ipaddress.ip_address(x.strip()) for x in part.split("-", 1))
                if lo.version == 4 and lo <= TAILSCALE_NET[-1] and hi >= TAILSCALE_NET[0]:
                    return True
            elif ipaddress.ip_network(part.strip(), strict=False).overlaps(TAILSCALE_NET):
                return True
        except ValueError:
            continue
    return False


def firewall_rules(port: int, text: str) -> list[dict]:
    """Enabled inbound TCP rules that allow this port to Tailscale addresses."""
    return [r for r in parse_netsh(text)
            if r.get("Enabled") == "Yes" and r.get("Direction") == "In" and r.get("Action") == "Allow"
            and r.get("Protocol") in ("TCP", "Any") and _port_in(port, r.get("LocalPort", ""))
            and _covers_tailscale(r.get("RemoteIP", "Any"))]


def read_firewall(run=subprocess.run) -> str | None:
    try:
        r = run(["netsh", "advfirewall", "firewall", "show", "rule", "name=all", "dir=in"],
                capture_output=True, text=True, timeout=20)
        return r.stdout if r.returncode == 0 and "Rule Name" in (r.stdout or "") else None
    except (OSError, subprocess.SubprocessError):
        return None


def last_tailscale_login(db) -> dict | None:
    for e in db.list_audit(500, action="login"):
        try:
            if ipaddress.ip_address(e["address"]) in TAILSCALE_NET:
                return e
        except ValueError:
            continue
    return None


CERT_WARN_DAYS = 30


def certificate_check(cert: Path, now=None) -> dict:
    """The HTTPS certificate's end date: browsers refuse an expired one, so other devices could not connect."""
    import datetime as dt

    from .tls import expires
    end = expires(cert)
    if end is None:
        return _check("HTTPS certificate", "ok", "made at the next start")
    now = now or dt.datetime.now(dt.timezone.utc)
    days = (end - now).days
    fix = f"delete {cert.name} and {cert.with_name('key.pem').name} in {cert.parent}, then restart the app"
    if days < 0:
        return _check("HTTPS certificate", "problem", f"expired on {end.date()}: other devices cannot connect; {fix}")
    if days < CERT_WARN_DAYS:
        return _check("HTTPS certificate", "warning", f"ends on {end.date()} (in {days} days); {fix} "
                                                      "(other devices then see a new fingerprint)")
    return _check("HTTPS certificate", "ok", f"valid until {end.date()} ({days} days)")


_cache: dict = {}
_lock = threading.Lock()


def _system(run) -> tuple[dict | None, str | None]:
    """Tailscale status and the firewall rules, at most every CACHE_S seconds."""
    with _lock:
        if _cache and time.monotonic() - _cache["at"] < CACHE_S and run is subprocess.run:
            return _cache["tailscale"], _cache["firewall"]
        ts = tailscale_status(run)
        fw = read_firewall(run) if ts is not None else None
        if run is subprocess.run:
            _cache.update(at=time.monotonic(), tailscale=ts, firewall=fw)
        return ts, fw


def checks(cfg: dict, db, run=subprocess.run) -> list[dict]:
    web = cfg["web"]
    users = db.count_users()
    if web["host"] in LOCAL_HOSTS:
        why = ("there is no login account yet; make one with tools\\users.py, then restart" if users == 0
               else "web.host is 127.0.0.1")
        return [_check("other devices", "off", f"this PC only ({why})")]
    out = []
    if users == 0:
        out.append(_check("other devices", "warning", "no login account: other devices cannot log in"))
    elif not web.get("https"):
        out.append(_check("other devices", "warning", f"plain HTTP on port {web['port']} (not encrypted); "
                                                       "turn on HTTPS on the Settings page"))
    else:
        from .tls import fingerprint
        fp = fingerprint(resolve(web["cert_file"]))
        out.append(_check("other devices", "ok", f"https://<this PC>:{web['https_port']}, {users} login account(s)"
                          + (f"; certificate fingerprint starts {fp[:23]}" if fp
                             else "; the HTTPS certificate is made at the next start")))

    if web.get("https"):
        out.append(certificate_check(resolve(web["cert_file"])))

    ts, fw = _system(run)
    if ts is None:
        out.append(_check("Tailscale (another site)", "off", f"not installed; only needed for another site ({GUIDE})"))
        return out
    if ts["state"] != "Running":
        out.append(_check("Tailscale (another site)", "warning", f"installed but {ts['state']}: open Tailscale "
                                                                   "and sign in"))
        return out
    login = last_tailscale_login(db)
    detail = (f"this PC is {ts['ip']}; {ts['peers']} other device(s) in your Tailscale network "
              f"({ts['online']} online now)")
    if login:
        detail += f"; last login over Tailscale: {login['time'].replace('T', ' ')} by {login['username']}"
    out.append(_check("Tailscale (another site)", "ok", detail))

    port = int(web.get("https_port", 8443)) if web.get("https") else int(web["port"])
    if fw is None:
        out.append(_check("firewall for Tailscale", "warning", "could not read the Windows firewall rules"))
    else:
        rules = firewall_rules(port, fw)
        if rules:
            out.append(_check("firewall for Tailscale", "ok", f"rule \"{rules[0]['Rule Name']}\" lets Tailscale "
                                                              f"devices reach port {port}"))
        elif ts["peers"]:
            out.append(_check("firewall for Tailscale", "warning", f"no rule lets Tailscale devices reach port {port}"
                                                                   f": step 3 in {GUIDE}"))
        else:
            out.append(_check("firewall for Tailscale", "off", f"no rule for port {port} yet; needed once another "
                                                               f"device joins (step 3 in {GUIDE})"))
    return out
