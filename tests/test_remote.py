"""Self-checks for other devices and Tailscale: they only read, and say which setup step is missing."""
import copy
import json
import subprocess

import pytest

from labeler import accounts, remote
from labeler.config import DEFAULTS

NETSH = """
Rule Name:                            Microsoft Teams
----------------------------------------------------------------------
Enabled:                              Yes
Direction:                            In
Profiles:                             Domain,Private,Public
LocalIP:                              Any
RemoteIP:                             Any
Protocol:                             UDP
LocalPort:                            Any
RemotePort:                           Any
Action:                               Allow

Rule Name:                            VisionLabeler HTTPS (Tailscale)
----------------------------------------------------------------------
Enabled:                              {enabled}
Direction:                            In
Profiles:                             Domain,Private,Public
LocalIP:                              Any
RemoteIP:                             {remote}
Protocol:                             TCP
LocalPort:                            {port}
RemotePort:                           Any
Action:                               Allow
"""


def _netsh(enabled="Yes", remote_ip="100.64.0.0/255.192.0.0", port="8443"):
    return NETSH.format(enabled=enabled, remote=remote_ip, port=port)


class FakeSystem:
    """Answers `tailscale status --json` and `netsh ...` like Windows would."""

    def __init__(self, peers=1, state="Running", netsh=None):
        self.peers, self.state, self.netsh = peers, state, netsh if netsh is not None else _netsh()

    def __call__(self, cmd, **kw):
        if cmd[0] == "netsh":
            return subprocess.CompletedProcess(cmd, 0, self.netsh, "")
        status = {"BackendState": self.state, "Self": {"TailscaleIPs": ["100.101.102.103", "fd7a::1"]},
                  "Peer": {f"k{i}": {"Online": i == 0} for i in range(self.peers)}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(status), "")


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(DEFAULTS)
    c["web"].update(host="0.0.0.0", https=True, cert_file=str(tmp_path / "cert.pem"))
    return c


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(remote, "tailscale_exe", lambda: "tailscale")


def _by_name(checks):
    return {c["name"]: c for c in checks}


@pytest.mark.parametrize("remote_ip,port,ok", [
    ("100.64.0.0/255.192.0.0", "8443", True),
    ("100.64.0.0/10", "8443", True),
    ("100.64.0.0-100.127.255.255", "8443", True),
    ("Any", "8000-9000", True),
    ("192.168.1.0/255.255.255.0", "8443", False),      # home network only: Tailscale devices are refused
    ("100.64.0.0/10", "8000", False),                   # another port
    ("100.64.0.0/10", "Any", False),                    # a program's "any port" rule is not this one
])
def test_firewall_rule_matching(remote_ip, port, ok):
    assert bool(remote.firewall_rules(8443, _netsh(remote_ip=remote_ip, port=port))) == ok


def test_disabled_rule_does_not_count():
    assert remote.firewall_rules(8443, _netsh(enabled="No")) == []


def test_this_pc_only_is_off(cfg, db):
    cfg["web"]["host"] = "127.0.0.1"
    [check] = remote.checks(cfg, db, FakeSystem())
    assert check["level"] == "off" and "no login account yet" in check["detail"]


def test_all_set_up(cfg, db, installed):
    db.add_user("tixuan", accounts.hash_password("correct horse battery"), "viewer")
    db.add_audit("tixuan", "viewer", "192.168.1.20", "login")
    db.add_audit("tixuan", "viewer", "100.88.1.2", "login")
    db.add_audit("tixuan", "viewer", "192.168.1.20", "login")
    c = _by_name(remote.checks(cfg, db, FakeSystem(peers=2)))
    assert c["other devices"]["level"] == "ok" and "made at the next start" in c["other devices"]["detail"]
    ts = c["Tailscale (another site)"]
    assert ts["level"] == "ok" and "100.101.102.103" in ts["detail"] and "2 other device(s)" in ts["detail"]
    assert "1 online" in ts["detail"] and "last login over Tailscale" in ts["detail"] and "tixuan" in ts["detail"]
    assert c["firewall for Tailscale"]["level"] == "ok"


def test_missing_steps_are_named(cfg, db, installed):
    c = _by_name(remote.checks(cfg, db, FakeSystem(peers=1, netsh=_netsh(port="9999"))))
    assert c["other devices"]["level"] == "warning"                       # no account yet
    assert c["firewall for Tailscale"]["level"] == "warning" and "step 3" in c["firewall for Tailscale"]["detail"]
    alone = _by_name(remote.checks(cfg, db, FakeSystem(peers=0, netsh=_netsh(port="9999"))))
    assert alone["firewall for Tailscale"]["level"] == "off"              # not needed until a device joins
    stopped = _by_name(remote.checks(cfg, db, FakeSystem(state="Stopped")))
    assert stopped["Tailscale (another site)"]["level"] == "warning"


def test_plain_http_is_a_warning(cfg, db, installed):
    db.add_user("ann", accounts.hash_password("correct horse battery"), "admin")
    cfg["web"]["https"] = False
    assert "plain HTTP" in _by_name(remote.checks(cfg, db, FakeSystem()))["other devices"]["detail"]


def test_certificate_end_date(tmp_path):
    import datetime as dt

    from labeler.tls import ensure_certificate, expires
    cert = tmp_path / "cert.pem"
    assert remote.certificate_check(cert)["detail"] == "made at the next start"
    ensure_certificate(cert, tmp_path / "key.pem")
    end = expires(cert)
    assert remote.certificate_check(cert)["level"] == "ok"
    soon = remote.certificate_check(cert, now=end - dt.timedelta(days=10))
    assert soon["level"] == "warning" and "in 10 days" in soon["detail"] and "restart" in soon["detail"]
    assert remote.certificate_check(cert, now=end + dt.timedelta(days=1))["level"] == "problem"


def test_without_tailscale(cfg, db, monkeypatch):
    monkeypatch.setattr(remote, "tailscale_exe", lambda: None)
    assert _by_name(remote.checks(cfg, db, FakeSystem()))["Tailscale (another site)"]["level"] == "off"
