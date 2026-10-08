"""HTTPS for other devices: the certificate, the two listeners, secure cookies."""
import copy
import socket
import threading
import time

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from labeler import accounts
from labeler.config import DEFAULTS
from labeler.web import create_app
from run import _ServerPair, _web_servers
from labeler.tls import ensure_certificate

pytest.importorskip("cryptography")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_certificate_is_made_once_and_covers_this_pc(tmp_path):
    from cryptography import x509
    cert, key = tmp_path / "tls" / "cert.pem", tmp_path / "tls" / "key.pem"
    fp = ensure_certificate(cert, key)
    assert cert.exists() and key.exists() and len(fp.split(":")) == 32
    assert ensure_certificate(cert, key) == fp                               # reused, not remade
    san = x509.load_pem_x509_certificate(cert.read_bytes()).extensions.get_extension_for_class(
        x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    assert "127.0.0.1" in [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]


def test_without_https_there_is_one_plain_server():
    server = _web_servers(FastAPI(), {"host": "0.0.0.0", "port": 8000, "https": False})
    assert not isinstance(server, _ServerPair) and server.config.host == "0.0.0.0"


def test_with_https_plain_http_stays_on_this_pc(tmp_path):
    web = {**DEFAULTS["web"], "host": "0.0.0.0", "https": True,
           "cert_file": str(tmp_path / "c.pem"), "key_file": str(tmp_path / "k.pem")}
    pair = _web_servers(FastAPI(), web)
    assert pair.plain.config.host == "127.0.0.1" and pair.plain.config.port == 8000
    assert pair.secure.config.host == "0.0.0.0" and pair.secure.config.port == 8443
    assert pair.secure.config.ssl_certfile == str(tmp_path / "c.pem")


def test_both_listeners_serve_and_stop_together(tmp_path):
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    web = {**DEFAULTS["web"], "host": "127.0.0.1", "port": _free_port(), "https": True,
           "https_port": _free_port(), "cert_file": str(tmp_path / "c.pem"), "key_file": str(tmp_path / "k.pem")}
    pair = _web_servers(app, web)
    thread = threading.Thread(target=pair.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not (pair.plain.started and pair.secure.started) and time.time() < deadline:
        time.sleep(0.05)
    try:
        assert requests.get(f"http://127.0.0.1:{web['port']}/ping", timeout=5).json() == {"ok": True}
        # verify= the certificate itself: it must be valid for 127.0.0.1, not just "accepted anyway"
        r = requests.get(f"https://127.0.0.1:{web['https_port']}/ping", timeout=5, verify=web["cert_file"])
        assert r.json() == {"ok": True}
    finally:
        pair.plain.should_exit = True                                       # e.g. the Stop button
        thread.join(timeout=10)
    assert not thread.is_alive() and pair.secure.should_exit                 # HTTPS stopped with it


def test_login_cookie_is_secure_over_https(db, tmp_path):
    db.add_user("ann", accounts.hash_password("correct horse battery"), "admin")
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    body = {"username": "ann", "password": "correct horse battery"}
    https = TestClient(create_app(cfg, db, None), base_url="https://testserver", client=("192.168.1.20", 5000))
    plain = TestClient(create_app(cfg, db, None), client=("192.168.1.20", 5000))
    assert "secure" in https.post("/api/login", json=body).headers["set-cookie"].lower()
    assert "secure" not in plain.post("/api/login", json=body).headers["set-cookie"].lower()
