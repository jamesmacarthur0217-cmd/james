"""A self-signed HTTPS certificate for other devices on the network (web.https in config.yaml).

Made once, valid for this PC's name and network addresses. Browsers warn about a self-signed certificate the
first time; the fingerprint logged at startup lets you check that the warning is really about this PC.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import socket
from pathlib import Path

log = logging.getLogger(__name__)
VALID_DAYS = 825   # the longest validity browsers accept for a certificate


def local_names() -> tuple[list[str], list[str]]:
    """(host names, IP addresses) other devices may use to reach this PC."""
    host = socket.gethostname()
    names = sorted({host, host.lower(), "localhost"})
    ips = {"127.0.0.1"}
    try:
        ips.update(socket.gethostbyname_ex(host)[2])
    except OSError:
        pass
    return names, sorted(ips)


def ensure_certificate(cert_path: Path, key_path: Path) -> str:
    """Create the certificate and key if missing; returns the certificate's SHA-256 fingerprint."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    if not (cert_path.exists() and key_path.exists()):
        names, ips = local_names()
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"VisionLabeler on {names[0]}")])
        now = dt.datetime.now(dt.timezone.utc)
        cert = (x509.CertificateBuilder()
                .subject_name(subject).issuer_name(subject)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(minutes=5))
                .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
                .add_extension(x509.SubjectAlternativeName(
                    [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]),
                    critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .sign(key, hashes.SHA256()))
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        log.info("made a self-signed HTTPS certificate for %s (valid %d days): %s",
                 ", ".join(names + ips), VALID_DAYS, cert_path)
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    if cert.not_valid_after_utc < dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=14):
        log.warning("the HTTPS certificate %s expires on %s; delete it and its key to get a new one",
                    cert_path, cert.not_valid_after_utc.date())
    return cert.fingerprint(hashes.SHA256()).hex(":").upper()
