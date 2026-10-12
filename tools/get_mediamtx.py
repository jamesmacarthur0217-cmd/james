"""MediaMTX for cameras sent from other devices (labeler/receiver.py): check, download or copy from an offline package.

    python tools/get_mediamtx.py check             # exit 0 when mediamtx.exe is in receiver.dir (data/mediamtx)
    python tools/get_mediamtx.py download          # MediaMTX v1.21.2 from its GitHub releases (28.6 MB, MIT license)
    python tools/get_mediamtx.py pack <folder>     # copy into an offline package (tools/pack_offline.ps1)
    python tools/get_mediamtx.py unpack <folder>   # copy from an offline package (install.bat -Offline)

Only mediamtx.exe and its LICENSE are kept. The download is checked against the release's checksums.sha256
before anything is unpacked (the same check as tools/get_ffmpeg.py).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from get_ffmpeg import expected_sum, sha256  # noqa: E402

VERSION = "v1.21.2"
RELEASE = f"https://github.com/bluenviron/mediamtx/releases/download/{VERSION}/"
ZIP = f"mediamtx_{VERSION}_windows_amd64.zip"
FILES = {"mediamtx.exe": "mediamtx.exe", "LICENSE": "LICENSE-mediamtx.txt"}


def target() -> Path:
    """The folder of receiver.exe (data/mediamtx unless changed)."""
    from labeler.config import load_config, resolve
    return resolve(load_config()["receiver"]["exe"]).parent


def unpack_zip(zip_path: Path, out: Path) -> None:
    """Only mediamtx.exe and its license, flat into `out` (no paths from the archive are used)."""
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for name, saved in FILES.items():
            with z.open(names[name]) as src, open(out / saved, "wb") as dst:
                shutil.copyfileobj(src, dst)


def download(folder: Path, fetch=urllib.request.urlretrieve) -> None:
    with tempfile.TemporaryDirectory(prefix="vl_mediamtx_") as tmp:
        tmp = Path(tmp)
        print(f"Downloading {ZIP} (about 29 MB) ...", flush=True)
        fetch(RELEASE + "checksums.sha256", tmp / "checksums.sha256")
        want = expected_sum((tmp / "checksums.sha256").read_text(encoding="utf-8"), ZIP)
        if want is None:
            raise SystemExit(f"{ZIP} is not in the release's checksums: download by hand")
        fetch(RELEASE + ZIP, tmp / ZIP)
        if sha256(tmp / ZIP) != want:
            raise SystemExit("the download does not match its checksum: not used")
        unpack_zip(tmp / ZIP, folder)
    print(f"MediaMTX saved in {folder}")


def main() -> int:
    ap = argparse.ArgumentParser(description="MediaMTX for cameras sent from other devices")
    ap.add_argument("action", choices=["check", "download", "pack", "unpack"])
    ap.add_argument("folder", nargs="?", type=Path, help="the offline package (pack, unpack)")
    args = ap.parse_args()
    here = target()
    if args.action == "check":
        ok = (here / "mediamtx.exe").exists()
        print("MediaMTX: " + (f"ready in {here}" if ok else f"not in {here}"))
        return 0 if ok else 1
    if args.action == "download":
        download(here)
        return 0
    if not args.folder:
        ap.error("pack and unpack need the offline package's folder")
    src, dst = (here, args.folder / "mediamtx") if args.action == "pack" else (args.folder / "mediamtx", here)
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        if not (src / name).exists():
            print(f"{src / name} is missing", file=sys.stderr)
            return 1
        shutil.copy2(src / name, dst / name)
    print(f"MediaMTX copied to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
