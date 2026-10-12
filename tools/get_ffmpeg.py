"""FFmpeg for recording (labeler/recorder.py): check, download or copy from an offline package.

    python tools/get_ffmpeg.py check              # exit 0 when ffmpeg.exe and ffprobe.exe work and can make H.264
    python tools/get_ffmpeg.py download           # the LGPL build of FFmpeg 8.1 from BtbN's GitHub releases (~170 MB)
    python tools/get_ffmpeg.py pack <folder>      # copy into an offline package (tools/pack_offline.ps1)
    python tools/get_ffmpeg.py unpack <folder>    # copy from an offline package (install.bat -Offline)

Only ffmpeg.exe, ffprobe.exe and LICENSE.txt are kept, in the folder of recording.ffmpeg (data/ffmpeg). The
download is checked against the release's own checksums.sha256 before anything is unpacked.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RELEASE = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
ZIP = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
FILES = ("bin/ffmpeg.exe", "bin/ffprobe.exe", "LICENSE.txt")


def target() -> Path:
    """The folder of recording.ffmpeg (data/ffmpeg unless changed)."""
    from labeler.config import load_config
    from labeler.recorder import ffmpeg_path
    return ffmpeg_path(load_config()).parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_sum(checksums: str, name: str) -> str | None:
    """The line for `name` in a checksums.sha256 file ("<hash> *<name>" or "<hash>  <name>")."""
    for line in checksums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0].lower()
    return None


def unpack_zip(zip_path: Path, out: Path) -> None:
    """Only the three files, flat into `out` (no paths from the archive are trusted)."""
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        top = z.namelist()[0].split("/")[0]
        for name in FILES:
            with z.open(f"{top}/{name}") as src, open(out / Path(name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def check(folder: Path, run=subprocess.run) -> list[str]:
    """What is wrong (empty: all fine)."""
    problems = [f"{n} missing" for n in ("ffmpeg.exe", "ffprobe.exe") if not (folder / n).exists()]
    if problems:
        return problems
    from labeler.recorder import pick_encoder
    if pick_encoder(folder / "ffmpeg.exe", run=run) is None:
        return ["no working H.264 encoder"]
    return []


def download(folder: Path, fetch=urllib.request.urlretrieve) -> None:
    with tempfile.TemporaryDirectory(prefix="vl_ffmpeg_") as tmp:
        tmp = Path(tmp)
        print(f"Downloading {ZIP} (about 170 MB) ...", flush=True)
        fetch(RELEASE + "checksums.sha256", tmp / "checksums.sha256")
        want = expected_sum((tmp / "checksums.sha256").read_text(encoding="utf-8"), ZIP)
        if want is None:
            raise SystemExit(f"{ZIP} is not in the release's checksums: the release changed; download by hand")
        fetch(RELEASE + ZIP, tmp / ZIP)
        if sha256(tmp / ZIP) != want:
            raise SystemExit("the download does not match its checksum: not used")
        unpack_zip(tmp / ZIP, folder)
    print(f"FFmpeg saved in {folder}")


def main() -> int:
    ap = argparse.ArgumentParser(description="FFmpeg for recording")
    ap.add_argument("action", choices=["check", "download", "pack", "unpack"])
    ap.add_argument("folder", nargs="?", type=Path, help="the offline package (pack, unpack)")
    args = ap.parse_args()
    here = target()
    if args.action == "check":
        problems = check(here)
        print("FFmpeg: " + ("ready in " + str(here) if not problems else "; ".join(problems)))
        return 1 if problems else 0
    if args.action == "download":
        download(here)
        return 1 if check(here) else 0
    if not args.folder:
        ap.error("pack and unpack need the offline package's folder")
    src, dst = (here, args.folder / "ffmpeg") if args.action == "pack" else (args.folder / "ffmpeg", here)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg.exe", "ffprobe.exe", "LICENSE.txt"):
        if not (src / name).exists():
            print(f"{src / name} is missing", file=sys.stderr)
            return 1
        shutil.copy2(src / name, dst / name)
    print(f"FFmpeg copied to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
