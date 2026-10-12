"""tools/get_ffmpeg.py: the checksum is checked before unpacking, only the three files are taken."""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import get_ffmpeg  # noqa: E402


def _zip(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        top = "ffmpeg-n8.1-latest-win64-lgpl-8.1"
        for name in ("bin/ffmpeg.exe", "bin/ffprobe.exe", "bin/ffplay.exe", "LICENSE.txt", "doc/x.html"):
            z.writestr(f"{top}/{name}", name)
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def test_expected_sum():
    sums = "aa11  other.zip\nBB22 *ffmpeg-n8.1-latest-win64-lgpl-8.1.zip\n"
    assert get_ffmpeg.expected_sum(sums, get_ffmpeg.ZIP) == "bb22"
    assert get_ffmpeg.expected_sum(sums, "missing.zip") is None


def _fetcher(tmp_path, good=True):
    data = _zip(tmp_path / "source.zip")
    digest = hashlib.sha256(data).hexdigest() if good else "0" * 64

    def fetch(url, dest):
        dest = Path(dest)
        if url.endswith("checksums.sha256"):
            dest.write_text(f"{digest} *{get_ffmpeg.ZIP}\n", encoding="utf-8")
        else:
            dest.write_bytes(data)
    return fetch


def test_download_checks_the_sum_and_takes_only_three_files(tmp_path):
    out = tmp_path / "ffmpeg"
    get_ffmpeg.download(out, fetch=_fetcher(tmp_path))
    assert sorted(p.name for p in out.iterdir()) == ["LICENSE.txt", "ffmpeg.exe", "ffprobe.exe"]
    assert (out / "ffmpeg.exe").read_text() == "bin/ffmpeg.exe"


def test_a_download_that_does_not_match_is_not_used(tmp_path):
    out = tmp_path / "ffmpeg"
    with pytest.raises(SystemExit, match="checksum"):
        get_ffmpeg.download(out, fetch=_fetcher(tmp_path, good=False))
    assert not out.exists()


def test_check_says_what_is_missing(tmp_path):
    assert get_ffmpeg.check(tmp_path) == ["ffmpeg.exe missing", "ffprobe.exe missing"]
