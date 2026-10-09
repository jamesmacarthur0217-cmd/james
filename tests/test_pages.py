"""The web pages' JavaScript must at least parse: a syntax error silently breaks the whole page."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "labeler" / "static"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("page", sorted(p.name for p in STATIC.glob("*.html")))
def test_page_script_parses(page, tmp_path):
    js = "\n".join(re.findall(r"<script>(.*?)</script>", (STATIC / page).read_text(encoding="utf-8"), re.S))
    script = tmp_path / f"{page}.js"
    script.write_text(js, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("name", sorted(p.name for p in STATIC.glob("*.js")))
def test_shared_script_parses(name):
    result = subprocess.run(["node", "--check", str(STATIC / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


ROOT = STATIC.parent.parent
TEXT_FILES = [p for pattern in ("*.bat", "*.ps1", "*.py", "*.yaml", "*.yml", "*.md", "labeler/**/*.py",
                                "labeler/static/*", "tools/*", "tests/*.py", "docs/*", ".github/**/*")
              for p in ROOT.glob(pattern) if p.is_file() and ".venv" not in p.parts]


@pytest.mark.parametrize("path", sorted({p.relative_to(ROOT).as_posix() for p in TEXT_FILES}))
def test_no_control_characters(path):
    """A shell that turns "\\a" into a bell character once broke stop.bat (data\\app.pid became data<BEL>pp.pid,
    so it never found the app's process). Text files must hold no control characters except tab and newlines."""
    data = (ROOT / path).read_bytes()
    bad = [i for i, c in enumerate(data) if c < 32 and c not in (9, 10, 13)]
    assert not bad, f"control character {data[bad[0]]:#04x} at byte {bad[0]}"
