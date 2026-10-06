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
