"""Installer helpers: model checks, the offline package's model copy, and the PowerShell scripts themselves."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import models  # noqa: E402

SCRIPTS = ["install.ps1", "tools/pack_offline.ps1"]
NAMES = {"detector": "yolo-test.pt", "huggingface": ["org/emb", "org/age"], "ollama": "qwen3-vl:4b-instruct"}


def _fake_hf(cache: Path, repo: str, weights=True):
    snap = models.hf_folder(repo, cache) / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    if weights:
        (snap / "model.safetensors").write_bytes(b"w" * 10)


def _fake_ollama(root: Path, model: str, blobs=("sha256:aa", "sha256:bb")):
    manifest = models.ollama_manifest(model, root)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"config": {"digest": blobs[0]},
                                    "layers": [{"digest": d} for d in blobs] + [{"digest": blobs[-1]}]}))
    (root / "blobs").mkdir()
    for d in blobs:
        (root / "blobs" / d.replace(":", "-")).write_bytes(b"blob")


def test_model_names_come_from_config_yaml():
    n = models.names()
    assert n["detector"].endswith(".pt") and n["ollama"] and len(n["huggingface"]) == 2


def test_hf_model_needs_its_weights(tmp_path):
    _fake_hf(tmp_path, "org/emb")
    _fake_hf(tmp_path, "org/age", weights=False)
    assert models.hf_present("org/emb", tmp_path)
    assert not models.hf_present("org/age", tmp_path)        # only config.json: an interrupted download
    assert not models.hf_present("org/none", tmp_path)


def test_ollama_model_needs_every_blob(tmp_path):
    _fake_ollama(tmp_path, "qwen3-vl:4b-instruct")
    manifest = models.ollama_manifest("qwen3-vl:4b-instruct", tmp_path)
    assert manifest.as_posix().endswith("manifests/registry.ollama.ai/library/qwen3-vl/4b-instruct")
    assert models.ollama_blobs(manifest) == ["sha256-aa", "sha256-bb"]           # repeated layers once
    assert models.ollama_present("qwen3-vl:4b-instruct", tmp_path)
    (tmp_path / "blobs" / "sha256-bb").unlink()
    assert not models.ollama_present("qwen3-vl:4b-instruct", tmp_path)


def test_pack_then_unpack_on_an_empty_pc(tmp_path, monkeypatch):
    here, there = tmp_path / "here", tmp_path / "there"
    for p in (here, there):
        p.mkdir()
    (here / "yolo-test.pt").write_bytes(b"yolo")
    for repo in NAMES["huggingface"]:
        _fake_hf(here / "hf", repo)
    _fake_ollama(here / "ollama", NAMES["ollama"])
    monkeypatch.setattr(models, "ROOT", here)
    models.pack(NAMES, tmp_path / "package", here / "hf", here / "ollama")

    monkeypatch.setattr(models, "ROOT", there)                 # the new PC: nothing there yet
    models.unpack(NAMES, tmp_path / "package", there / "hf", there / "ollama")
    assert (there / "yolo-test.pt").read_bytes() == b"yolo"
    assert all(models.hf_present(repo, there / "hf") for repo in NAMES["huggingface"])
    assert models.ollama_present(NAMES["ollama"], there / "ollama")


def test_pack_refuses_when_a_model_is_missing(tmp_path, monkeypatch):
    (tmp_path / "yolo-test.pt").write_bytes(b"yolo")
    _fake_hf(tmp_path / "hf", "org/emb")
    monkeypatch.setattr(models, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="org/age is not on this PC"):
        models.pack(NAMES, tmp_path / "package", tmp_path / "hf", tmp_path / "ollama")


@pytest.mark.parametrize("script", SCRIPTS)
def test_powershell_scripts_are_ascii(script):
    """Windows PowerShell 5.1 reads a script without a byte-order mark as the local code page, so a "•" or "→"
    would become garbage (or break a string)."""
    data = (ROOT / script).read_bytes()
    assert all(b < 128 for b in data), f"non-ASCII byte at {next(i for i, b in enumerate(data) if b >= 128)}"


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell not available")
@pytest.mark.parametrize("script", SCRIPTS)
def test_powershell_scripts_parse(script):
    check = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile('{}', [ref]$null, [ref]$e); "
             "if ($e) {{ $e | ForEach-Object {{ $_.ToString() }}; exit 1 }}").format(ROOT / script)
    r = subprocess.run(["powershell", "-NoProfile", "-Command", check], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_batch_files_have_windows_line_endings():
    for bat in ROOT.glob("*.bat"):
        data = bat.read_bytes()
        assert data.count(b"\n") == data.count(b"\r\n"), bat.name
