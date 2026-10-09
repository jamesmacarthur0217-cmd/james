"""The models the app needs, for the installer: are they on this PC, download them, or copy them to and from an
offline package (for a PC without internet).

    python tools/models.py check              # lists each model; exit code 1 if any is missing
    python tools/models.py download           # the detector and the Hugging Face models (Ollama: ollama pull)
    python tools/models.py pack <folder>      # copies all of them into <folder>/models
    python tools/models.py unpack <folder>    # copies them from <folder>/models to where the app looks

Model names come from config.yaml (detector.model, embeddings.model, age_model.model, vlm.model).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeler.config import DEFAULTS, _merge  # noqa: E402

WEIGHTS = (".safetensors", ".bin")


def names(config: Path = ROOT / "config.yaml") -> dict:
    """The model names, without loading .env (the installer runs this before .env exists)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if config.exists():
        _merge(cfg, yaml.safe_load(config.read_text(encoding="utf-8")) or {})
    return {"detector": cfg["detector"]["model"], "huggingface": [cfg["embeddings"]["model"], cfg["age_model"]["model"]],
            "ollama": cfg["vlm"]["model"]}


# --- Hugging Face (SigLIP 2 embeddings, age model) ------------------------------------------------

def hf_cache() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def hf_folder(repo: str, cache: Path | None = None) -> Path:
    return (cache or hf_cache()) / ("models--" + repo.replace("/", "--"))


def hf_present(repo: str, cache: Path | None = None) -> bool:
    """A snapshot with weights in it (config.json alone is not enough)."""
    snaps = hf_folder(repo, cache) / "snapshots"
    return snaps.is_dir() and any(p.name.endswith(WEIGHTS) for p in snaps.rglob("*"))


def hf_download(repo: str) -> None:
    """The same files the app loads (labeler/embed.py, labeler/age.py), not the whole repository."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor
    AutoConfig.from_pretrained(repo)
    AutoModel.from_pretrained(repo)
    try:
        AutoProcessor.from_pretrained(repo)
    except Exception:   # an image classifier has only an image processor
        AutoImageProcessor.from_pretrained(repo)


# --- Ollama (the vision-language model) -----------------------------------------------------------

def ollama_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")


def ollama_manifest(model: str, root: Path | None = None) -> Path:
    name, _, tag = model.partition(":")
    if "/" not in name:
        name = "library/" + name
    return (root or ollama_dir()) / "manifests" / "registry.ollama.ai" / name / (tag or "latest")


def ollama_blobs(manifest: Path) -> list[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    layers = data.get("layers", []) + ([data["config"]] if data.get("config") else [])
    return list(dict.fromkeys(layer["digest"].replace(":", "-") for layer in layers))   # a layer can repeat


def ollama_present(model: str, root: Path | None = None) -> bool:
    manifest = ollama_manifest(model, root)
    if not manifest.is_file():
        return False
    blobs = (root or ollama_dir()) / "blobs"
    return all((blobs / b).is_file() for b in ollama_blobs(manifest))


# --- commands -------------------------------------------------------------------------------------

def check(n: dict) -> list[tuple[str, bool]]:
    out = [(f"detector {n['detector']}", (ROOT / n["detector"]).is_file())]
    out += [(f"Hugging Face {repo}", hf_present(repo)) for repo in n["huggingface"]]
    out.append((f"Ollama {n['ollama']}", ollama_present(n["ollama"])))
    return out


def download(n: dict) -> None:
    if not (ROOT / n["detector"]).is_file():
        print(f"downloading {n['detector']}...", flush=True)
        os.chdir(ROOT)   # Ultralytics saves the weights in the current folder
        from ultralytics import YOLO
        YOLO(n["detector"])
    for repo in n["huggingface"]:
        if not hf_present(repo):
            print(f"downloading {repo}...", flush=True)
            hf_download(repo)


def _long(p: Path) -> str:
    """Windows: an extended-length path, so copies deep inside a long folder pass the 260-character limit
    (Hugging Face cache paths are long already)."""
    p = Path(os.path.abspath(p))
    return "\\\\?\\" + str(p) if os.name == "nt" and not str(p).startswith("\\\\") else str(p)


def _copy_tree(src: Path, dst: Path) -> None:
    # Symlinks in the HF cache are copied as files. ".no_exist" only remembers files a model does not have.
    shutil.copytree(_long(src), _long(dst), dirs_exist_ok=True, ignore=shutil.ignore_patterns(".no_exist"))


def _copy_file(src: Path, dst: Path) -> None:
    shutil.copy2(_long(src), _long(dst))


def pack(n: dict, folder: Path, hf_root: Path | None = None, ollama_root: Path | None = None) -> None:
    out = folder / "models"
    out.mkdir(parents=True, exist_ok=True)
    _copy_file(ROOT / n["detector"], out / Path(n["detector"]).name)
    for repo in n["huggingface"]:
        if not hf_present(repo, hf_root):
            raise SystemExit(f"{repo} is not on this PC; run: python tools/models.py download")
        _copy_tree(hf_folder(repo, hf_root), out / "huggingface" / hf_folder(repo, hf_root).name)
    root = ollama_root or ollama_dir()
    manifest = ollama_manifest(n["ollama"], root)
    if not ollama_present(n["ollama"], root):
        raise SystemExit(f"Ollama model {n['ollama']} is not on this PC; run: ollama pull {n['ollama']}")
    target = ollama_manifest(n["ollama"], out / "ollama")
    target.parent.mkdir(parents=True, exist_ok=True)
    _copy_file(manifest, target)
    (out / "ollama" / "blobs").mkdir(parents=True, exist_ok=True)
    for blob in ollama_blobs(manifest):
        print(f"  Ollama {blob[:19]}... ({(root / 'blobs' / blob).stat().st_size / 1e9:.1f} GB)", flush=True)
        _copy_file(root / "blobs" / blob, out / "ollama" / "blobs" / blob)


def unpack(n: dict, folder: Path, hf_root: Path | None = None, ollama_root: Path | None = None) -> None:
    src = folder / "models"
    if not (ROOT / n["detector"]).is_file():
        _copy_file(src / Path(n["detector"]).name, ROOT / n["detector"])
    for repo in n["huggingface"]:
        if not hf_present(repo, hf_root):
            _copy_tree(src / "huggingface" / hf_folder(repo, hf_root).name, hf_folder(repo, hf_root))
    root = ollama_root or ollama_dir()
    if not ollama_present(n["ollama"], root):
        _copy_tree(src / "ollama", root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "download", "pack", "unpack"])
    ap.add_argument("folder", nargs="?", type=Path)
    args = ap.parse_args()
    n = names()
    if args.command in ("pack", "unpack") and args.folder is None:
        ap.error(f"{args.command} needs a folder")
    if args.command == "download":
        download(n)
    elif args.command == "pack":
        pack(n, args.folder.resolve())
    elif args.command == "unpack":
        unpack(n, args.folder.resolve())
    results = check(n)
    for what, ok in results:
        print(f"  {'OK     ' if ok else 'MISSING'} {what}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
