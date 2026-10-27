"""Compare speech-to-text models on your own voice, recorded at the kiosk: Whisper (faster-whisper) against NVIDIA
Parakeet (onnx-asr). Not part of the app; it needs its own packages:

    pip install faster-whisper "onnx-asr[cpu,hub]" sounddevice

1. Record: each phrase is shown, press Enter, say it, press Enter. Clips go to data/stt/<set>/ (16 kHz mono WAV)
   with clips.tsv (file, what was said). Record one set per situation to compare them (near, far, busy room):

    python tools/stt_bench.py devices                                # the microphones, with their numbers
    python tools/stt_bench.py record --set near --mic 1              # the built-in phrases
    python tools/stt_bench.py record --set far --phrases my_phrases.txt   # one phrase per line

   Clips recorded elsewhere (a phone, any format) work too: put them in data/stt/<set>/ and list them in clips.tsv,
   one line per clip: file name, a tab, what was said.

2. Compare:

    python tools/stt_bench.py run --set near
    python tools/stt_bench.py run --set near --models whisper:large-v3-turbo whisper:small parakeet:v2 parakeet:v3
    python tools/stt_bench.py run --set near --language en --prompt "Menu, receipt, loyalty card"

Models: whisper:<size> (tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3, ...) and
parakeet:v2 (English only) or parakeet:v3 (English and 24 other European languages). Downloads on first use.
--prompt gives Whisper the words the kiosk expects (Parakeet has no such option).
If Whisper stops on the GPU with a missing cublas64_12.dll or cudnn DLL (faster-whisper needs CUDA 12 libraries, the
app's PyTorch has CUDA 13), add --device cpu, or see "GPU" in the faster-whisper README. Parakeet uses the CPU unless
onnx-asr[gpu,hub] is installed.

Scores per model: WER (words wrong, lower is better), CER (letters wrong: use this for languages written without
spaces), clips where it heard nothing, seconds per clip, and how many times faster than real time. Then the clips it
got most wrong, side by side. Every transcript is saved to data/stt/<set>/report-<time>.json.
Numbers 0-99 count the same as words ("5" = "five"); write the phrases the way you would want them written.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "stt"
RATE = 16000
DEFAULT_MODELS = ["whisper:large-v3-turbo", "parakeet:v3"]
PARAKEET = {"v2": "nemo-parakeet-tdt-0.6b-v2", "v3": "nemo-parakeet-tdt-0.6b-v3"}
# Kiosk-like requests: short, with numbers, names and words that sound alike.
PHRASES = [
    "I would like to check in please",
    "Where is the nearest restroom",
    "Can I get a large coffee with oat milk",
    "My booking number is four seven two nine",
    "Cancel that and go back",
    "How much does the parking cost for three hours",
    "Print my receipt",
    "I need help from a member of staff",
    "Show me the menu in English",
    "Is the pharmacy open on Sunday",
    "Add two more tickets for the six thirty show",
    "No thank you that is all",
    "What time does the next bus to the airport leave",
    "Change my seat to the window",
    "Yes",
]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])


def normalize(text: str) -> list[str]:
    """Lower case, no punctuation (apostrophes kept), numbers 0-99 as words, "twenty-one" as two words."""
    text = re.sub(r"[^\w\s']", " ", text.lower().replace("-", " "))
    words = []
    for w in text.split():
        w = w.strip("'")
        if w.isdigit() and int(w) < 100:
            words.extend(number_words(int(w)).split())
        elif w:
            words.append(w)
    return words


def edits(ref: list, hyp: list) -> int:
    """Levenshtein distance: substitutions, insertions and deletions."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    return edits(r, h) / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    r, h = list("".join(normalize(ref))), list("".join(normalize(hyp)))
    return edits(r, h) / max(len(r), 1)


def read_clips(folder: Path) -> list[dict]:
    """clips.tsv: file name, tab, what was said. Blank lines and lines starting with # are skipped."""
    clips = []
    for line in (folder / "clips.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _, text = line.partition("\t")
        if not text.strip():
            raise SystemExit(f"clips.tsv: no tab between the file name and the text: {line!r}")
        clips.append({"file": name.strip(), "text": text.strip()})
    return clips


# --- recording ------------------------------------------------------------------------------------

def devices() -> None:
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "*" if i == sd.default.device[0] else " "
            print(f"{mark}{i:3d}  {d['name']}  ({int(d['default_samplerate'])} Hz)")
    print("* = the default microphone")


def record_one(mic: int | None) -> np.ndarray:
    import sounddevice as sd
    chunks = []
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", device=mic,
                        callback=lambda data, frames, t, status: chunks.append(data.copy())):
        input("  recording... Enter to stop ")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.int16)


def load_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32. Our own WAV files directly; anything else (MP3, M4A, other rates) through faster-whisper."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2):
                return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    from faster_whisper import decode_audio
    return decode_audio(str(path), sampling_rate=RATE)


def save_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.astype(np.int16).tobytes())


def record(set_name: str, phrases: list[str], mic: int | None) -> None:
    folder = DATA / set_name
    folder.mkdir(parents=True, exist_ok=True)
    tsv = folder / "clips.tsv"
    start = len(read_clips(folder)) if tsv.exists() else 0
    print(f"{len(phrases)} phrases into {folder}. Stand where a customer would and speak normally.")
    for k, text in enumerate(phrases):
        print(f"\n[{k + 1}/{len(phrases)}] {text}")
        while True:
            input("  Enter to start ")
            samples = record_one(mic)
            secs = len(samples) / RATE
            peak = int(np.abs(samples).max()) if len(samples) else 0
            note = ("  very quiet: move the microphone closer or raise its level in Windows" if peak < 1000 else
                    "  too loud, it clipped: lower the microphone level" if peak >= 32000 else "")
            print(f"  {secs:.1f} s, loudest {peak / 32767:.0%}{note}")
            if input("  keep it? [Y/n] ").strip().lower() != "n":
                break
        name = f"{start + k + 1:03d}.wav"
        save_wav(folder / name, samples)
        with open(tsv, "a", encoding="utf-8") as f:
            f.write(f"{name}\t{text}\n")
    print(f"\nDone. Compare with: python tools/stt_bench.py run --set {set_name}")


# --- the models -----------------------------------------------------------------------------------

def whisper(size: str, device: str, language: str | None, prompt: str | None):
    import ctranslate2
    from faster_whisper import WhisperModel
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = WhisperModel(size, device=device, compute_type="int8_float16" if device == "cuda" else "int8")

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, language=language, initial_prompt=prompt, beam_size=5,
                                       vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments)
    return transcribe, device


def parakeet(version: str, device: str):
    import onnx_asr
    import onnxruntime as ort
    if version not in PARAKEET:
        raise SystemExit(f"parakeet:{version}: choose one of " + ", ".join(f"parakeet:{v}" for v in PARAKEET))
    cuda = "CUDAExecutionProvider" in ort.get_available_providers() and device != "cpu"
    if device == "cuda" and not cuda:
        print("  parakeet: no CUDA in onnxruntime (pip install onnx-asr[gpu,hub]), using the CPU")
    model = onnx_asr.load_model(PARAKEET[version], quantization="int8",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda
                                else ["CPUExecutionProvider"])

    def transcribe(audio: np.ndarray) -> str:
        return model.recognize(audio, sample_rate=RATE)
    return transcribe, "cuda" if cuda else "cpu"


def load_model(spec: str, device: str, language: str | None, prompt: str | None):
    kind, _, variant = spec.partition(":")
    if kind == "whisper" and variant:
        return whisper(variant, device, language, prompt)
    if kind == "parakeet" and variant:
        return parakeet(variant, device)
    raise SystemExit(f"unknown model {spec!r}: use whisper:<size> or parakeet:v2 / parakeet:v3")


def score(clips: list[dict], heard: list[dict]) -> dict:
    return {"wer": sum(edits(normalize(c["text"]), normalize(h["text"])) for c, h in zip(clips, heard))
            / max(sum(len(normalize(c["text"])) for c in clips), 1),
            "cer": statistics.mean(cer(c["text"], h["text"]) for c, h in zip(clips, heard)),
            "nothing": sum(not normalize(h["text"]) for h in heard),
            "seconds": statistics.median(h["s"] for h in heard),
            "speed": sum(c["duration"] for c in clips) / max(sum(h["s"] for h in heard), 1e-9)}


def run(set_name: str, models: list[str], device: str, language: str | None, prompt: str | None,
        worst: int) -> None:
    folder = DATA / set_name
    if not (folder / "clips.tsv").exists():
        raise SystemExit(f"no {folder / 'clips.tsv'}: record first (python tools/stt_bench.py record --set {set_name})")
    clips = read_clips(folder)
    if not clips:
        raise SystemExit(f"no clips listed in {folder / 'clips.tsv'}")
    for c in clips:
        c["audio"] = load_audio(folder / c["file"])
        c["duration"] = len(c["audio"]) / RATE
    print(f"{len(clips)} clips in {folder}, {sum(c['duration'] for c in clips):.0f} s of speech")

    results = {}
    for spec in models:
        print(f"loading {spec} ...", flush=True)
        transcribe, used = load_model(spec, device, language, prompt)
        transcribe(clips[0]["audio"])   # warm up before timing
        heard = []
        for c in clips:
            t = time.perf_counter()
            text = transcribe(c["audio"])
            heard.append({"file": c["file"], "text": text, "s": time.perf_counter() - t})
        results[spec] = {"device": used, "heard": heard, **score(clips, heard)}

    print(f"\n{'model':26s} {'device':>6s} {'WER':>6s} {'CER':>6s} {'nothing':>8s} {'s/clip':>7s} {'speed':>7s}")
    for spec, r in results.items():
        print(f"{spec:26s} {r['device']:>6s} {r['wer']:6.1%} {r['cer']:6.1%} {r['nothing']:8d} "
              f"{r['seconds']:7.2f} {r['speed']:6.0f}x")
    print("WER, CER: lower is better. nothing: clips where it heard no words. speed: times faster than real time.")

    if worst:
        mean_wer = {c["file"]: statistics.mean(wer(c["text"], r["heard"][i]["text"]) for r in results.values())
                    for i, c in enumerate(clips)}
        bad = sorted(range(len(clips)), key=lambda i: -mean_wer[clips[i]["file"]])[:worst]
        print(f"\nThe {len(bad)} clips heard worst:")
        for i in bad:
            if mean_wer[clips[i]["file"]] == 0:
                break
            print(f"\n{clips[i]['file']}  said:  {clips[i]['text']}")
            for spec, r in results.items():
                print(f"  {spec:24s} {r['heard'][i]['text'] or '(nothing)'}")

    out = folder / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report = {"set": set_name, "language": language, "prompt": prompt,
              "clips": [{"file": c["file"], "text": c["text"], "duration": round(c["duration"], 2)} for c in clips],
              "models": results}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nEvery transcript: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list the microphones")
    rec = sub.add_parser("record", help="record clips of your voice")
    rec.add_argument("--set", default="kiosk", help="folder name under data/stt (default: kiosk)")
    rec.add_argument("--phrases", type=Path, help="text file, one phrase per line (default: built-in phrases)")
    rec.add_argument("--mic", type=int, help="microphone number from 'devices' (default: the Windows default)")
    r = sub.add_parser("run", help="transcribe the clips with each model and score them")
    r.add_argument("--set", default="kiosk")
    r.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    r.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    r.add_argument("--language", help="e.g. en (Whisper; default: detected per clip)")
    r.add_argument("--prompt", help="words the kiosk expects, for Whisper")
    r.add_argument("--worst", type=int, default=5, help="how many badly heard clips to show")
    args = ap.parse_args()
    if args.cmd == "devices":
        devices()
    elif args.cmd == "record":
        phrases = PHRASES
        if args.phrases:
            phrases = [p.strip() for p in args.phrases.read_text(encoding="utf-8").splitlines() if p.strip()]
        record(args.set, phrases, args.mic)
    else:
        run(args.set, args.models, args.device, args.language, args.prompt, args.worst)


if __name__ == "__main__":
    sys.exit(main())
