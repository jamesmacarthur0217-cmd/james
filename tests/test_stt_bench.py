"""The speech-to-text comparison tool (tools/stt_bench.py): scoring, the clip list, recording and a whole run, with
stand-in models and microphone (the real ones need their own packages and downloads)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import stt_bench as sb  # noqa: E402


def test_normalize():
    assert sb.normalize("Add 2 more tickets, for the 6:30 show!") == \
        "add two more tickets for the six thirty show".split()
    assert sb.normalize("Twenty-one") == ["twenty", "one"] and sb.normalize("21") == ["twenty", "one"]
    assert sb.normalize("I'd like 'that' one") == ["i'd", "like", "that", "one"]
    assert sb.normalize("Room 101") == ["room", "101"]   # 100 and up stay as digits


def test_wer_and_cer():
    assert sb.wer("print my receipt", "Print my receipt.") == 0
    assert sb.wer("print my receipt", "print receipt") == pytest.approx(1 / 3)       # one word missed
    assert sb.wer("print my receipt", "print my new receipt") == pytest.approx(1 / 3)  # one word added
    assert sb.wer("yes", "") == 1 and sb.wer("", "") == 0
    assert sb.cer("receipt", "receit") == pytest.approx(1 / 7)


def test_read_clips(tmp_path):
    (tmp_path / "clips.tsv").write_text("# my clips\n001.wav\tPrint my receipt\n\nphone.m4a\tYes \n", encoding="utf-8")
    assert sb.read_clips(tmp_path) == [{"file": "001.wav", "text": "Print my receipt"},
                                       {"file": "phone.m4a", "text": "Yes"}]
    (tmp_path / "clips.tsv").write_text("001.wav Print my receipt\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no tab"):
        sb.read_clips(tmp_path)


def test_wav_round_trip(tmp_path):
    samples = (np.sin(np.arange(sb.RATE) / 10) * 10000).astype(np.int16)
    sb.save_wav(tmp_path / "a.wav", samples)
    audio = sb.load_audio(tmp_path / "a.wav")
    assert audio.dtype == np.float32 and len(audio) == sb.RATE
    assert np.allclose(audio, samples / 32768)


def test_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sb, "DATA", tmp_path)
    loud = np.full(sb.RATE, 8000, np.int16)
    takes = iter([np.full(sb.RATE, 300, np.int16), loud, loud])   # the first take is too quiet and redone
    monkeypatch.setattr(sb, "record_one", lambda mic: next(takes))
    answers = iter(["", "n", "", "", "", ""])   # start, "keep it?" no, start, keep, start, keep
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    sb.record("near", ["Print my receipt", "Yes"], None)
    assert "very quiet" in capsys.readouterr().out
    assert sb.read_clips(tmp_path / "near") == [{"file": "001.wav", "text": "Print my receipt"},
                                                {"file": "002.wav", "text": "Yes"}]
    answers = iter(["", ""])
    takes = iter([loud])
    sb.record("near", ["Cancel"], None)   # a second session adds to the set
    assert sb.read_clips(tmp_path / "near")[-1] == {"file": "003.wav", "text": "Cancel"}


def test_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sb, "DATA", tmp_path)
    folder = tmp_path / "near"
    folder.mkdir()
    said = {"001.wav": "Print my receipt", "002.wav": "Add 2 more tickets", "003.wav": "Yes"}
    for i, name in enumerate(said):
        sb.save_wav(folder / name, np.full(sb.RATE * (i + 1), i + 1, np.int16))   # the clip's number in the samples
    (folder / "clips.tsv").write_text("".join(f"{n}\t{t}\n" for n, t in said.items()), encoding="utf-8")
    heard = {"good": ["print my receipt", "Add two more tickets.", "yes"],
             "deaf": ["print receipt", "at two more tickets", ""]}

    def load_model(spec, device, language, prompt):
        return (lambda audio: heard[spec][round(audio[0] * 32768) - 1]), "cpu"
    monkeypatch.setattr(sb, "load_model", load_model)
    sb.run("near", ["good", "deaf"], "auto", "en", None, worst=2)
    out = capsys.readouterr().out
    assert "3 clips" in out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith(("good ", "deaf "))}
    assert rows["good"][2] == "0.0%" and rows["good"][4] == "0"
    assert rows["deaf"][2] == "37.5%" and rows["deaf"][4] == "1"   # 3 words wrong of 8; one clip heard as nothing
    assert "003.wav  said:  Yes" in out and "(nothing)" in out   # the worst clip first
    report = json.loads(next(folder.glob("report-*.json")).read_text(encoding="utf-8"))
    assert report["language"] == "en" and report["models"]["deaf"]["heard"][2]["text"] == ""
    assert [c["duration"] for c in report["clips"]] == [1, 2, 3]


def test_run_needs_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "DATA", tmp_path)
    with pytest.raises(SystemExit, match="record first"):
        sb.run("none", ["whisper:small"], "auto", None, None, 5)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "clips.tsv").write_text("# nothing yet\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no clips"):
        sb.run("empty", ["whisper:small"], "auto", None, None, 5)


def test_unknown_model():
    with pytest.raises(SystemExit, match="unknown model"):
        sb.load_model("vosk", "auto", None, None)
    with pytest.raises(SystemExit, match="unknown model"):
        sb.load_model("whisper:", "auto", None, None)
