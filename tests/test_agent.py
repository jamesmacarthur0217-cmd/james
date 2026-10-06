"""Natural-language search: plan cleaning and execution (the model itself is faked)."""
from datetime import datetime

import numpy as np

from labeler.agent import EARLIEST, QueryAgent, clean_plan, explain

NOW = datetime(2026, 10, 7, 17, 0)
CAMS = ["cam1", "entrance"]


def test_conditions_must_be_in_the_vocabulary_and_mentioned():
    raw = {"conditions": [{"attribute": "upper_color", "value": "red"},
                          {"attribute": "headwear", "value": "none"},          # not asked for
                          {"attribute": "upper_type", "value": "shirt"},       # not mentioned
                          {"attribute": "upper_type", "value": "fur coat"}],   # not in the vocabulary
           "appearance": "", "camera": "", "start": "", "end": ""}
    plan = clean_plan(raw, CAMS, NOW, "who wore a red fur coat?")
    assert plan["conditions"] == [["upper_color", "red"]]
    assert plan["appearance"] == "fur coat"


def test_synonyms_count_as_mentions():
    raw = {"conditions": [{"attribute": "apparent_gender", "value": "woman"},
                          {"attribute": "age_group", "value": "child"},
                          {"attribute": "carrying", "value": "luggage"},
                          {"attribute": "upper_color", "value": "gray"}]}
    plan = clean_plan(raw, CAMS, NOW, "ladies and kids with a suitcase in grey")
    assert plan["conditions"] == [["apparent_gender", "woman"], ["age_group", "child"],
                                  ["carrying", "luggage"], ["upper_color", "gray"]]


def test_age_comes_from_numbers_in_the_question():
    raw = {"conditions": [{"attribute": "age_range", "value": "35-44"}]}          # model guessed wrong
    assert clean_plan(raw, CAMS, NOW, "a 30 year old man")["conditions"] == [["age_range", "25-34"]]
    assert clean_plan({}, CAMS, NOW, "people aged 70")["conditions"] == [["age_range", "65+"]]
    assert clean_plan(raw, CAMS, NOW, "an older man")["conditions"] == []


def test_camera_must_be_named():
    assert clean_plan({"camera": "cam1"}, CAMS, NOW, "who wore red?")["camera"] == ""
    assert clean_plan({"camera": "entrance"}, CAMS, NOW, "red top on the entrance camera")["camera"] == "entrance"
    assert clean_plan({"camera": "garage"}, CAMS, NOW, "garage")["camera"] == ""


def test_time_ranges():
    plan = clean_plan({"start": "2026-10-07T15:00", "end": ""}, CAMS, NOW, "after 3pm")
    assert plan["start"] == datetime(2026, 10, 7, 15) and plan["end"] == NOW
    plan = clean_plan({"start": "", "end": "2026-10-07T09:00"}, CAMS, NOW, "before 9am")
    assert plan["start"] == EARLIEST
    plan = clean_plan({"start": "2026-10-07T12:00", "end": "2026-10-07T09:00"}, CAMS, NOW, "x")
    assert plan["start"] < plan["end"]
    assert clean_plan({"start": "3pm"}, CAMS, NOW, "x")["start"] is None


def test_explanation_reads_naturally():
    plan = clean_plan({"start": "2026-10-07T15:00", "conditions": [{"attribute": "upper_color", "value": "red"}],
                       "appearance": "striped scarf"}, CAMS, NOW, "red top with a striped scarf after 3pm")
    text = explain(plan)
    assert "07 Oct 15:00" in text and "upper color = red" in text and "striped scarf" in text


class FakeEmbedder:
    model_name = "m"

    def text(self, query):
        return np.array([1, 0, 0, 0], np.float32)


def test_run_filters_and_ranks(db, camera, monkeypatch):
    agent = QueryAgent({"vlm": {"host": "http://x", "model": "m"}}, db, FakeEmbedder(), ["test-cam"])
    ids = {}
    for name, color, vec in [("a", "red", [0.6, 0.8, 0, 0]), ("b", "red", [1, 0, 0, 0]), ("c", "blue", [1, 0, 0, 0])]:
        t = db.create_track(camera, 1, datetime.now())
        db.add_attributes(t, [("upper_color", color, 1.0)], "vlm", "q", threshold=0.6)
        db.upsert_embedding(t, np.array(vec, np.float32), "m")
        ids[name] = t
    monkeypatch.setattr(agent, "_ask_llm", lambda q, now: {
        "conditions": [{"attribute": "upper_color", "value": "red"}], "appearance": "backpack strap"})
    res = agent.run("someone in red with a backpack strap")
    # Red matches first, best look first; the blue one only as a look-alike, marked as such.
    assert [p["id"] for p in res["people"]] == [ids["b"], ids["a"], ids["c"]]
    assert [p.get("labels_differ", False) for p in res["people"]] == [False, False, True]
    assert res["total"] == 2 and "upper color = red" in res["explanation"]


def test_run_falls_back_to_appearance_when_the_model_fails(db, camera, monkeypatch):
    agent = QueryAgent({"vlm": {"host": "http://x", "model": "m"}}, db, None, [])

    def broken(q, now):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(agent, "_ask_llm", broken)
    assert agent.plan("man in a beige coat")["appearance"] == "man in a beige coat"


def test_look_alikes_are_added_when_labels_disagree(db, camera, monkeypatch):
    agent = QueryAgent({"vlm": {"host": "http://x", "model": "m"}}, db, FakeEmbedder(), ["test-cam"])
    right = db.create_track(camera, 1, datetime.now())
    db.add_attributes(right, [("upper_color", "beige", 1.0)], "vlm", "q", threshold=0.6)
    db.upsert_embedding(right, np.array([0.8, 0.6, 0, 0], np.float32), "m")
    mislabeled = db.create_track(camera, 1, datetime.now())                 # same coat, wrong label
    db.add_attributes(mislabeled, [("upper_color", "orange", 1.0)], "vlm", "q", threshold=0.6)
    db.upsert_embedding(mislabeled, np.array([1, 0, 0, 0], np.float32), "m")
    monkeypatch.setattr(agent, "_ask_llm", lambda q, now: {
        "conditions": [{"attribute": "upper_color", "value": "beige"}], "appearance": "fur coat"})
    res = agent.run("beige fur coat")
    assert [p["id"] for p in res["people"]] == [right, mislabeled]
    assert res["people"][1]["labels_differ"] is True and "labeled differently" in res["explanation"]
