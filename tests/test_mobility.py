"""Mobility aids (wheelchair, walking stick, crutches, walker): a label that a setting can turn off."""
import copy
from datetime import datetime

from fastapi.testclient import TestClient

from labeler import schema, settings
from labeler.agent import clean_plan
from labeler.alerts import AlertEngine, matches
from labeler.config import DEFAULTS
from labeler.pipeline import TrackState
from labeler.vlm import PROMPT_MOBILITY, VLMClient
from labeler.web import create_app

from test_pipeline_logic import make_pipeline

OFF = {"mobility_aids": False}


def test_on_by_default_in_both_modes():
    assert DEFAULTS["labels"]["mobility_aids"] is True
    for mode in ("kiosk", "security"):
        assert "mobility_aid" in schema.mode_fields(mode, DEFAULTS["labels"])
        assert "mobility_aid" in schema.mode_fields(mode)          # no labels section = defaults
        assert "mobility_aid" not in schema.mode_fields(mode, OFF)
        assert "carrying" in schema.mode_fields(mode, OFF)         # only optional fields can be turned off


def test_asked_as_one_choice_with_look_alikes_that_count_as_none():
    choices = schema.json_schema("kiosk")["properties"]["mobility_aid"]["enum"]
    assert {"wheelchair", "walking stick", "crutches", "walker", "ski or hiking poles", "none"} <= set(choices)
    assert schema.clean({"mobility_aid": "Walking stick"}, "kiosk")["mobility_aid"] == "walking stick"
    assert schema.clean({"mobility_aid": "ski or hiking poles"}, "kiosk")["mobility_aid"] == "none"
    assert schema.clean({"mobility_aid": "a broom"}, "kiosk")["mobility_aid"] == "none"
    assert schema.clean({"mobility_aid": "unknown"}, "kiosk")["mobility_aid"] == "unknown"
    assert schema.clean({}, "security")["mobility_aid"] == "unknown"
    assert schema.clean({"carrying": {"phone": "yes"}}, "kiosk")["carrying"] == "phone"   # other lists as before


def test_turned_off_it_is_neither_asked_nor_stored():
    assert "mobility_aid" not in schema.json_schema("kiosk", OFF)["properties"]
    assert "mobility_aid" not in schema.clean({"mobility_aid": "wheelchair"}, "kiosk", OFF)


def test_prompt_explains_aids_only_when_asked():
    vcfg = DEFAULTS["vlm"]
    for mode in ("kiosk", "security"):
        assert PROMPT_MOBILITY in VLMClient(vcfg, mode).prompt(mode, False)
        assert PROMPT_MOBILITY not in VLMClient(vcfg, mode, OFF).prompt(mode, False)
    for choice in schema.DECOYS["mobility_aid"]:   # the prompt explains every look-alike answer
        assert choice in " ".join(PROMPT_MOBILITY.split())


def test_pipeline_stores_the_vote_and_nothing_when_off(db, camera):
    base = {f: "unknown" for f in schema.MODE_FIELDS["kiosk"]}
    p = make_pipeline(db)
    st = TrackState(1, datetime.now(), 0.0, db_id=db.create_track(camera, 1, datetime.now()))
    p.by_db_id[st.db_id] = st
    p.on_vlm_result(st.db_id, [dict(base, mobility_aid="wheelchair"), dict(base, mobility_aid="wheelchair")])
    assert db.resolved_values(st.db_id)["mobility_aid"] == "wheelchair"

    p = make_pipeline(db)
    p.cfg["labels"]["mobility_aids"] = False
    st = TrackState(2, datetime.now(), 0.0, db_id=db.create_track(camera, 2, datetime.now()))
    p.by_db_id[st.db_id] = st
    p.on_vlm_result(st.db_id, [dict(base, mobility_aid="wheelchair")])
    values = db.resolved_values(st.db_id)
    assert "mobility_aid" not in values and "upper_color" in values


def test_ask_box_words():
    def conds(question, value):
        raw = {"conditions": [{"attribute": "mobility_aid", "value": value}]}
        return clean_plan(raw, ["cam1"], datetime(2026, 10, 13, 12), question)["conditions"]
    assert conds("old man with a cane", "walking stick") == [["mobility_aid", "walking stick"]]
    assert conds("someone on crutches", "crutches") == [["mobility_aid", "crutches"]]
    assert conds("a lady with a rollator", "walker") == [["mobility_aid", "walker"]]
    assert conds("people in wheelchairs", "wheelchair") == [["mobility_aid", "wheelchair"]]
    assert conds("a man in a red shirt", "wheelchair") == []        # not asked for: not added


def test_alert_rules_can_use_it():
    assert matches([["mobility_aid", "walker"]], {"mobility_aid": "crutches,walker"})
    assert not matches([["mobility_aid", "walker"]], {"mobility_aid": "none"})


def _schema(db, tmp_path, labels):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["labels"] = labels
    client = TestClient(create_app(cfg, db, None, None, AlertEngine(db)), client=("127.0.0.1", 5000))
    return client.get("/api/schema").json()


def test_pages_offer_it_only_when_on(db, tmp_path):
    on = _schema(db, tmp_path, {"mobility_aids": True})
    assert "mobility_aid" in on["fields"] and "mobility_aid" in on["all_fields"]
    off = _schema(db, tmp_path, OFF)
    assert "mobility_aid" not in off["fields"] and "mobility_aid" not in off["all_fields"]
    assert "carrying" in off["all_fields"]


def test_settings_switch_maps_to_the_labels_section():
    assert settings.SIMPLE["mobility_aids"] == ("labels", "mobility_aids")
    base = {**copy.deepcopy(DEFAULTS), "cameras": [{"name": "cam1", "source": 0}], "mode": "kiosk"}
    cfg = settings.apply(base, {"labels": {"mobility_aids": False}})
    assert cfg["labels"]["mobility_aids"] is False
    assert settings.current(cfg)["mobility_aids"] is False
