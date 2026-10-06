"""Vocabulary, colors, geometry and config: the pure functions everything else relies on."""
import os

import numpy as np
import pytest

from labeler import colors, geometry, schema
from labeler.config import _expand


def test_clean_maps_off_vocabulary_answers_to_unknown():
    out = schema.clean({"upper_color": "crimson", "apparent_gender": "MAN", "age_range": "20-29"}, "kiosk")
    assert out["upper_color"] == "unknown"
    assert out["apparent_gender"] == "man"
    assert out["age_range"] == "20-29"
    assert out["eyewear"] == "unknown"          # missing field


def test_clean_turns_yes_no_items_into_a_sorted_list():
    raw = {"carrying": {"backpack": "yes", "phone": "yes", "shopping_bag": "no"}}
    assert schema.clean(raw, "kiosk")["carrying"] == "backpack,phone"
    assert schema.clean({"carrying": {}}, "kiosk")["carrying"] == "none"


def test_json_schema_asks_yes_no_per_carried_item():
    js = schema.json_schema("security")
    assert set(js["required"]) == set(schema.MODE_FIELDS["security"])
    assert js["properties"]["carrying"]["properties"]["shopping_bag"]["enum"] == ["yes", "no"]


@pytest.mark.parametrize("bgr,name", [
    ((0, 0, 200), "red"), ((200, 0, 0), "blue"), ((10, 10, 10), "black"),
    ((240, 240, 240), "white"), ((0, 200, 0), "green"), ((128, 128, 128), "gray"),
])
def test_color_names_for_solid_patches(bgr, name):
    patch = np.full((40, 40, 3), bgr, np.uint8)
    assert colors.dominant(colors.color_histogram(patch), min_pixels=10)[0] == name


def test_grayscale_frame_is_low_color():
    gray = np.random.default_rng(0).integers(0, 255, (72, 128), dtype=np.uint8)
    assert colors.is_low_color(np.dstack([gray] * 3))
    assert not colors.is_low_color(np.full((72, 128, 3), (0, 0, 200), np.uint8))


def test_box_iou():
    assert geometry.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert geometry.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert geometry.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def _pose(**points):
    kpts = np.zeros((17, 3), np.float32)
    for name, (x, y) in points.items():
        kpts[getattr(geometry, name.upper())] = (x, y, 0.9)
    return kpts


def test_torso_box_sits_between_shoulders_and_hips():
    kpts = _pose(l_shoulder=(60, 100), r_shoulder=(140, 100), l_hip=(70, 250), r_hip=(130, 250))
    x1, y1, x2, y2 = geometry.torso_box(kpts, (40, 50, 160, 400))
    assert 60 < x1 < x2 < 140 and 100 < y1 < y2 < 250


def test_face_box_needs_nose_and_eyes():
    assert geometry.face_box(_pose(nose=(100, 100))) is None
    box = geometry.face_box(_pose(nose=(100, 100), l_eye=(110, 90), r_eye=(90, 90)))
    assert box is not None and box[0] < 100 < box[2]


def test_config_expands_environment_variables(monkeypatch):
    monkeypatch.setenv("VL_TEST_URL", "postgresql://x")
    assert _expand({"database": {"url": "${VL_TEST_URL}"}}) == {"database": {"url": "postgresql://x"}}
    monkeypatch.delenv("VL_TEST_URL")
    with pytest.raises(KeyError):
        _expand("${VL_TEST_URL}")
