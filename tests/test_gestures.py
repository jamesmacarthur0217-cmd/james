"""The help sign (Phase 32): labeler/gestures.py on made-up body points."""
import numpy as np

from labeler.gestures import GestureWatch, hands_up

# A person standing, facing the camera (image y grows downwards): x, y, confidence for the 17 COCO points.
STAND = {0: (100, 50), 1: (95, 45), 2: (105, 45), 3: (90, 48), 4: (110, 48), 5: (80, 80), 6: (120, 80),
         7: (75, 120), 8: (125, 120), 9: (75, 160), 10: (125, 160), 11: (85, 170), 12: (115, 170),
         13: (85, 230), 14: (115, 230), 15: (85, 290), 16: (115, 290)}


def person(**moved):
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in STAND.items():
        k[i] = (x, y, 0.9)
    for name, (x, y) in moved.items():
        k[{"lw": 9, "rw": 10, "le": 7, "re": 8}[name]] = (x, y, 0.9)
    return k


LEFT_UP = person(le=(70, 60), lw=(70, 20))            # left elbow above the shoulder, wrist above the head
BOTH_UP = person(le=(70, 60), lw=(70, 20), re=(130, 60), rw=(130, 20))


def test_one_frame():
    assert hands_up(person()) == []
    assert hands_up(LEFT_UP) == ["left"] and hands_up(BOTH_UP) == ["left", "right"]
    assert hands_up(person(lw=(80, 40), le=(70, 70))) == []                  # at the head (scratching it): not up
    assert hands_up(person(lw=(70, 20), le=(70, 100))) == []                 # elbow below the shoulder: not up
    hidden_head = LEFT_UP.copy()
    hidden_head[[0, 1, 2, 3, 4], 2] = 0.1
    assert hands_up(hidden_head) == [] and hands_up(None) == []              # no head found: not checked
    far = LEFT_UP.copy()
    far[:, :2] = far[:, :2] / 4                                               # the same person far away
    assert hands_up(far) == ["left"]


def test_held_three_seconds_once_per_sign():
    w = GestureWatch({"hold_s": 3.0})
    t = 0.0
    events = []
    for _ in range(40):                                                       # 4 s up, at 10 frames a second
        events.append(w.update(1, t, LEFT_UP))
        t += 0.1
    fired = [e for e in events if e]
    assert len(fired) == 1 and fired[0] == {"since": 0.0, "hand": "left"}
    assert [i for i, e in enumerate(events) if e] == [30]                     # at 3.0 s
    for _ in range(10):                                                       # down 1 s, up again: the same sign
        w.update(1, t, person())
        t += 0.1
    assert not any(w.update(1, t + i * 0.1, LEFT_UP) for i in range(40))
    t += 4
    for _ in range(25):                                                       # down 2.5 s: a new sign may come
        w.update(1, t, person())
        t += 0.1
    assert any(w.update(1, t + i * 0.1, LEFT_UP) for i in range(31))


def test_short_raises_and_gaps():
    w = GestureWatch({"hold_s": 3.0, "gap_s": 0.5})
    t = 0.0
    for _ in range(25):                                                       # 2.5 s up, then down: nothing
        assert w.update(1, t, LEFT_UP) is None
        t += 0.1
    for _ in range(10):
        w.update(1, t, person())
        t += 0.1
    out = []
    for i in range(35):                                                       # a frame missed now and then
        out.append(w.update(1, t, None if i in (10, 20) else LEFT_UP))
        t += 0.1
    assert any(out)
    assert GestureWatch({"both_ok": False}).update(2, 0, BOTH_UP) is None
    w2 = GestureWatch({"both_ok": False})
    assert not any(w2.update(2, i * 0.1, BOTH_UP) for i in range(40))        # stretching both arms: not a sign
