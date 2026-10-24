import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reid import ReID, similarity, _hist

GROUPS = {'any': [0, 1, 3, 4, 5, 8, 9], 'person': [0, 1], 'vehicle': [3, 4, 5, 8, 9]}
RED, BLUE, GREEN = (0, 0, 220), (220, 0, 0), (0, 200, 0)


def frame_with(*boxes):
    f = np.full((480, 640, 3), 90, np.uint8)
    for (x1, y1, x2, y2), col in boxes:
        f[y1:y2, x1:x2] = col
    return f


def trk(box, tid, cls=3):
    return [box[0], box[1], box[2], box[3], tid, 0.9, cls]


def locked_memory(r, n=10, box=(300, 200, 360, 240), colour=RED):
    for i in range(n):
        t = i * 0.033
        r.observe_frame([trk(box, 1)], t)
        r.remember(frame_with((box, colour)), trk(box, 1), t)
    return n * 0.033


def run_search(r, frames, t0, step=0.033):
    out = None
    for k, (f, tracks) in enumerate(frames):
        t = t0 + (k + 1) * step
        r.observe_frame(tracks, t)
        out = r.propose(f, tracks, t)
        if out is not None:
            return out
    return None


def test_relocks_to_same_looking_new_track_nearby():
    r = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    t0 = locked_memory(r)
    box = (320, 205, 370, 240)                                   # smaller, slightly displaced, new id
    got = run_search(r, [(frame_with((box, RED)), [trk(box, 7)])] * 5, t0)
    assert got is not None and int(got[4]) == 7 and "id 1->7" in r.last_event


def test_does_not_relock_to_a_different_colour():
    r = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    t0 = locked_memory(r)
    box = (320, 205, 370, 240)
    assert run_search(r, [(frame_with((box, BLUE)), [trk(box, 7)])] * 10, t0) is None


def test_ignores_tracks_that_existed_while_target_was_visible():
    r = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    box = (300, 200, 360, 240)
    other = (330, 210, 380, 245)
    for i in range(10):
        t = i * 0.033
        r.observe_frame([trk(box, 1), trk(other, 5)], t)
        r.remember(frame_with((box, RED), (other, RED)), trk(box, 1), t)
    got = run_search(r, [(frame_with((other, RED)), [trk(other, 5)])] * 10, 0.33)
    assert got is None                                            # id 5 is a known different object


def test_wrong_class_group_is_rejected():
    r = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    t0 = locked_memory(r)
    box = (320, 205, 370, 240)
    assert run_search(r, [(frame_with((box, RED)), [trk(box, 7, cls=0)])] * 10, t0) is None   # person vs car
    r2 = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    t0 = locked_memory(r2)
    assert run_search(r2, [(frame_with((box, RED)), [trk(box, 8, cls=5)])] * 5, t0) is not None   # truck ~ car


def test_far_away_candidate_is_outside_gate():
    r = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    t0 = locked_memory(r)
    box = (20, 400, 70, 440)
    assert run_search(r, [(frame_with((box, RED)), [trk(box, 7)])] * 10, t0) is None


def test_ambiguity_waits_instead_of_guessing():
    r = ReID(frame_w=640, frame_h=480, class_groups=GROUPS)
    t0 = locked_memory(r)
    a, b = (322, 205, 372, 240), (324, 206, 374, 241)
    tracks = [trk(a, 7), trk(b, 8)]
    assert run_search(r, [(frame_with((a, RED), (b, RED)), tracks)] * 10, t0) is None


def test_needs_consecutive_confirmation():
    r = ReID({'confirm_frames': 4}, 640, 480, GROUPS)
    t0 = locked_memory(r)
    box = (320, 205, 370, 240)
    f = frame_with((box, RED))
    for k in range(3):
        t = t0 + (k + 1) * 0.033
        r.observe_frame([trk(box, 7)], t)
        assert r.propose(f, [trk(box, 7)], t) is None
    t = t0 + 4 * 0.033
    r.observe_frame([trk(box, 7)], t)
    assert r.propose(f, [trk(box, 7)], t) is not None


def test_times_out():
    r = ReID({'timeout_s': 2.0}, 640, 480, GROUPS)
    t0 = locked_memory(r)
    box = (320, 205, 370, 240)
    r.observe_frame([trk(box, 7)], t0 + 3.0)
    assert r.propose(frame_with((box, RED)), [trk(box, 7)], t0 + 3.0) is None


def test_disabled_and_empty_memory_never_propose():
    box = (320, 205, 370, 240)
    assert ReID({'enabled': False}, 640, 480, GROUPS).propose(frame_with((box, RED)), [trk(box, 7)], 1.0) is None
    assert ReID(None, 640, 480, GROUPS).propose(frame_with((box, RED)), [trk(box, 7)], 1.0) is None


def test_similarity_is_one_for_same_and_low_for_different():
    f = frame_with(((300, 200, 360, 240), RED), ((100, 100, 160, 140), BLUE))
    a, b, c = _hist(f, (300, 200, 360, 240)), _hist(f, (302, 202, 358, 238)), _hist(f, (100, 100, 160, 140))
    assert similarity(a, b) > 0.95 and similarity(a, c) < 0.3
    assert similarity(None, a) == 0.5
