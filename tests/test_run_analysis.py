"""Run logger + analyzer: synthetic runs that must PASS and must FAIL."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze_run as ar
from run_logger import RunLogger, FIELDS


def mk(t, **kw):
    r = {k: "" for k in FIELDS}
    r.update(t=t, airborne=1, mode="4", fps=30, locked_id=1, target_visible=1,
             err_x=0.0, err_y=0.0, area=0.01, setpoint=0.01,
             cmd_vx=0.0, cmd_vz=0.0, cmd_yaw=0.0,
             safety_action="NONE", safety_reason="", errors=0)
    r.update(kw)
    return {k: str(v) for k, v in r.items()}


def good_follow(seconds=90, dt=1 / 30):
    rows, t, i = [], 0.0, 0
    while t < seconds:
        # small wobble, well inside the bands
        rows.append(mk(round(t, 3), err_x=0.02 * (1 if i % 90 < 45 else -1),
                       area=0.0105, cmd_yaw=0.0))
        t += dt; i += 1
    return rows


def status(results, name):
    return {n: s for n, s, _ in results}[name]


def test_good_run_follows():
    res = ar.run_all(good_follow(), follow_seconds=60)
    assert status(res, "log integrity") == ar.PASS
    assert status(res, "follow target") == ar.PASS
    assert not any(s == ar.FAIL for _, s, _ in res)


def test_target_never_centred_fails():
    rows = [mk(i / 30, err_x=0.4) for i in range(30 * 90)]
    assert status(ar.run_all(rows, 60), "follow target") == ar.FAIL


def test_wrong_distance_fails():
    rows = [mk(i / 30, area=0.03) for i in range(30 * 90)]          # 3x setpoint
    assert status(ar.run_all(rows, 60), "follow target") == ar.FAIL


def test_oscillating_yaw_fails():
    rows = [mk(i / 30, cmd_yaw=0.4 * (1 if i % 4 < 2 else -1)) for i in range(30 * 90)]
    assert status(ar.run_all(rows, 60), "follow target") == ar.FAIL


def test_short_run_is_inconclusive_not_pass():
    res = ar.run_all(good_follow(20), follow_seconds=60)
    assert status(res, "follow target") == ar.INCONCLUSIVE


def test_creep_at_rest_detected():
    rows = [mk(i / 30, cmd_vx=0.5) for i in range(30 * 20)]
    assert status(ar.run_all(rows, 5), "no creep at rest") == ar.FAIL


def test_lock_loss_ok_when_commands_decay_and_lock_released():
    rows = good_follow(20)
    t0 = float(rows[-1]["t"])
    for i in range(1, 30 * 6):
        t = t0 + i / 30
        rows.append(mk(round(t, 3), target_visible=0,
                       locked_id=1 if i < 30 * 3 else "",
                       cmd_vx=0.0 if i > 15 else 0.5))
    assert status(ar.run_all(rows, 5, max_lost_frames=60), "lock loss handling") == ar.PASS


def test_lock_loss_fails_if_commands_persist():
    rows = good_follow(20)
    t0 = float(rows[-1]["t"])
    for i in range(1, 30 * 6):
        rows.append(mk(round(t0 + i / 30, 3), target_visible=0, cmd_vx=1.0, locked_id=1))
    assert status(ar.run_all(rows, 5), "lock loss handling") == ar.FAIL


def test_lock_loss_fails_if_lock_never_released():
    rows = good_follow(20)
    t0 = float(rows[-1]["t"])
    for i in range(1, 30 * 10):
        rows.append(mk(round(t0 + i / 30, 3), target_visible=0, cmd_vx=0.0, locked_id=1))
    assert status(ar.run_all(rows, 5, max_lost_frames=60), "lock loss handling") == ar.FAIL


def test_lock_loss_inconclusive_if_never_lost():
    assert status(ar.run_all(good_follow(), 60), "lock loss handling") == ar.INCONCLUSIVE


def test_failsafe_must_suppress_commands():
    rows = good_follow(10)
    rows.append(mk(11.0, safety_action="HOVER", safety_reason="video_stale(1s)", cmd_vx=0.3))
    assert status(ar.run_all(rows, 5), "failsafes / override") == ar.FAIL


def test_failsafe_pass_when_suppressed():
    rows = good_follow(10)
    for i in range(10):
        rows.append(mk(11.0 + i / 30, safety_action="HOVER", safety_reason="video_stale(1s)"))
    assert status(ar.run_all(rows, 5), "failsafes / override") == ar.PASS


def test_failsafe_inconclusive_if_none_triggered():
    assert status(ar.run_all(good_follow(10), 5), "failsafes / override") == ar.INCONCLUSIVE


def test_soak():
    rows = good_follow(120)
    assert status(ar.run_all(rows, 5, soak_seconds=100), "soak") == ar.PASS
    for r in rows[len(rows) // 2:]:
        r["errors"] = str(int(float(r["t"]) * 2))     # errors keep climbing
    assert status(ar.run_all(rows, 5, soak_seconds=100), "soak") == ar.FAIL


def test_missing_column_fails_integrity():
    rows = [{"t": "0"}]
    assert ar.check_log_integrity(rows)[0] == ar.FAIL


def test_run_logger_roundtrip(tmp_path):
    lg = RunLogger(str(tmp_path), flush_every=1)
    lg.row(frame=1, locked_id=3, cmd_vx=0.5, bogus_field=1)
    lg.close()
    rows = list(csv.DictReader(open(lg.path)))
    assert rows[0]["locked_id"] == "3" and rows[0]["cmd_vx"] == "0.5"
    assert "bogus_field" not in rows[0]
    assert ar.check_log_integrity(rows)[0] == ar.PASS


def test_run_logger_disabled_writes_nothing(tmp_path):
    lg = RunLogger(str(tmp_path / "x"), enabled=False)
    lg.row(frame=1)
    lg.close()
    assert not (tmp_path / "x").exists()


def test_correcting_a_large_range_error_is_not_creep():
    # area 0.033 vs setpoint 0.02 is a 22% range error: commanding motion is correct there
    rows = [mk(i / 30, area=0.033, cmd_vx=-0.6) for i in range(30 * 20)]
    assert status(ar.run_all(rows, 5), "no creep at rest") != ar.FAIL
