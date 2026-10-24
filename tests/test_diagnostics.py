import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_sim_limits as cs
import pursuit_report as pr
from run_logger import FIELDS


def test_limits_default_params_flagged_as_blocking_a_fast_chase():
    r = cs.assess({"WPNAV_SPEED": 500, "WPNAV_ACCEL": 250, "ANGLE_MAX": 3000})
    assert r[0][0] == "FAIL" and "5 m/s" in r[0][1]


def test_limits_fast_params_ok_and_high_accel_warned():
    r = dict((t.split()[0], l) for l, t in cs.assess({"WPNAV_SPEED": 2500, "WPNAV_ACCEL": 600, "ANGLE_MAX": 6000}))
    assert set(r.values()) == {"OK"}
    assert cs.assess({"WPNAV_SPEED": 2500, "WPNAV_ACCEL": 900})[1][0] == "WARN"
    assert cs.assess({})[0][0] == "WARN"


def row(t, locked=True, vis=True, cmd=0.0, gs=0.0):
    r = {k: "" for k in FIELDS}
    r.update(t=t, airborne=1, locked_id=1 if locked else "", target_visible=int(vis), cmd_vx=cmd, gs=gs, v_fwd=gs, fps=30)
    return r


def test_dropout_bursts_measured():
    rows = [row(i * 0.1, vis=not (10 <= i < 14 or 30 <= i < 31)) for i in range(50)]
    b = pr.dropout_bursts(rows)
    assert len(b) == 2 and abs(max(b) - 0.3) < 1e-6


def test_rocking_counted_only_for_big_swings():
    steady = [10.0 + 0.5 * (i % 2) for i in range(100)]
    rocking = [10.0 + 6.0 * (1 if (i // 10) % 2 else 0) for i in range(100)]
    assert pr.reversals(steady) == 0 and pr.reversals(rocking) >= 4


def test_summary_flags_drone_not_reaching_command():
    rows = [row(i * 0.1, cmd=15.0, gs=5.0) for i in range(100)]
    text = "\n".join(pr.summarise(rows))
    assert "not reaching the commanded speed" in text
    ok = "\n".join(pr.summarise([row(i * 0.1, cmd=15.0, gs=14.5) for i in range(100)]))
    assert "not reaching" not in ok
