"""
FC backend (fake MAVLink master) and the FC/safety thread (fake FC):
verified takeoff, abort, watchdog, stale-command refusal, pilot override,
RTL latch, altitude limiter, failed-takeoff recovery, landing detection.
"""
import os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import yaml

pytest.importorskip("pymavlink")
pytest.importorskip("cv2")
pytest.importorskip("flask")

import fc_interface5 as fci
import tracker
from safety import SafetySupervisor
from tracker_state import TrackerState

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config.yaml')))


# =====================================================================
# Fake MAVLink master + backend
# =====================================================================
class FakeMav:
    def __init__(self, owner): self.o = owner
    def command_long_send(self, ts, tc, cmd, conf, *p):
        self.o.sent.append(("cmd", cmd, p))
        self.o.on_command(cmd, p)
    def set_position_target_local_ned_send(self, *a):
        self.o.sent.append(("vel", a[8], a[9], a[10], a[15]))  # vx, vy, vz, yaw_rate


class FakeMaster:
    def __init__(self, backend, arm=True, climb=True):
        self.b, self.arm, self.climb = backend, arm, climb
        self.sent, self.mav = [], FakeMav(self)
        self.modes = []
    def set_mode(self, m):
        self.modes.append(m)
        self.b._telem["mode"] = str(m)
    def on_command(self, cmd, p):
        from pymavlink import mavutil
        if cmd == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and p[0] == 1 and self.arm:
            self.b._telem["armed"] = True
        if cmd == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF and self.climb:
            target = p[6]
            def ramp():
                for i in range(1, 11):
                    time.sleep(0.01)
                    self.b._telem["alt"] = target * i / 10
            threading.Thread(target=ramp, daemon=True).start()
    def recv_match(self, **kw):
        time.sleep(0.01)
        return None


class FakeArdu(fci.ArduPilotBackend):
    """Real flight-command logic, fake socket."""
    def __init__(self, arm=True, climb=True, **safety):
        sc = dict(CFG['safety'], takeoff_timeout_s=1.0, mode_timeout_s=0.3,
                  arm_timeout_s=0.3, **safety)
        super().__init__("fake", CFG['fc'], CFG['rc'], sc)
        from pymavlink import mavutil
        self._mavutil = mavutil
        self.target_system = self.target_component = 1
        self._tx_lock = threading.Lock()
        self._telem_lock = threading.Lock()
        self._telem = {"armed": False, "mode": "0", "alt": 0.0, "heading": 0,
                       "vx": 0, "vy": 0, "vz": 0, "battery": 0.0, "gps_sats": 0}
        self._last_rx = time.monotonic()
        self.master = FakeMaster(self, arm, climb)
        self.master.sent = self.master.sent
    @property
    def sent(self): return self.master.sent


def test_takeoff_success_verified():
    b = FakeArdu()
    assert b.arm_and_takeoff(10.0) is True
    assert b.get_telemetry()["alt"] >= 9.5
    assert b.takeoff_in_progress is False
    assert b.master.modes[0] == fci.ARDUCOPTER_MODE_GUIDED


def test_takeoff_fails_if_vehicle_never_arms():
    b = FakeArdu(arm=False)
    assert b.arm_and_takeoff(10.0) is False
    assert not any(s[0] == "cmd" and s[1] == 22 for s in b.sent), "must not send TAKEOFF unarmed"
    assert b.takeoff_in_progress is False


def test_takeoff_fails_if_altitude_not_reached():
    b = FakeArdu(climb=False)
    assert b.arm_and_takeoff(10.0) is False
    assert b.takeoff_in_progress is False


def test_takeoff_abort():
    b = FakeArdu(climb=False)
    res = []
    t = threading.Thread(target=lambda: res.append(b.arm_and_takeoff(10.0)))
    t.start()
    time.sleep(0.45)
    b.abort_takeoff.set()
    t.join(2)
    assert res == [False]


def test_rtl_and_land_modes_and_hover_first():
    b = FakeArdu()
    b.rtl()
    b.land()
    assert b.master.modes == [fci.ARDUCOPTER_MODE_RTL, fci.ARDUCOPTER_MODE_LAND]
    assert fci.ARDUCOPTER_MODE_RTL == 6 and fci.ARDUCOPTER_MODE_LAND == 9
    assert b.cmd_queue.get_nowait() == (0.0, 0.0, 0.0, 0.0)


def test_worker_zero_velocity_watchdog():
    b = FakeArdu()
    b._telem["armed"] = True
    b.watchdog_timeout = 0.15
    b.running = True
    b.is_connected = True
    b.last_cmd_time = time.monotonic()
    th = threading.Thread(target=b._backend_worker, daemon=True)
    th.start()
    b.send_velocity(1.0, 0.0, 0.5, 0.3)
    time.sleep(0.1)
    b.running = False
    th.join(1)
    vels = [s for s in b.sent if s[0] == "vel"]
    assert any(v[1] == 1.0 for v in vels), "command should have been streamed"

    b.sent.clear()
    b.running = True
    th = threading.Thread(target=b._backend_worker, daemon=True)
    th.start()
    time.sleep(0.5)                                   # nobody sends: watchdog must hold zero
    b.running = False
    th.join(1)
    vels = [s for s in b.sent if s[0] == "vel"]
    assert vels and all(v[1:] == (0.0, 0.0, 0.0, 0.0) for v in vels[-3:]), vels[-3:]


def test_worker_does_not_stream_when_disarmed():
    b = FakeArdu()
    b._telem["armed"] = False
    b.running = True
    b.is_connected = True
    b.last_cmd_time = time.monotonic()
    th = threading.Thread(target=b._backend_worker, daemon=True)
    th.start()
    b.send_velocity(1, 0, 0, 0)
    time.sleep(0.2)
    b.running = False
    th.join(1)
    assert not [s for s in b.sent if s[0] == "vel"]


def test_velocity_errors_are_counted_not_swallowed():
    b = FakeArdu()
    b.master.mav.set_position_target_local_ned_send = lambda *a: (_ for _ in ()).throw(OSError("boom"))
    b._safe_send((1, 0, 0, 0))
    assert b.errors.count('velocity_tx') == 1


def test_px4_is_explicitly_unsupported():
    fc = fci.FCInterface('px4', 'x')
    with pytest.raises(NotImplementedError):
        fc.backend._connect_impl()


def test_unknown_fc_rejected():
    with pytest.raises(ValueError):
        fci.FCInterface('betaflight', 'x')


# =====================================================================
# FC/safety thread with a fake FC
# =====================================================================
class FakeFC:
    def __init__(self, **telem):
        self.t = dict(armed=True, mode="4", alt=10.0, battery=12.0, gps_sats=10, heading=0)
        self.t.update(telem)
        self.age = 0.05
        self.calls, self.vel = [], []
        self.takeoff_ok = True
    def get_telemetry(self): return dict(self.t)
    def telemetry_age(self): return self.age
    def send_velocity(self, vx, vy, vz, yaw): self.vel.append((vx, vy, vz, yaw))
    def hover(self): self.calls.append("hover")
    def rtl(self): self.calls.append("rtl")
    def land(self): self.calls.append("land")
    def arm_and_takeoff(self, alt):
        self.calls.append(f"takeoff{alt:g}")
        return self.takeoff_ok


def run_thread(fc, state, seconds=0.5, no_fly=False, **safety):
    cfg = dict(CFG)
    cfg['safety'] = dict(CFG['safety'], **safety)
    sup = SafetySupervisor(cfg['safety'])
    sup.note_frame()
    th = threading.Thread(target=tracker.fc_thread_fn,
                          args=(fc, state, cfg, no_fly, sup), daemon=True)
    th.start()
    return th, sup


def stop(th, state):
    state.flag_quit.set()
    th.join(2)
    assert not th.is_alive()


def fresh_airborne():
    st = TrackerState()
    st.is_airborne = True
    return st


def test_nominal_velocity_is_forwarded():
    fc, st = FakeFC(), fresh_airborne()
    th, sup = run_thread(fc, st)
    t_end = time.time() + 0.4
    while time.time() < t_end:
        st.cmd.update(1.0, 0.0, 0.2)
        sup.note_frame()
        time.sleep(0.02)
    stop(th, st)
    assert fc.vel and fc.vel[-1] == (1.0, 0.0, 0.0, 0.2)
    assert st.safety.action == "NONE"


def test_stale_vision_loop_never_leaves_old_velocity_flying():
    """The vision loop writes once then 'hangs'; the FC thread must not keep
    replaying that command."""
    fc, st = FakeFC(), fresh_airborne()
    th, sup = run_thread(fc, st, video_loss_hover_s=60, video_loss_rtl_s=120)
    st.cmd.update(2.0, 0.0, 0.5)
    time.sleep(0.9)                                      # > watchdog_timeout (0.5 s)
    stop(th, st)
    assert any(v[0] == 2.0 for v in fc.vel), "fresh command should have gone out"
    assert fc.vel[-1] == (0.0, 0.0, 0.0, 0.0), "stale command must be replaced by zero"


def test_pilot_override_stops_commands_and_flags_release():
    fc, st = FakeFC(mode="5"), fresh_airborne()          # LOITER
    th, sup = run_thread(fc, st)
    for _ in range(10):
        st.cmd.update(1.0, 0.0, 0.0)
        sup.note_frame()
        time.sleep(0.03)
    for _ in range(100):                     # the FC thread zeroes the stale command on its next tick
        if st.cmd.read() == (0.0, 0.0, 0.0):
            break
        time.sleep(0.02)
    stop(th, st)
    # Safety property: no MOTION while the pilot has control. (A zero-velocity hover from
    # a coincident video-loss rule under heavy CPU load is harmless, so it is not counted.)
    assert all(v == (0.0, 0.0, 0.0, 0.0) for v in fc.vel), \
        f"must not command velocity while pilot has control: {fc.vel[:3]}"
    assert st.safety.action == "RELEASE_CONTROL"
    assert st.cmd.read() == (0.0, 0.0, 0.0)
    assert "rtl" not in fc.calls


def test_video_loss_triggers_rtl_exactly_once():
    fc, st = FakeFC(), fresh_airborne()
    th, sup = run_thread(fc, st, video_loss_hover_s=0.1, video_loss_rtl_s=0.3)
    time.sleep(0.9)                                      # no note_frame() -> video lost
    stop(th, st)
    assert fc.calls.count("rtl") == 1, fc.calls
    assert fc.vel == [] or all(v == (0.0, 0.0, 0.0, 0.0) for v in fc.vel)


def test_link_loss_releases_control():
    fc, st = FakeFC(), fresh_airborne()
    fc.age = 9.0
    th, sup = run_thread(fc, st)
    for _ in range(8):
        sup.note_frame(); time.sleep(0.04)
    stop(th, st)
    assert st.safety.action == "RELEASE_CONTROL" and st.safety.reason.startswith("link_lost")
    assert fc.vel == []


def test_altitude_floor_enforced_on_outgoing_velocity():
    fc, st = FakeFC(alt=CFG['flight']['min_altitude']), fresh_airborne()
    th, sup = run_thread(fc, st)
    t_end = time.time() + 0.4
    while time.time() < t_end:
        st.cmd.update(0.5, 1.5, 0.0)                     # vision loop asks to descend hard
        sup.note_frame()
        time.sleep(0.02)
    stop(th, st)
    assert fc.vel, "should still be sending"
    assert all(v[2] == 0.0 for v in fc.vel), "descent below the floor must be blocked"
    assert any(v[0] == 0.5 for v in fc.vel), "forward motion must still pass through"


def test_failed_takeoff_while_armed_lands_and_stays_grounded():
    fc, st = FakeFC(alt=0.2), TrackerState()
    fc.takeoff_ok = False
    th, sup = run_thread(fc, st)
    st.flag_takeoff.set()
    time.sleep(0.4)
    stop(th, st)
    assert "land" in fc.calls
    assert st.is_airborne is False


def test_successful_takeoff_sets_airborne():
    fc, st = FakeFC(alt=0.2), TrackerState()
    th, sup = run_thread(fc, st)
    st.flag_takeoff.set()
    time.sleep(0.3)
    stop(th, st)
    assert fc.calls[0] == "takeoff10" and st.is_airborne


def test_operator_rtl_and_land_keys():
    fc, st = FakeFC(), fresh_airborne()
    th, sup = run_thread(fc, st)
    st.flag_rtl.set(); time.sleep(0.2)
    st.flag_land.set(); time.sleep(0.2)
    stop(th, st)
    assert "rtl" in fc.calls and "land" in fc.calls


def test_landed_detection_clears_airborne():
    fc, st = FakeFC(armed=False, alt=0.1), fresh_airborne()
    th, sup = run_thread(fc, st)
    time.sleep(2.6)
    stop(th, st)
    assert st.is_airborne is False


def test_no_fly_sends_nothing_and_never_acts():
    fc, st = FakeFC(mode="0"), TrackerState()
    th, sup = run_thread(fc, st, no_fly=True)
    st.flag_takeoff.set()
    st.cmd.update(1, 0, 0)
    time.sleep(0.3)
    stop(th, st)
    assert fc.vel == [] and fc.calls == []
    assert st.is_airborne is True                          # simulated


def test_no_fly_without_fc_object():
    st = TrackerState()
    th, sup = run_thread(None, st, no_fly=True)
    st.flag_takeoff.set()
    time.sleep(0.2)
    stop(th, st)
    assert st.is_airborne


def test_speed_caps_applied_before_sending():
    fc, st = FakeFC(alt=20.0), fresh_airborne()
    th, sup = run_thread(fc, st)
    t_end = time.time() + 0.4
    while time.time() < t_end:
        st.cmd.update(9.0, 0.0, 5.0)                       # PID bug / absurd request
        sup.note_frame()
        time.sleep(0.02)
    stop(th, st)
    sent = [v for v in fc.vel if v[0] != 0.0]
    assert sent, "should still be sending"
    assert all(abs(v[0]) <= CFG['flight']['max_forward_speed'] + 1e-9 for v in sent)
    assert all(abs(v[3]) <= CFG['flight']['max_yaw_rate'] + 1e-9 for v in sent)
