"""MAVLink readiness checks used by the demo launcher (simulation wait + hardware checklist)."""
import time

EKF_OK = 1 | 2 | 4 | 16 | 32          # attitude, horizontal+vertical velocity, absolute horizontal+vertical position


def _conn(connect):
    from pymavlink import mavutil
    return mavutil.mavlink_connection(connect)


def wait_ekf_ready(connect: str, timeout: float = 180.0, log=print) -> bool:
    """Block until the autopilot reports a 3D fix and a usable EKF position (safe to arm). Never skips arming checks."""
    m = _conn(connect)
    if m.wait_heartbeat(timeout=30) is None:
        log(f"no heartbeat on {connect}")
        return False
    t0, ok = time.time(), False
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2.0)
        if msg is not None and (msg.flags & EKF_OK) == EKF_OK:
            ok = True
            break
    try:
        m.close()
    except Exception:
        pass
    return ok


def hardware_checklist(connect: str, min_sats: int = 8, min_battery_pct: int = 60, log=print):
    """[(status, text)] with status OK / WARN / FAIL. Any FAIL means: do not fly the demo."""
    out = []
    m = _conn(connect)
    if m.wait_heartbeat(timeout=15) is None:
        return [("FAIL", f"No heartbeat on {connect}: flight controller link is down.")]
    out.append(("OK", "Flight controller link is up."))
    seen = {}
    t0 = time.time()
    while time.time() - t0 < 8.0 and not {"GPS_RAW_INT", "SYS_STATUS", "EKF_STATUS_REPORT", "HEARTBEAT"} <= set(seen):
        msg = m.recv_match(blocking=True, timeout=1.0)
        if msg is not None:
            seen[msg.get_type()] = msg
    gps = seen.get("GPS_RAW_INT")
    if gps is None:
        out.append(("FAIL", "No GPS data."))
    elif gps.fix_type < 3 or gps.satellites_visible < min_sats:
        out.append(("FAIL", f"GPS not good enough: fix {gps.fix_type}, {gps.satellites_visible} satellites (need 3D fix, {min_sats}+)."))
    else:
        out.append(("OK", f"GPS 3D fix, {gps.satellites_visible} satellites."))
    ekf = seen.get("EKF_STATUS_REPORT")
    out.append(("OK", "EKF position estimate healthy.") if ekf is not None and (ekf.flags & EKF_OK) == EKF_OK
               else ("FAIL", "EKF is not ready (no absolute position)."))
    ss = seen.get("SYS_STATUS")
    if ss is None:
        out.append(("WARN", "No battery data."))
    elif 0 <= ss.battery_remaining < min_battery_pct:
        out.append(("FAIL", f"Battery {ss.battery_remaining}% (need {min_battery_pct}%+ for a demo)."))
    else:
        out.append(("OK", f"Battery {ss.battery_remaining}% ({ss.voltage_battery / 1000.0:.1f} V)."))
    hb = seen.get("HEARTBEAT")
    if hb is not None and hb.base_mode & 128:
        out.append(("WARN", "Vehicle is already armed."))
    try:
        m.close()
    except Exception:
        pass
    return out
