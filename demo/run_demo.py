#!/usr/bin/env python3
"""One-command exhibition: starts the stack, flies a scripted 4-5 minute show, records it, writes a results page.

    python3 demo/run_demo.py --dry-run                    # print the plan, check the files, start nothing
    python3 demo/run_demo.py                              # simulation show, automatic target lock
    python3 demo/run_demo.py --lock manual                # you click each target (a click always works too)
    python3 demo/run_demo.py --use-running-sim            # sim/launch_sim.sh already running in another terminal
    python3 demo/run_demo.py --mode hardware --connect /dev/ttyACM0 ...     # see docs/DEMO.md before using

Nothing existing changes: this only launches the same commands you would type, plus two opt-in tracker flags.
Ctrl+C anywhere = the drone is told to return home, then everything started here is stopped.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "demo"))

import yaml                                                    # noqa: E402
import show_paths as sp                                        # noqa: E402
import report as rp                                            # noqa: E402
from profiles import apply_profile                             # noqa: E402

LIVE_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Show control</title><style>
:root{--bg:#fff;--fg:#14171c;--mut:#5a6270;--card:#f4f6f9;--ok:#0a7d3c;--bad:#b3261e;--acc:#1d5fd1}
@media(prefers-color-scheme:dark){:root{--bg:#0f1318;--fg:#e8ebf0;--mut:#9aa3b0;--card:#171c24;--ok:#4cc27d;--bad:#ff7b72;--acc:#6ea0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:1.3rem;margin:0 0 4px}.mut{color:var(--mut)}.big{font-size:2rem;font-weight:700}
.g{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:14px 0}.c{background:var(--card);border-radius:10px;padding:10px 12px}
.c b{display:block;font-size:1.5rem}.bar{height:8px;background:var(--card);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.ok{color:var(--ok)}.bad{color:var(--bad)}</style></head><body>
<h1 id=t>Show control</h1><div class=mut id=ph></div><div class=big id=seg>…</div><div class=mut id=note></div>
<div class=bar><i id=bar style="width:0"></i></div><div class=g id=g></div><div class=mut id=msg></div>
<script>
async function tick(){try{const s=await (await fetch('status.json',{cache:'no-store'})).json();
t.textContent=s.title;ph.textContent=s.phase+(s.index?`  ·  segment ${s.index}/${s.count}`:'');seg.textContent=s.segment||'';
note.textContent=s.note||'';bar.style.width=(100*s.progress)+'%';msg.textContent=s.message||'';
const k=(a,b,c)=>`<div class=c><span class=mut>${a}</span><b class="${c||''}">${b}</b></div>`;
g.innerHTML=k('Lock',s.locked?(s.visible?'LOCKED':'searching…'):'no lock',s.locked&&s.visible?'ok':'bad')+k('Drone speed',s.gs.toFixed(1)+' m/s')
+k('Target speed',s.target_speed.toFixed(1)+' m/s')+k('Altitude',s.alt.toFixed(1)+' m')+k('Frame rate',s.fps.toFixed(0)+' fps')
+k('Re-ID',s.reid||'–')+k('Time',s.elapsed.toFixed(0)+' s')+k('Battery',s.battery.toFixed(1)+' V');}catch(e){}}
setInterval(tick,500);tick();</script></body></html>"""


def say(msg):
    print(f"[DEMO] {msg}", flush=True)


class Managed:
    """A child process in its own process group, so one stop reaches everything it spawned."""

    def __init__(self, name, cmd, log_path, env=None, cwd=None, keep_stdin=False):
        self.name = name
        self.logf = open(log_path, "ab")
        self.p = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                  stdin=subprocess.PIPE if keep_stdin else subprocess.DEVNULL,
                                  start_new_session=True)

    def alive(self):
        return self.p.poll() is None

    def stop(self, grace=6.0):
        if self.alive():
            for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 4.0), (signal.SIGKILL, 2.0)):
                try:
                    os.killpg(self.p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.p.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.logf.close()


class RunLogTail:
    """Latest row of the tracker's CSV run log (the single source of truth for what the drone is doing)."""

    def __init__(self, directory, newer_than):
        self.dir, self.t0, self.path = Path(directory), newer_than, None

    def find(self):
        if self.path is None and self.dir.exists():
            c = [p for p in self.dir.glob("run_*.csv") if p.stat().st_mtime >= self.t0 - 1]
            self.path = max(c, key=lambda p: p.stat().st_mtime) if c else None
        return self.path

    def last(self):
        p = self.find()
        if p is None:
            return {}
        try:
            with open(p, "rb") as fh:
                head = fh.readline().decode().strip().split(",")
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                lines = fh.read().decode("utf-8", "replace").splitlines()
            for ln in reversed(lines[1:] if size > 4096 else lines[1:]):
                cells = ln.split(",")
                if len(cells) == len(head):
                    return dict(zip(head, cells))
        except (OSError, ValueError):
            pass
        return {}


def fnum(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class ActorMover(threading.Thread):
    """Moves the sprinter and car in Gazebo (wall-clock, 30 Hz). `send` is injectable so tests need no simulator."""

    def __init__(self, send, rate=30.0):
        super().__init__(daemon=True)
        self.send, self.dt = send, 1.0 / rate
        self.seg, self.t_go = None, None
        self.stop_ev = threading.Event()
        self.failures = 0
        self.lock = threading.Lock()

    def set_segment(self, seg):
        with self.lock:
            self.seg, self.t_go = seg, None

    def go(self):
        with self.lock:
            self.t_go = time.monotonic()

    def run(self):
        while not self.stop_ev.is_set():
            with self.lock:
                seg, t_go = self.seg, self.t_go
            if seg is not None and self.send is not None:
                t_move = -1.0 if t_go is None else time.monotonic() - t_go
                for name, (x, y, z, yaw) in sp.actor_poses(seg, t_move).items():
                    try:
                        ok = self.send(name, x, y, z, yaw)
                    except Exception:
                        ok = False
                    self.failures = 0 if ok else self.failures + 1
            time.sleep(self.dt)


def gz_sender():
    import math
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.transport13 import Node
    node = Node()

    def send(name, x, y, z, yaw):
        req = Pose()
        req.name = name
        req.position.x, req.position.y, req.position.z = x, y, z
        req.orientation.w, req.orientation.z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        ok, _ = node.request("/world/tracker_world/set_pose", req, Pose, Boolean, 500)
        return ok
    return send


class Show:
    def __init__(self, args, doc, segments, out_dir):
        self.a, self.doc, self.segs, self.out = args, doc, segments, out_dir
        self.procs, self.abort = [], threading.Event()
        self.status = dict(title=doc.get("name", "Drone Tracker"), phase="starting", index=0, count=len(segments),
                           segment="", note="", progress=0.0, message="", locked=False, visible=False, gs=0.0,
                           target_speed=0.0, alt=0.0, fps=0.0, reid="", elapsed=0.0, battery=0.0)
        self.t_start = time.monotonic()
        self.cmd_path = out_dir / "commands.txt"
        self.cmd_path.touch(mode=0o600)
        self.tail = None
        self.mover = None
        self.last_lock_cmd = 0.0

    # -- plumbing -----------------------------------------------------
    def cmd(self, line):
        with open(self.cmd_path, "a") as fh:
            fh.write(line + "\n")

    def set(self, **kw):
        self.status.update(kw)

    def refresh(self):
        r = self.tail.last() if self.tail else {}
        self.status.update(
            locked=r.get("locked_id", "") != "", visible=r.get("target_visible") == "1",
            gs=fnum(r.get("gs")), alt=fnum(r.get("alt")), fps=fnum(r.get("fps")),
            reid=r.get("reid", ""), battery=fnum(r.get("battery")), elapsed=time.monotonic() - self.t_start)

    def wait(self, cond, timeout, poll=0.25, what=""):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.abort.is_set():
                return False
            self.check_alive()
            self.refresh()
            if cond():
                return True
            time.sleep(poll)
        return False

    def check_alive(self):
        for p in self.procs:
            if not p.alive() and p.name in ("tracker", "sim"):
                self.set(message=f"{p.name} stopped unexpectedly: ending the show safely")
                say(f"{p.name} stopped unexpectedly (exit code {p.p.returncode}); aborting. Last lines of {p.name}.log:")
                try:
                    for ln in (self.out / f"{p.name}.log").read_text(errors="replace").splitlines()[-25:]:
                        say(f"    | {ln}")
                except OSError:
                    pass
                self.abort.set()

    def locked_now(self):
        return self.status["locked"] and self.status["visible"]

    def lock_class(self, seg):
        return sp.ACTORS[seg["actor"]][2] if seg is not None else "largest"

    def ensure_lock(self, seg):
        if self.a.lock == "auto" and time.monotonic() - self.last_lock_cmd > 3.0:
            self.cmd(f"lock {self.lock_class(seg)}")
            self.last_lock_cmd = time.monotonic()

    # -- phases -------------------------------------------------------
    def run_segments(self):
        scale = self.a.time_scale
        for i, seg in enumerate(self.segs, 1):
            if self.abort.is_set():
                return
            self.set(index=i, segment=seg["name"], note=seg["note"], target_speed=0.0, progress=0.0,
                     phase="acquiring target", message="")
            if self.mover:
                self.mover.set_segment(seg)
            self.cmd("release")
            self.cmd(f"mark {seg['name']}")
            self.last_lock_cmd = 0.0
            if self.a.lock == "manual":
                self.set(message="Click the highlighted target in the tracker window to lock it.")
            stable = {"n": 0}

            def got_lock():
                stable["n"] = stable["n"] + 1 if self.locked_now() else 0
                if not self.locked_now():
                    self.ensure_lock(seg)
                return stable["n"] >= 6
            locked = self.wait(got_lock, self.doc.get("lock_timeout_s", 25) * (60.0 / 25.0 if self.a.lock == "manual" else 1.0)
                               * scale)
            if self.abort.is_set():
                return
            if not locked:
                say(f"{seg['name']}: no stable lock in time; starting the run anyway (segment will be flagged).")
            self.set(message="")
            self.wait(lambda: False, 4.0 * scale)               # let the drone settle on the target
            self.cmd(f"mark {seg['name']}{rp.MOVE_SUFFIX}")
            if self.mover:
                self.mover.go()
            self.set(phase="chasing", target_speed=seg["speed"])
            t0, dur = time.monotonic(), seg["run_s"] * scale
            while time.monotonic() - t0 < dur and not self.abort.is_set():
                self.set(progress=(time.monotonic() - t0) / dur)
                self.refresh()
                self.check_alive()
                if not self.status["locked"]:
                    self.ensure_lock(seg)
                if self.mover and self.mover.failures > 60:
                    self.set(message="Lost contact with the simulator: ending the show safely")
                    self.abort.set()
                time.sleep(0.25)
            self.set(progress=1.0)

    def finish(self):
        self.set(phase="returning home", message="", segment="", note="", target_speed=0.0)
        self.cmd("release")
        self.cmd("mark landing")
        self.cmd("rtl")
        if self.a.test_stack:
            time.sleep(1.0)
        else:
            self.wait(lambda: fnum(self.tail.last().get("airborne"), 1) == 0, 150.0, poll=1.0)
        self.set(phase="finished")


def stack_commands(args, doc, out):
    """The exact commands a person would type, in the order they would type them."""
    py = sys.executable
    sim = ["bash", str(ROOT / "sim" / "launch_sim.sh"), "ardupilot", doc.get("scene", "highway-fast")]
    tr = [py, str(ROOT / "tracker.py"), "--config", str(args.config), "--profile", doc.get("profile", "highway"),
          "--connect", args.connect, "--source", args.source, "--cmd-file", str(out / "commands.txt"),
          "--record", str(out / "annotated_raw.mp4")]
    if args.headless:
        tr.append("--no-display")
    if args.test_stack:
        tr.append("--no-fly")
    return sim, tr


def make_server(show, out, port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/status.json"):
                body, ctype = json.dumps(show.status).encode(), "application/json"
            else:
                body, ctype = LIVE_HTML.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def open_url(url):
    for opener in ("xdg-open", "gio"):
        if shutil.which(opener):
            subprocess.Popen([opener, url] if opener == "xdg-open" else [opener, "open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return


def preflight_files(args, doc):
    problems = []
    cfg = yaml.safe_load(open(args.config))
    mp = Path(cfg["model"]["path"])
    if not (mp if mp.is_absolute() else ROOT / mp).exists():
        problems.append(f"model file missing: {cfg['model']['path']} (python3 model_download.py)")
    if args.mode == "sim" and not args.use_running_sim:
        world = ROOT / "sim" / "worlds" / ("highway_world_fast.sdf" if doc.get("scene") == "highway-fast" else "highway_world.sdf")
        if not world.exists():
            problems.append(f"world missing: {world} (see docs/SIM_VALIDATION.md to generate it)")
        for tool in ("gz",):
            if not shutil.which(tool):
                problems.append(f"{tool} not found on PATH")
        if not (Path.home() / "ardupilot").exists():
            problems.append("~/ardupilot not found")
    return problems, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "hardware"], default="sim")
    ap.add_argument("--scenario", default=str(ROOT / "demo" / "scenarios" / "exhibition.yaml"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--lock", choices=["auto", "manual"], default="auto",
                    help="auto: the launcher locks each target for you (clicking also works). manual: you click it.")
    ap.add_argument("--connect", default="127.0.0.1:14550", help="tracker's flight-controller link")
    ap.add_argument("--monitor", default="udpin:127.0.0.1:14551", help="second link for readiness checks (sim)")
    ap.add_argument("--source", default="gz:/drone/camera", help="tracker video source (hardware: camera index / URL)")
    ap.add_argument("--use-running-sim", action="store_true", help="do not start the simulator (already running)")
    ap.add_argument("--headless", action="store_true", help="no tracker window (still records the video)")
    ap.add_argument("--no-open", action="store_true", help="do not open the live page / report in the browser")
    ap.add_argument("--live-port", type=int, default=8090)
    ap.add_argument("--out", default=str(ROOT / "logs" / "demo"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--duration", type=float, default=240.0, help="hardware: seconds to follow after lock, then return home")
    ap.add_argument("--auto-takeoff", action="store_true", help="hardware: let the launcher take off (default: pilot does)")
    ap.add_argument("--time-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--test-stack", action="store_true", help=argparse.SUPPRESS)   # tracker --no-fly, no sim (tests)
    args = ap.parse_args(argv)

    doc = yaml.safe_load(open(args.scenario))
    if args.mode == "hardware":
        doc["segments"] = [dict(name="Live pursuit", actor="person", speed=1.0, run_s=args.duration,
                                note="Operator-selected target, live flight.")]
    segs = sp.load_segments(doc)
    problems, cfg = preflight_files(args, doc)
    prof_cfg = apply_profile(cfg, doc.get("profile", "highway"))
    alt = prof_cfg["flight"]["target_altitude"]

    total = sum(s["run_s"] for s in segs)
    say(f"{doc.get('name')}: {len(segs)} segments, about {total:.0f} s of chasing "
        f"(~{(total + 8 * len(segs)) / 60:.1f} min with locking). Profile {doc.get('profile')}, altitude {alt:.0f} m, lock {args.lock}.")
    for i, s in enumerate(segs, 1):
        say(f"  {i}. {s['name']:<44} {s['speed']:>5.1f} m/s  {s['run_s']:>3.0f} s  road {s['x0']:.0f}-{s['x0'] + s['distance']:.0f} m")
    for p in problems:
        say(f"PROBLEM: {p}")
    if args.dry_run:
        say("dry run: nothing started." + (" Fix the problems above first." if problems else " Ready."))
        return 1 if problems else 0
    if problems and not args.test_stack:
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    show = Show(args, doc, segs, out)
    sim_cmd, tr_cmd = stack_commands(args, doc, out)
    started = time.time()
    run_dir = Path(cfg.get("run_log", {}).get("dir", "logs/runs"))
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    show.tail = RunLogTail(run_dir, started)

    def on_signal(signum, frame):
        say("interrupt: returning the drone home and stopping.")
        show.abort.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    srv = make_server(show, out, args.live_port)
    live = f"http://127.0.0.1:{args.live_port}/"
    say(f"show control page: {live}")
    if not args.no_open:
        open_url(live)

    try:
        if args.mode == "sim" and not args.use_running_sim and not args.test_stack:
            show.set(phase="starting simulator (a minute or two)")
            env = dict(os.environ, DEMO_NONINTERACTIVE="1")
            sim = Managed("sim", sim_cmd, out / "sim.log", env=env, keep_stdin=True)
            show.procs.append(sim)
            if not show.wait(lambda: b"ALL SYSTEMS RUNNING" in (out / "sim.log").read_bytes(), 300.0, poll=1.0):
                raise RuntimeError("simulator did not come up (see sim.log)")
        if not args.test_stack:
            from preflight import wait_ekf_ready, hardware_checklist
            if args.mode == "sim":
                show.set(phase="waiting for GPS / EKF")
                if not wait_ekf_ready(args.monitor, 240.0, say):
                    raise RuntimeError("autopilot never became ready to arm (see sim.log)")
            else:
                show.set(phase="pre-flight checklist")
                bad = False
                for st, text in hardware_checklist(args.monitor):
                    say(f"  [{st}] {text}")
                    bad |= st == "FAIL"
                if bad:
                    raise RuntimeError("pre-flight checklist failed: not flying")
                if input("[DEMO] Area clear, pilot ready with RC override? Type FLY to start: ").strip() != "FLY":
                    raise RuntimeError("not confirmed")

        show.set(phase="starting tracker")
        tracker = Managed("tracker", tr_cmd, out / "tracker.log", cwd=str(ROOT))
        show.procs.append(tracker)
        if not show.wait(lambda: show.tail.find() is not None, 120.0, poll=0.5):
            raise RuntimeError("tracker never started (see tracker.log)")
        if not show.wait(lambda: fnum(show.tail.last().get("fps")) > 1.0 or args.test_stack, 90.0, poll=0.5):
            raise RuntimeError("tracker produced no video frames (camera topic? see tracker.log)")

        if args.mode == "sim" and not args.test_stack:
            try:
                show.mover = ActorMover(gz_sender())
            except ImportError:
                raise RuntimeError("Gazebo Python bindings missing: sudo apt install python3-gz-transport13")
            show.mover.set_segment(segs[0])
            show.mover.start()

        if args.mode == "sim" or args.auto_takeoff:
            show.set(phase=f"taking off to {alt:.0f} m")
            show.cmd("takeoff")
            ok = show.wait(lambda: fnum(show.tail.last().get("airborne")) == 1 and
                           (args.test_stack or fnum(show.tail.last().get("alt")) >= alt - 1.5), 150.0, poll=0.5)
            if not ok and not show.abort.is_set():
                raise RuntimeError("takeoff did not complete (see tracker.log)")
        else:
            show.set(phase="waiting for the pilot to take off")
            show.wait(lambda: fnum(show.tail.last().get("airborne")) == 1, 900.0, poll=1.0)

        if not show.abort.is_set():
            show.run_segments()
    except RuntimeError as e:
        say(f"ERROR: {e}")
        show.set(message=str(e), phase="error")
    finally:
        try:
            if show.tail and any(p.name == "tracker" and p.alive() for p in show.procs):
                show.finish()
                show.cmd("quit")
                for p in show.procs:
                    if p.name == "tracker":
                        try:
                            p.p.wait(20)
                        except subprocess.TimeoutExpired:
                            pass
        finally:
            if show.mover:
                show.mover.stop_ev.set()
            for p in reversed(show.procs):
                p.stop()
            if args.mode == "sim" and not args.use_running_sim and not args.test_stack and not os.environ.get("DEMO_NO_PKILL"):
                subprocess.run(["pkill", "-f", "gz sim"], stderr=subprocess.DEVNULL)
                subprocess.run(["pkill", "-x", "arducopter"], stderr=subprocess.DEVNULL)

    csvp = show.tail.find() if show.tail else None
    if csvp is None:
        say("no run log was produced; no report.")
        return 1
    video = ""
    raw = out / "annotated_raw.mp4"
    if raw.exists() and raw.stat().st_size > 0:
        final = out / "annotated.mp4"
        if shutil.which("ffmpeg") and subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(final)]).returncode == 0:
            raw.unlink()
            video = final.name
        else:
            video = raw.name
    shutil.copy(csvp, out / "run_log.csv")
    meta = f"{doc.get('name')} · {args.mode} · profile {doc.get('profile')} · {time.strftime('%Y-%m-%d %H:%M')}"
    summary = rp.build(str(out / "run_log.csv"), segs, str(out), video=video, meta=meta)
    say(f"RESULT: {summary['passed']}/{summary['total']} segments passed, top speed {summary['top_speed']:.1f} m/s")
    for s in summary["segments"]:
        say(f"  {s['verdict']:<8} {s['name']}")
    report = out / "report.html"
    say(f"report: {report}")
    if not args.no_open:
        open_url(report.as_uri())
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
