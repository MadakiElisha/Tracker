"""Turn a demo run log into a per-segment results page (self-contained HTML) and a JSON summary.

Segments are found through the `seg` column the tracker writes from `mark` commands:
    "<name>"        lock phase (target standing, drone acquiring it)
    "<name> >>"     moving phase (the chase)
"""
import csv
import html
import json
import math
import os
from statistics import median
from typing import Dict, List

MOVE_SUFFIX = " >>"
PASS_VISIBLE = 0.80       # target seen in at least this share of the chase
PARTIAL_VISIBLE = 0.50


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def load_rows(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _range_err(r):
    a, sp = _f(r.get("area")), _f(r.get("setpoint"))
    return math.sqrt(sp / a) - 1.0 if a > 0 and sp > 0 else None


def _pctl(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def longest_dropout(rows) -> float:
    best, start, last = 0.0, None, None
    for r in rows:
        lost = r.get("locked_id", "") != "" and r.get("target_visible") not in ("1", 1)
        t = _f(r.get("t"))
        if lost:
            start = t if start is None else start
            last = t
        else:
            if start is not None:
                best = max(best, last - start)
            start = None
    if start is not None:
        best = max(best, last - start)
    return best


def segment_stats(rows: List[dict], seg: dict) -> dict:
    name = seg["name"]
    lock_rows = [r for r in rows if r.get("seg") == name]
    move = [r for r in rows if r.get("seg") == name + MOVE_SUFFIX]
    out = dict(name=name, actor=seg["actor"], target_speed=seg["speed"], note=seg.get("note", ""),
               planned_s=seg["run_s"])
    first_t = _f(lock_rows[0]["t"]) if lock_rows else None
    locked_first = next((r for r in lock_rows + move if r.get("locked_id", "") != "" and
                         r.get("target_visible") in ("1", 1)), None)
    out["lock_time_s"] = (_f(locked_first["t"]) - first_t) if locked_first is not None and first_t is not None else None
    if len(move) < 5:
        out.update(verdict="NOT RUN", moving_s=0.0, visible=0.0, max_speed=0.0, median_fps=0.0,
                   range_med=None, range_p95=None, longest_dropout=0.0, relocks=0, lock_lost=True)
        return out
    t0, t1 = _f(move[0]["t"]), _f(move[-1]["t"])
    vis = sum(1 for r in move if r.get("target_visible") in ("1", 1)) / len(move)
    errs = [abs(e) for e in (_range_err(r) for r in move if r.get("target_visible") in ("1", 1)) if e is not None]
    gs = [_f(r.get("gs")) for r in move if r.get("gs", "") != ""]
    fps = [_f(r.get("fps")) for r in move if _f(r.get("fps")) > 0]
    ends_locked = move[-1].get("locked_id", "") != ""
    verdict = "PASS" if vis >= PASS_VISIBLE and ends_locked and out["lock_time_s"] is not None else \
              "PARTIAL" if vis >= PARTIAL_VISIBLE else "FAIL"
    out.update(verdict=verdict, moving_s=t1 - t0, visible=vis, max_speed=max(gs) if gs else 0.0,
               median_fps=median(fps) if fps else 0.0,
               range_med=median(errs) if errs else None, range_p95=_pctl(errs, 0.95),
               longest_dropout=longest_dropout(move),
               relocks=sum(1 for r in move if r.get("reid") == "relocked"), lock_lost=not ends_locked)
    return out


def summarise(rows: List[dict], segments: List[dict]) -> dict:
    segs = [segment_stats(rows, s) for s in segments]
    ran = [s for s in segs if s["verdict"] != "NOT RUN"]
    return dict(segments=segs, passed=sum(s["verdict"] == "PASS" for s in segs), total=len(segs),
                moving_s=sum(s["moving_s"] for s in ran),
                top_speed=max((s["max_speed"] for s in ran), default=0.0),
                relocks=sum(s["relocks"] for s in ran))


def _chart(rows, segments, w=980, h=220) -> str:
    pts = [(_f(r["t"]), _f(r.get("gs"))) for r in rows if r.get("gs", "") != ""]
    if len(pts) < 2:
        return ""
    tmax = max(p[0] for p in pts) or 1.0
    vmax = max(max(p[1] for p in pts), max(s["speed"] for s in segments), 1.0) * 1.1
    X = lambda t: 40 + (w - 50) * t / tmax
    Y = lambda v: h - 22 - (h - 40) * v / vmax
    poly = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts[:: max(1, len(pts) // 1500)])
    bands, lost = [], []
    for s in segments:
        mv = [r for r in rows if r.get("seg") == s["name"] + MOVE_SUFFIX]
        if len(mv) > 2:
            a, b = _f(mv[0]["t"]), _f(mv[-1]["t"])
            bands.append(f'<rect x="{X(a):.1f}" y="10" width="{X(b) - X(a):.1f}" height="{h - 32}" class="band"/>'
                         f'<line x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{Y(s["speed"]):.1f}" y2="{Y(s["speed"]):.1f}" class="tgt"/>')
    run_start = None
    for r in rows:
        isl = r.get("locked_id", "") != "" and r.get("target_visible") not in ("1", 1)
        t = _f(r["t"])
        if isl and run_start is None:
            run_start = t
        elif not isl and run_start is not None:
            lost.append(f'<rect x="{X(run_start):.1f}" y="{h - 22}" width="{max(1.5, X(t) - X(run_start)):.1f}" height="6" class="lost"/>')
            run_start = None
    grid = "".join(f'<text x="2" y="{Y(v) + 4:.1f}" class="ax">{v:.0f}</text>' for v in range(0, int(vmax) + 1, 5))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Drone ground speed over the show">'
            f'{grid}{"".join(bands)}<polyline points="{poly}" class="spd"/>{"".join(lost)}'
            f'<text x="{w - 6}" y="{h - 4}" text-anchor="end" class="ax">time, s (red marks: target not seen)</text></svg>')


CSS = """
:root{--bg:#fff;--fg:#14171c;--mut:#5a6270;--card:#f4f6f9;--line:#d8dde5;--ok:#0a7d3c;--warn:#8a5a00;--bad:#b3261e;--acc:#1d5fd1}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1318;--fg:#e8ebf0;--mut:#9aa3b0;--card:#171c24;--line:#2a313c;--ok:#4cc27d;--warn:#e0a63a;--bad:#ff7b72;--acc:#6ea0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
h1{margin:0 0 4px;font-size:1.6rem}.sub{color:var(--mut);margin:0 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:1.6rem}.kpi span{color:var(--mut);font-size:.85rem}
table{width:100%;border-collapse:collapse;font-size:.92rem}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600}.wrap{overflow-x:auto}
.v{font-weight:700;padding:2px 8px;border-radius:6px;border:1px solid currentColor}
.PASS{color:var(--ok)}.PARTIAL{color:var(--warn)}.FAIL,.NOTRUN{color:var(--bad)}
svg{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
.spd{fill:none;stroke:var(--acc);stroke-width:2}.tgt{stroke:var(--warn);stroke-width:2;stroke-dasharray:6 4}
.band{fill:var(--acc);opacity:.07}.lost{fill:var(--bad)}.ax{fill:var(--mut);font-size:11px}
video{width:100%;border-radius:10px;border:1px solid var(--line);margin-top:8px}
h2{margin:28px 0 8px;font-size:1.15rem}.note{color:var(--mut);font-size:.85rem}
"""


def _fmt(v, nd=1, unit=""):
    return "–" if v is None else f"{v:.{nd}f}{unit}"


def render_html(summary: dict, title: str, chart: str, video: str = "", meta: str = "") -> str:
    rows = []
    for s in summary["segments"]:
        v = s["verdict"]
        rows.append(
            f'<tr><td><b>{html.escape(s["name"])}</b><div class="note">{html.escape(s["note"])}</div></td>'
            f'<td><span class="v {v.replace(" ", "")}">{v}</span></td>'
            f'<td>{_fmt(s["lock_time_s"], 1, " s")}</td><td>{s["visible"] * 100:.0f}%</td>'
            f'<td>{s["max_speed"]:.1f} / {s["target_speed"]:.1f}</td>'
            f'<td>{_fmt(s["range_med"] * 100 if s["range_med"] is not None else None, 0, "%")}</td>'
            f'<td>{s["longest_dropout"]:.1f} s</td><td>{s["relocks"]}</td></tr>')
    k = lambda b, l: f'<div class="kpi"><b>{b}</b><span>{l}</span></div>'
    vid = (f'<h2>Annotated flight video</h2><video controls preload="metadata" src="{html.escape(video)}"></video>'
           if video else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="sub">{html.escape(meta)}</p>
<div class="kpis">{k(f'{summary["passed"]}/{summary["total"]}', 'segments passed')}
{k(f'{summary["top_speed"]:.1f} m/s', 'top drone ground speed')}
{k(f'{summary["moving_s"]:.0f} s', 'time chasing')}{k(summary["relocks"], 'target re-identifications')}</div>
<h2>Per segment</h2><div class="wrap"><table><thead><tr><th>Segment</th><th>Result</th><th>Time to lock</th>
<th>Target in view</th><th>Drone speed / target, m/s</th><th>Typical distance error</th><th>Longest dropout</th><th>Re-IDs</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="note">PASS = target in view for at least {PASS_VISIBLE * 100:.0f}% of the chase and still locked at the end.
Distance error is the median of |sqrt(setpoint area / measured area) − 1|, a unitless range error (0% = exactly on the set distance).
This is a simulation result, not a flight-test result.</p>
<h2>Drone speed through the show</h2>{chart}{vid}</main></body></html>"""


def build(run_csv: str, segments: List[dict], out_dir: str, title: str = "Drone Tracker – exhibition results",
          video: str = "", meta: str = "") -> Dict:
    rows = load_rows(run_csv)
    summary = summarise(rows, segments)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "report.html"), "w") as fh:
        fh.write(render_html(summary, title, _chart(rows, segments), video, meta))
    return summary
