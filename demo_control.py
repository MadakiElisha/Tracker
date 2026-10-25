"""Opt-in remote control and recording for scripted demos (off unless tracker.py gets --cmd-file / --record).

CommandChannel   a text file the demo launcher appends one command per line to; the tracker reads new lines
                 each frame. Local file, no network port, so it adds no new way into the drone.
VideoRecorder    writes the annotated frames (what the HUD shows) to a video file.

Commands:  takeoff | lock [largest|<class group or name>] | release | rtl | land | quit | mark <label> | profile <name>
"""
import os
from typing import List, Optional, Tuple

VERBS = {"takeoff", "lock", "release", "rtl", "land", "quit", "mark", "profile"}


def parse_command(line: str) -> Optional[Tuple[str, str]]:
    parts = line.strip().split(None, 1)
    if not parts:
        return None
    verb = parts[0].lower()
    if verb not in VERBS:
        return None
    return verb, (parts[1].strip() if len(parts) > 1 else "")


class CommandChannel:
    def __init__(self, path: str):
        self.path = path
        self._pos = 0
        self._buf = b""
        # Commands from an earlier show must not replay: start reading at the current end of the file.
        try:
            self._pos = os.path.getsize(path)
        except OSError:
            self._pos = 0

    def poll(self) -> List[Tuple[str, str]]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self._pos:                      # file replaced / truncated
            self._pos, self._buf = 0, b""
        if size == self._pos:
            return []
        with open(self.path, "rb") as fh:
            fh.seek(self._pos)
            data = fh.read(size - self._pos)
        self._pos += len(data)
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")   # keep a half-written last line for next time
        out = []
        for ln in lines:
            cmd = parse_command(ln.decode("utf-8", "replace"))
            if cmd:
                out.append(cmd)
        return out


def pick_target(tracks, wanted: str, target_classes: dict, class_groups: dict):
    """The largest track matching `wanted` ('' / 'largest' = any; else a group name like person/vehicle,
    or a class name like car). None if nothing matches."""
    wanted = (wanted or "largest").lower()
    if wanted in ("largest", "any"):
        pool = list(tracks)
    elif wanted in class_groups:
        ids = {int(i) for i in class_groups[wanted]}
        pool = [t for t in tracks if int(t[6]) in ids]
    else:
        ids = {int(k) for k, v in target_classes.items() if str(v).lower() == wanted}
        pool = [t for t in tracks if int(t[6]) in ids]
    if not pool:
        return None
    return max(pool, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))


class VideoRecorder:
    """Annotated-frame video. Browsers need H.264; the demo launcher converts afterwards if ffmpeg exists."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        import cv2
        self.path = path
        self._cv2 = cv2
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
        self.ok = self._w.isOpened()
        self.size = size

    def write(self, frame):
        if self.ok:
            h, w = frame.shape[:2]
            if (w, h) != self.size:
                frame = self._cv2.resize(frame, self.size)
            self._w.write(frame)

    def close(self):
        if self._w is not None:
            self._w.release()
            self._w = None
