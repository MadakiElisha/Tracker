#!/usr/bin/env python3
"""
Drone Tracker — OpenCV window click-to-lock.

Register once:  cv2.setMouseCallback(window, locker.handle_click)
Each loop:      click = locker.consume_click()   # (x, y) in frame pixels or None

Thread-safe. Only the most recent unconsumed left-click is kept.
"""
import threading
from typing import Optional, Tuple

import cv2


class MouseLocker:
    def __init__(self):
        self._lock = threading.Lock()
        self._click: Optional[Tuple[int, int]] = None

    def handle_click(self, event, x, y, flags=0, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self._lock:
                self._click = (int(x), int(y))

    def consume_click(self) -> Optional[Tuple[int, int]]:
        with self._lock:
            click, self._click = self._click, None
            return click
