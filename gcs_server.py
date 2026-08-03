#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Browser-based GCS Tap-to-Lock Server

 Streams the annotated video feed over HTTP (MJPEG) and
 accepts tap coordinates via POST to select a target.

 Access from any device on the same network:
   http://<WSL2_OR_HOST_IP>:8080

 Integration:
   from gcs_server import GCSServer
   gcs = GCSServer(port=8080)
   gcs.start()
   gcs.update_frame(annotated_frame)      # call every loop
   tap = gcs.consume_tap()                # returns (x, y) or None
=============================================================
"""

import threading
import time
import cv2
from flask import Flask, Response, request, jsonify


class GCSServer:
    def __init__(self, port=8080, frame_w=640, frame_h=480):
        self.port    = port
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._frame      = None
        self._frame_lock = threading.Lock()

        self._tap        = None
        self._tap_lock   = threading.Lock()

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):

        @self.app.route('/')
        def index():
            return f"""
            <html>
            <head>
                <title>Drone Tracker GCS</title>
                <style>
                    body {{ background:#111; margin:0; text-align:center;
                            font-family: monospace; color:#0f0; }}
                    #wrap {{ position:relative; display:inline-block; }}
                    img {{ width:{self.frame_w}px; height:{self.frame_h}px;
                           cursor:crosshair; }}
                    #status {{ padding:8px; font-size:14px; }}
                </style>
            </head>
            <body>
                <div id="status">Tap on the video to lock a target</div>
                <div id="wrap">
                    <img id="feed" src="/stream">
                </div>
                <script>
                    const img = document.getElementById('feed');
                    const status = document.getElementById('status');
                    img.addEventListener('click', function(e) {{
                        const rect = img.getBoundingClientRect();
                        // Scale click position to actual frame resolution
                        const x = Math.round((e.clientX - rect.left) *
                                  ({self.frame_w} / rect.width));
                        const y = Math.round((e.clientY - rect.top) *
                                  ({self.frame_h} / rect.height));
                        fetch('/tap', {{
                            method: 'POST',
                            headers: {{'Content-Type':'application/json'}},
                            body: JSON.stringify({{x: x, y: y}})
                        }}).then(r => r.json()).then(d => {{
                            status.innerText = 'Tap sent: (' + x + ', ' + y + ')';
                        }});
                    }});
                </script>
            </body>
            </html>
            """

        @self.app.route('/stream')
        def stream():
            return Response(self._mjpeg_generator(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/tap', methods=['POST'])
        def tap():
            data = request.get_json()
            x, y = int(data['x']), int(data['y'])
            with self._tap_lock:
                self._tap = (x, y)
            return jsonify({'status': 'ok', 'x': x, 'y': y})

    def _mjpeg_generator(self):
        while True:
            with self._frame_lock:
                frame = None if self._frame is None else self._frame.copy()

            if frame is not None:
                ok, jpeg = cv2.imencode('.jpg', frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)   # ~25 FPS stream cap

    def update_frame(self, frame):
        """Call this every loop iteration with the annotated frame."""
        with self._frame_lock:
            self._frame = frame

    def consume_tap(self):
        """Returns (x, y) once, then resets to None. Call every frame."""
        with self._tap_lock:
            t = self._tap
            self._tap = None
            return t

    def start(self):
        def _run():
            self.app.run(host='0.0.0.0', port=self.port,
                         debug=False, use_reloader=False,
                         threaded=True)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        print(f"[GCS_SERVER] Running on http://0.0.0.0:{self.port}")
        print(f"[GCS_SERVER] Open from Windows browser at:")
        print(f"             http://<WSL2-IP>:{self.port}")


if __name__ == '__main__':
    import numpy as np
    gcs = GCSServer()
    gcs.start()

    print("Streaming test pattern. Open the URL above and tap the frame.")
    try:
        while True:
            frame = np.full((480, 640, 3), 60, dtype='uint8')
            cv2.putText(frame, "TEST FEED", (220, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            gcs.update_frame(frame)

            tap = gcs.consume_tap()
            if tap:
                print(f"\n[TAP RECEIVED] {tap}")

            time.sleep(0.04)
    except KeyboardInterrupt:
        print("\nStopped.")
