import cv2
import threading
import queue
import time
import socket
import struct
import collections
import numpy as np


class USBReceiver:
    def __init__(self, camera_id=0, width=640, height=480, auto_exposure=None):
        """auto_exposure: None leaves the camera's power-on default untouched.
        Otherwise a value passed straight to cv2.CAP_PROP_AUTO_EXPOSURE -- the
        convention for this property is backend/driver-dependent (0.25 vs 0.75
        vs 1 vs 0 all mean different things depending on OpenCV/DirectShow/V4L2
        combination), so probe_camera_modes.py exists to find the value that
        actually does something on this specific camera before trusting a
        number here. Do not assume cv2.VideoCapture.get(CAP_PROP_FPS) reflects
        real achieved throughput after calling .set() -- confirmed this session
        it just echoes the request back; only the timed gaps below are real.
        """
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)

        # Force MJPG codec to prevent USB 2.0 bandwidth bottlenecks
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if auto_exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)

        # Target size for the resize-guard below -- must track whatever was
        # actually requested, not a hardcoded 640x480, or testing a smaller
        # capture size (e.g. to check whether resolution is the throughput
        # constraint) would silently get upscaled straight back afterward,
        # defeating the point of the test and adding a pointless resize cost.
        self._target_w = width
        self._target_h = height

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True

        # Camera-only timing, isolated from everything downstream (ArUco/CNN/
        # serial/etc) -- reported independently of the main loop's own FPS
        # print, which only times work *after* get_latest_frame() returns and
        # so has always been blind to camera-side stalls. This is the only
        # trustworthy source of the camera's real achieved rate -- see
        # PROJECT_LOGBOOK.md 19/08 for why cap.get(CAP_PROP_FPS) is not.
        self._read_durations_ms = collections.deque(maxlen=200)
        self._frame_gaps_ms = collections.deque(maxlen=200)
        self._last_frame_ts = None
        self._last_report_t = time.perf_counter()

        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        if self.cap.isOpened():
            self.thread.start()
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(
                f"USB Camera {camera_id} initialized (requested {width}x{height}, "
                f"actual {actual_w:.0f}x{actual_h:.0f} -- watch the [camera] log lines "
                f"below for the real achieved fps, not this line)."
            )
        else:
            print(f"ERROR: Could not open USB Camera {camera_id}")

    def _receive_loop(self):
        while self.running and self.cap.isOpened():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            t1 = time.perf_counter()
            if ret and frame is not None:
                self._read_durations_ms.append((t1 - t0) * 1000.0)
                if self._last_frame_ts is not None:
                    self._frame_gaps_ms.append((t1 - self._last_frame_ts) * 1000.0)
                self._last_frame_ts = t1

                if frame.shape[:2] != (self._target_h, self._target_w):
                    frame = cv2.resize(frame, (self._target_w, self._target_h))
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

                if t1 - self._last_report_t >= 2.0 and len(self._read_durations_ms) > 5:
                    reads = list(self._read_durations_ms)
                    gaps = list(self._frame_gaps_ms)
                    mean_gap_ms = sum(gaps) / len(gaps) if gaps else 0.0
                    cam_fps = 1000.0 / mean_gap_ms if mean_gap_ms > 0 else 0.0
                    print(
                        f"[camera] cap.read(): mean={sum(reads)/len(reads):.1f}ms max={max(reads):.1f}ms | "
                        f"inter-frame gap: mean={mean_gap_ms:.1f}ms max={max(gaps) if gaps else 0.0:.1f}ms "
                        f"(~{cam_fps:.1f}fps camera-only, isolated from ArUco/CNN/serial)"
                    )
                    self._last_report_t = t1
            else:
                time.sleep(0.01)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.cap.release()


class UDPReceiver:
    def __init__(self, port=5001, width=640, height=480):
        self.port = port
        self.width = width
        self.height = height
        self.pixel_bytes = 2
        self.frame_size = self.width * self.height * self.pixel_bytes
        self.packet_payload = 1024
        self.packets_per_frame = (
            self.frame_size + self.packet_payload - 1
        ) // self.packet_payload

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(1.0)

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"UDP Receiver initialized on port {port}.")

    def _receive_loop(self):
        frame_buffer = bytearray(self.frame_size)
        current_frame_id = -1
        packets_received = 0

        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if len(data) < 4:
                    continue

                frame_id = struct.unpack("<H", data[0:2])[0]
                packet_id = struct.unpack("<H", data[2:4])[0]
                payload = data[4:]

                if frame_id != current_frame_id:
                    if (
                        current_frame_id != -1
                        and packets_received > self.packets_per_frame * 0.8
                    ):
                        # Process previous frame
                        img_np = np.frombuffer(frame_buffer, dtype=np.uint16).reshape(
                            (self.height, self.width)
                        )
                        # Convert RGB565 to BGR
                        b = ((img_np & 0x001F) << 3).astype(np.uint8)
                        g = ((img_np & 0x07E0) >> 3).astype(np.uint8)
                        r = ((img_np & 0xF800) >> 8).astype(np.uint8)
                        bgr_frame = cv2.merge([b, g, r])

                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put(bgr_frame)

                    current_frame_id = frame_id
                    packets_received = 0

                offset = packet_id * self.packet_payload
                length = len(payload)
                if offset + length <= self.frame_size:
                    frame_buffer[offset : offset + length] = payload
                    packets_received += 1

            except socket.timeout:
                pass
            except Exception as e:
                print(f"UDP Error: {e}")
                time.sleep(0.01)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.sock.close()
