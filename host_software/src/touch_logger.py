"""
src/touch_logger.py

Host side of the bidirectional link. Reads the MCU's touchscreen telemetry on a
background thread, joins each record to the vision frame it was taken against,
and appends a row to a CSV.

Why a thread: main.py's loop is dominated by YOLO inference (tens of ms). If we
drained the serial port inline the OS RX buffer would overflow at 60 Hz and we
would silently lose ground-truth samples -- the exact data we are trying to
collect. The reader thread does nothing but read, parse and write.

Thread safety: only the main thread ever WRITES to the serial port, only this
thread ever READS from it. pyserial supports that split; do not add a second
writer without a lock.

Uplink record (see TouchProbe.h):
    T,<seq>,<mcu_ms>,<touch_x>,<touch_y>,<valid>,<vis_x>,<vis_y>,<a>,<b>,<c>
with all coordinates in HUNDREDTHS of a millimetre.
"""

import csv
import os
import threading
import time
from collections import OrderedDict

CSV_FIELDS = [
    "seq",
    "host_send_ts",  # perf_counter when the vision frame was sent to the MCU
    "host_recv_ts",  # perf_counter when this telemetry line was parsed
    "rtt_ms",  # recv - send; sensor-to-log round trip for this frame
    "mcu_ms",  # MCU millis() at sample time
    "vision_x_mm",  # what the PC computed and sent
    "vision_y_mm",
    "target_x_mm",  # target in force for that frame
    "target_y_mm",
    "mcu_vision_x_mm",  # what the MCU was actually acting on (echo)
    "mcu_vision_y_mm",
    "touch_x_mm",  # GROUND TRUTH
    "touch_y_mm",
    "touch_valid",
    "err_x_mm",  # vision - touch
    "err_y_mm",
    "err_mm",  # euclidean
    "motor_a",  # actual stepper positions
    "motor_b",
    "motor_c",
]

# How many in-flight vision frames to remember while waiting for their
# telemetry to come back. 256 frames at ~30 Hz is ~8 s of slack, far more than
# the link's actual latency, and it bounds memory.
PENDING_MAX = 256


class TouchTelemetryLogger:
    def __init__(self, ser, csv_path, print_status_lines=True):
        self.ser = ser
        self.csv_path = os.path.abspath(csv_path)
        self.print_status_lines = print_status_lines

        self._pending = OrderedDict()  # seq -> (send_ts, vx, vy, tx, ty)
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._buf = bytearray()

        # counters, read from the main thread for the console line
        self.rows = 0
        self.parse_errors = 0
        self.unmatched = 0
        self.touch_lost = 0
        self._last_err_mm = None

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    # -- main-thread API ----------------------------------------------------

    def register_frame(self, seq, vision_x, vision_y, target_x, target_y, send_ts):
        """Call immediately after writing a V,... line, with the same values."""
        with self._pending_lock:
            self._pending[seq] = (
                send_ts,
                float(vision_x),
                float(vision_y),
                float(target_x),
                float(target_y),
            )
            while len(self._pending) > PENDING_MAX:
                self._pending.popitem(last=False)

    @property
    def last_err_mm(self):
        return self._last_err_mm

    def start(self):
        if self.ser is None:
            print("[touch-log] no serial port; ground-truth logging disabled.")
            return
        self._thread = threading.Thread(
            target=self._run, name="touch-reader", daemon=True
        )
        self._thread.start()
        print(f"[touch-log] logging touchscreen ground truth to {self.csv_path}")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        print(
            f"[touch-log] {self.rows} rows | {self.unmatched} unmatched seq | "
            f"{self.touch_lost} no-contact | {self.parse_errors} bad lines"
        )

    # -- reader thread ------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            try:
                n = self.ser.in_waiting
                chunk = self.ser.read(n if n else 1)
            except Exception as e:
                if not self._stop.is_set():
                    print(f"[touch-log] serial read error: {e}")
                time.sleep(0.05)
                continue

            if not chunk:
                time.sleep(0.001)
                continue

            self._buf.extend(chunk)
            # Guard against a wedged link filling RAM with a partial line.
            if len(self._buf) > 65536:
                del self._buf[:-1024]

            while True:
                i = self._buf.find(b"\n")
                if i < 0:
                    break
                raw = bytes(self._buf[:i])
                del self._buf[: i + 1]
                self._handle_line(raw.strip())

    def _handle_line(self, raw):
        if not raw:
            return
        try:
            line = raw.decode("ascii", errors="ignore").strip()
        except Exception:
            self.parse_errors += 1
            return

        if not line.startswith("T,"):
            # '#' status lines and any stray firmware println. Never a record.
            if self.print_status_lines and line:
                print(f"[mcu] {line}")
            return

        parts = line.split(",")
        if len(parts) != 11:
            self.parse_errors += 1
            return

        try:
            seq = int(parts[1])
            mcu_ms = int(parts[2])
            touch_x = int(parts[3]) / 100.0
            touch_y = int(parts[4]) / 100.0
            valid = int(parts[5])
            mcu_vis_x = int(parts[6]) / 100.0
            mcu_vis_y = int(parts[7]) / 100.0
            mot_a = int(parts[8])
            mot_b = int(parts[9])
            mot_c = int(parts[10])
        except ValueError:
            self.parse_errors += 1
            return

        recv_ts = time.perf_counter()

        with self._pending_lock:
            sent = self._pending.get(seq)

        if sent is None:
            self.unmatched += 1
            send_ts = ""
            rtt_ms = ""
            # Fall back to the MCU's echo so the row is still usable.
            vis_x, vis_y = mcu_vis_x, mcu_vis_y
            tgt_x = tgt_y = ""
        else:
            send_ts, vis_x, vis_y, tgt_x, tgt_y = sent
            rtt_ms = round((recv_ts - send_ts) * 1000.0, 3)

        if valid:
            err_x = vis_x - touch_x
            err_y = vis_y - touch_y
            err = (err_x * err_x + err_y * err_y) ** 0.5
            self._last_err_mm = err
        else:
            # No ball on the plate -> no ground truth -> no error. Leaving these
            # blank rather than 0 keeps them out of any mean you compute later.
            self.touch_lost += 1
            err_x = err_y = err = ""

        row = {
            "seq": seq,
            "host_send_ts": round(send_ts, 6) if send_ts != "" else "",
            "host_recv_ts": round(recv_ts, 6),
            "rtt_ms": rtt_ms,
            "mcu_ms": mcu_ms,
            "vision_x_mm": round(vis_x, 3),
            "vision_y_mm": round(vis_y, 3),
            "target_x_mm": round(tgt_x, 3) if tgt_x != "" else "",
            "target_y_mm": round(tgt_y, 3) if tgt_y != "" else "",
            "mcu_vision_x_mm": mcu_vis_x,
            "mcu_vision_y_mm": mcu_vis_y,
            "touch_x_mm": touch_x,
            "touch_y_mm": touch_y,
            "touch_valid": valid,
            "err_x_mm": round(err_x, 3) if err_x != "" else "",
            "err_y_mm": round(err_y, 3) if err_y != "" else "",
            "err_mm": round(err, 3) if err != "" else "",
            "motor_a": mot_a,
            "motor_b": mot_b,
            "motor_c": mot_c,
        }

        try:
            self._writer.writerow(row)
            self.rows += 1
            # Flush periodically, not per row: a Ctrl-C should not cost you more
            # than a fraction of a second of data, but per-row fsync at 60 Hz is
            # pointless I/O.
            if self.rows % 60 == 0:
                self._fh.flush()
        except Exception as e:
            self.parse_errors += 1
            print(f"[touch-log] csv write error: {e}")
