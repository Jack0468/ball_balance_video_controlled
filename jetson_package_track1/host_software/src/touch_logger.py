"""
src/touch_logger.py

Host side of the bidirectional link. A single background thread owns the
entire serial handle -- both writing outgoing "V,<seq>,..."/"L" lines AND
reading the MCU's touchscreen telemetry back -- joins each ground-truth
record to the vision frame it was taken against, and (optionally) appends a
row to a CSV.

Why one thread owns BOTH directions, not just reading: an earlier version had
the main thread call ser.write() directly while this thread only read.
Live-recorded telemetry showed the vision loop's sustained throughput collapse
from ~95-107 FPS (no logger) to ~20-25 FPS (logger active) -- see
PROJECT_LOGBOOK.md. Regardless of whether that was GIL contention between the
two threads' Python-level work, or contention at the OS/driver level from two
threads touching one handle concurrently, consolidating all I/O onto this one
thread fixes either cause: the main thread now only ever enqueues outgoing
bytes (cheap, no serial access at all) and this thread drains the outgoing
queue before each read attempt, so writes are never delayed behind more than
one read-timeout window.

CSV logging is now an independent, optional feature of this same thread, not
a reason for it to exist -- pass csv_path=None (or omit it) to run this purely
as the serial I/O owner with no file written, e.g. when the caller wants the
performance benefit of single-thread serial ownership without paying for
telemetry persistence.

Thread safety: only this thread ever touches self.ser (read or write). The
main thread's only interaction is send_frame()/send_raw(), which just push
onto a thread-safe queue.Queue -- no lock needed for that hand-off.

Uplink record (see TouchProbe.h):
    T,<seq>,<mcu_ms>,<touch_x>,<touch_y>,<valid>,<vis_x>,<vis_y>,<a>,<b>,<c>
with all coordinates in HUNDREDTHS of a millimetre.

Downlink record (see SerialCoords.cpp):
    V,<seq>,<ball_x>,<ball_y>,<target_x>,<target_y>
"""

import csv
import os
import queue
import threading
import time
from collections import OrderedDict

CSV_FIELDS = [
    "seq",
    "host_send_ts",  # perf_counter when the vision frame was enqueued to send
    "host_recv_ts",  # perf_counter when this telemetry line was parsed
    "rtt_ms",  # recv - send; sensor-to-log round trip for this frame
    "mcu_ms",  # MCU millis() at sample time
    "vision_x_mm",  # what was actually SENT (post-gate/dead-band/MLP -- the
    "vision_y_mm",   # fully processed output, NOT the raw CNN reading)
    "raw_vision_x_mm",  # the raw CNN estimate BEFORE the gate/dead-band/MLP --
    "raw_vision_y_mm",   # this is what a from-scratch filter (e.g. a Kalman
                          # filter meant to replace that stack) needs to
                          # characterize its measurement noise (R) from, not
                          # vision_x_mm/vision_y_mm above. Blank if the caller
                          # didn't pass raw_x/raw_y to send_frame().
    "target_x_mm",  # target in force for that frame
    "target_y_mm",
    "mcu_vision_x_mm",  # what the MCU was actually acting on (echo)
    "mcu_vision_y_mm",
    "touch_x_mm",  # GROUND TRUTH
    "touch_y_mm",
    "touch_valid",
    "err_x_mm",  # vision (processed/sent) - touch
    "err_y_mm",
    "err_mm",  # euclidean
    "raw_err_x_mm",  # raw_vision - touch -- the actual measurement-noise signal
    "raw_err_y_mm",
    "raw_err_mm",
    "motor_a",  # actual stepper positions
    "motor_b",
    "motor_c",
]

# How many in-flight vision frames to remember while waiting for their
# telemetry to come back. 256 frames at ~30 Hz is ~8 s of slack, far more than
# the link's actual latency, and it bounds memory.
PENDING_MAX = 256


class TouchTelemetryLogger:
    def __init__(self, ser, csv_path=None, print_status_lines=True):
        self.ser = ser
        self.csv_path = os.path.abspath(csv_path) if csv_path else None
        self.print_status_lines = print_status_lines

        self._pending = OrderedDict()  # seq -> (send_ts, vx, vy, tx, ty)
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._buf = bytearray()
        self._out_queue = queue.Queue()

        # counters, read from the main thread for the console line
        self.rows = 0
        self.parse_errors = 0
        self.unmatched = 0
        self.touch_lost = 0
        self.write_errors = 0
        self._last_err_mm = None

        self._fh = None
        self._writer = None
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self._fh = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
            self._writer.writeheader()
            self._fh.flush()

    # -- main-thread API ----------------------------------------------------

    def send_frame(self, seq, vision_x, vision_y, target_x, target_y, raw_x=None, raw_y=None):
        """Enqueue a "V,<seq>,..." line for the worker thread to write, and
        register it so the eventual "T,..." echo can be joined back to it.
        Replaces calling ser.write() directly from the main thread -- this
        object's worker thread is now the sole owner of the serial handle.
        Returns the send timestamp (perf_counter), matching the old
        register_frame()'s send_ts parameter, in case a caller wants it.

        raw_x/raw_y (optional): the CNN's estimate BEFORE the gate/dead-band/
        MLP stack, if the caller has it -- vision_x/vision_y is what actually
        gets SENT (the fully processed output), which is the wrong signal to
        characterize measurement noise from for anything meant to replace
        that processing stack. Pass None (default) if not available; the CSV
        row's raw_vision_*/raw_err_* columns are left blank in that case."""
        send_ts = time.perf_counter()
        payload = f"V,{seq},{vision_x:.2f},{vision_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode("ascii")
        with self._pending_lock:
            self._pending[seq] = (
                send_ts,
                float(vision_x),
                float(vision_y),
                float(target_x),
                float(target_y),
                float(raw_x) if raw_x is not None else None,
                float(raw_y) if raw_y is not None else None,
            )
            while len(self._pending) > PENDING_MAX:
                self._pending.popitem(last=False)
        self._out_queue.put(payload)
        return send_ts

    def send_raw(self, payload: bytes):
        """Enqueue a raw payload (e.g. b"L\\n") with no frame registration --
        for the "no ball this frame" markers, which have no vision position to
        pair against a future echo."""
        self._out_queue.put(payload)

    @property
    def last_err_mm(self):
        return self._last_err_mm

    def start(self):
        if self.ser is None:
            print("[touch-log] no serial port; telemetry I/O disabled.")
            return
        self._thread = threading.Thread(
            target=self._run, name="serial-io", daemon=True
        )
        self._thread.start()
        if self.csv_path:
            print(f"[touch-log] logging touchscreen ground truth to {self.csv_path}")
        else:
            print("[touch-log] serial I/O thread active (CSV logging off)")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
        print(
            f"[touch-log] {self.rows} rows | {self.unmatched} unmatched seq | "
            f"{self.touch_lost} no-contact | {self.parse_errors} bad lines | "
            f"{self.write_errors} write errors"
        )

    # -- worker thread: writes THEN reads, every iteration ------------------

    def _run(self):
        while not self._stop.is_set():
            # Drain and write all currently-queued outgoing messages first, so
            # a pending write is never delayed behind more than one read
            # attempt below.
            while True:
                try:
                    payload = self._out_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.ser.write(payload)
                except Exception as e:
                    self.write_errors += 1
                    if not self._stop.is_set():
                        print(f"[touch-log] serial write error: {e}")

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
            raw_x = raw_y = None
        else:
            send_ts, vis_x, vis_y, tgt_x, tgt_y, raw_x, raw_y = sent
            rtt_ms = round((recv_ts - send_ts) * 1000.0, 3)

        if valid:
            err_x = vis_x - touch_x
            err_y = vis_y - touch_y
            err = (err_x * err_x + err_y * err_y) ** 0.5
            self._last_err_mm = err
            if raw_x is not None and raw_y is not None:
                raw_err_x = raw_x - touch_x
                raw_err_y = raw_y - touch_y
                raw_err = (raw_err_x * raw_err_x + raw_err_y * raw_err_y) ** 0.5
            else:
                raw_err_x = raw_err_y = raw_err = ""
        else:
            # No ball on the plate -> no ground truth -> no error. Leaving these
            # blank rather than 0 keeps them out of any mean you compute later.
            self.touch_lost += 1
            err_x = err_y = err = ""
            raw_err_x = raw_err_y = raw_err = ""

        if self._writer is None:
            return

        row = {
            "seq": seq,
            "host_send_ts": round(send_ts, 6) if send_ts != "" else "",
            "host_recv_ts": round(recv_ts, 6),
            "rtt_ms": rtt_ms,
            "mcu_ms": mcu_ms,
            "vision_x_mm": round(vis_x, 3),
            "vision_y_mm": round(vis_y, 3),
            "raw_vision_x_mm": round(raw_x, 3) if raw_x is not None else "",
            "raw_vision_y_mm": round(raw_y, 3) if raw_y is not None else "",
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
            "raw_err_x_mm": round(raw_err_x, 3) if raw_err_x != "" else "",
            "raw_err_y_mm": round(raw_err_y, 3) if raw_err_y != "" else "",
            "raw_err_mm": round(raw_err, 3) if raw_err != "" else "",
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
