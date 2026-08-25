"""Jetson-side consumer for the STM32 telemetry proposal
(`../stm32_interface/Telemetry.cpp`, `../stm32_interface/TELEMETRY_PROTOCOL.md`).

**Only usable once the firmware owner has reviewed and merged that proposal** --
`stm32_interface/` is staged code, not live firmware, since `firmware/` is outside this
agent's write boundary. Until then this module has nothing to parse.

Parses the `0xAABBCCDD`-prefixed binary packet described in TELEMETRY_PROTOCOL.md and
writes rows matching `host_software/evaluations/evaluate_system_control.py`'s
`REQUIRED_COLUMNS` schema exactly, so a Jetson-hosted run's CSV drops into
`evaluate_system_control.py --runs label=csv ...` with zero changes to that tool.
"""

from __future__ import annotations

import csv
import struct
import time
from pathlib import Path
from typing import BinaryIO, Optional

SYNC_HEADER = b"\xaa\xbb\xcc\xdd"

# Must match TelemetryPacket's layout in stm32_interface/Telemetry.cpp exactly (the
# struct's sync_header field itself is consumed by the byte-at-a-time search below, not
# part of this format string): mcu_micros(u32), target_x/y_mm(f32), touch_x/y_mm(f32),
# actual_step_a/b/c(i32), theta_a/b/c_deg(f32). Little-endian (STM32 is Cortex-M).
_STRUCT_FORMAT = "<Iffffiiifff"
_PACKET_SIZE = struct.calcsize(_STRUCT_FORMAT)

# Column order matches evaluate_system_control.py's REQUIRED_COLUMNS + its preferred
# timestamp candidate (host_timestamp_ms, first in TIMESTAMP_CANDIDATES). mcu_micros_ts
# and actual_step_a/b/c are extra columns -- harmless, evaluate_system_control.py only
# checks REQUIRED_COLUMNS are present, it doesn't reject additional ones.
CSV_COLUMNS = [
    "host_timestamp_ms",
    "target_x",
    "target_y",
    "touch_x",
    "touch_y",
    "theta_a",
    "theta_b",
    "theta_c",
    "mcu_micros_ts",
    "actual_step_a",
    "actual_step_b",
    "actual_step_c",
]


def parse_stream(port: BinaryIO):
    """Generator: yields one dict (CSV_COLUMNS-shaped) per valid packet read from an
    open, blocking-or-nonblocking-doesn't-matter byte source with a `.read(n)` method
    (a `serial.Serial` instance satisfies this). Byte-at-a-time sync-header search,
    matching the firmware's own line-based parsers' "drop until a clean boundary"
    philosophy (SerialCoords.cpp/RemoteStepControl.cpp) rather than assuming the stream
    is already packet-aligned."""
    sync_buf = bytearray()
    while True:
        b = port.read(1)
        if not b:
            continue  # non-blocking source with nothing waiting -- caller controls pacing
        sync_buf += b
        if len(sync_buf) > 4:
            sync_buf.pop(0)
        if bytes(sync_buf) != SYNC_HEADER:
            continue

        data = port.read(_PACKET_SIZE)
        if len(data) != _PACKET_SIZE:
            sync_buf.clear()
            continue  # truncated read (port closed / stalled) -- resync on next header

        host_timestamp_ms = int(time.time() * 1000)
        (
            mcu_micros_ts,
            target_x,
            target_y,
            touch_x,
            touch_y,
            actual_step_a,
            actual_step_b,
            actual_step_c,
            theta_a,
            theta_b,
            theta_c,
        ) = struct.unpack(_STRUCT_FORMAT, data)

        yield {
            "host_timestamp_ms": host_timestamp_ms,
            "target_x": target_x,
            "target_y": target_y,
            "touch_x": touch_x,
            "touch_y": touch_y,
            "theta_a": theta_a,
            "theta_b": theta_b,
            "theta_c": theta_c,
            "mcu_micros_ts": mcu_micros_ts,
            "actual_step_a": actual_step_a,
            "actual_step_b": actual_step_b,
            "actual_step_c": actual_step_c,
        }
        sync_buf.clear()


class TelemetryLogger:
    """Wraps `parse_stream()` with a CSV writer. One row written per packet consumed
    via `poll_and_log()` -- call it from a runtime loop (e.g. `run_jetson_standalone.py`)
    once telemetry is available; it does not spawn its own thread, matching this repo's
    convention of keeping receivers/loggers simple and caller-driven rather than
    background magic (see `ENGINEERING_STANDARDS.md`'s non-blocking-execution rule --
    this only blocks on `port.read()`, which the caller controls via the port's own
    timeout)."""

    def __init__(self, port: BinaryIO, csv_path: Path) -> None:
        self._gen = parse_stream(port)
        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

    def poll_and_log(self) -> Optional[dict]:
        """Consume exactly one packet (or block per the port's own timeout settings --
        use a short `serial.Serial(..., timeout=...)` to keep this non-blocking-ish in a
        larger loop). Returns the row dict written, or None if the underlying port has
        nothing more to give right now."""
        row = next(self._gen, None)
        if row is not None:
            self._writer.writerow(row)
            self._csv_file.flush()
        return row

    def close(self) -> None:
        self._csv_file.close()


if __name__ == "__main__":
    import argparse

    import serial

    parser = argparse.ArgumentParser(description="Standalone STM32 telemetry logger (Track 2 proposal)")
    parser.add_argument("--port", required=True, help="Serial port the STM32 telemetry is on, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=2000000)
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    logger = TelemetryLogger(ser, Path(args.out))
    print(f"Logging telemetry from {args.port} to {args.out} (Ctrl+C to stop)...")
    try:
        while True:
            row = logger.poll_and_log()
            if row is None:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        ser.close()
        print("Stopped.")
