import os
import sys
import time
import csv
import cv2
import struct
import argparse
import threading
import serial
from datetime import datetime

# Ensure host_software/src is importable when run from this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import find_stm32_port


class WebcamDataCollector:
    def __init__(
        self, port, baudrate=2000000, cam_id=0, width=640, height=480, fps=30.0
    ):
        self.port = port
        self.baudrate = baudrate
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps

        self.bronze_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "01_bronze")
        )
        os.makedirs(self.bronze_dir, exist_ok=True)

        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = os.path.join(self.bronze_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        self.video_path = os.path.join(self.session_dir, "rgb_video.mp4")
        self.telemetry_path = os.path.join(self.session_dir, "telemetry.csv")
        self.frame_timestamps_path = os.path.join(
            self.session_dir, "frame_timestamps.csv"
        )

        self.telemetry_writer = None
        self.telemetry_file = None
        self.frame_timestamps_writer = None
        self.frame_timestamps_file = None

        self.latest_telemetry = None
        self.telemetry_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.telemetry_thread = threading.Thread(target=self._read_serial_telemetry)
        self.telemetry_thread.daemon = True

        self.is_recording = False
        self.video_writer = None
        self.recorded_frame_index = 0

        self.sync_header = b"\xaa\xbb\xcc\xdd"
        self.struct_format = "<Ifffffffffffffff"
        self.expected_size = struct.calcsize(self.struct_format)

    def _open_telemetry_file(self):
        self.telemetry_file = open(self.telemetry_path, mode="a", newline="")
        self.telemetry_writer = csv.writer(self.telemetry_file)
        if os.path.getsize(self.telemetry_path) == 0:
            self.telemetry_writer.writerow(
                [
                    "host_timestamp_ms",
                    "mcu_micros",
                    "target_x",
                    "target_y",
                    "touch_x",
                    "touch_y",
                    "error_x",
                    "error_y",
                    "pitch",
                    "roll",
                    "theta_a",
                    "theta_b",
                    "theta_c",
                    "integral_x",
                    "integral_y",
                    "deriv_x",
                    "deriv_y",
                ]
            )
            self.telemetry_file.flush()

    def _open_frame_timestamps_file(self):
        self.frame_timestamps_file = open(
            self.frame_timestamps_path, mode="a", newline=""
        )
        self.frame_timestamps_writer = csv.writer(self.frame_timestamps_file)
        if os.path.getsize(self.frame_timestamps_path) == 0:
            self.frame_timestamps_writer.writerow(["frame_index", "frame_timestamp_ms"])
            self.frame_timestamps_file.flush()

    def _start_video_writer(self):
        if self.video_writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.video_path, fourcc, self.fps, (self.width, self.height)
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.video_path}")

    def _read_serial_telemetry(self):
        print(f"Connecting to STM32 on {self.port} at {self.baudrate} baud...")
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
        except Exception as e:
            print(f"Error opening serial port: {e}")
            return

        self._open_telemetry_file()
        sync_buf = bytearray()

        while not self.stop_event.is_set():
            try:
                if ser.in_waiting > 0:
                    b = ser.read(1)
                    if not b:
                        continue
                    sync_buf.append(b[0])
                    if len(sync_buf) > 4:
                        sync_buf.pop(0)

                    if bytes(sync_buf) == self.sync_header:
                        data = ser.read(self.expected_size)
                        if len(data) == self.expected_size:
                            host_time_ms = int(time.time() * 1000)
                            unpacked = struct.unpack(self.struct_format, data)
                            row = (host_time_ms,) + unpacked
                            with self.telemetry_lock:
                                self.latest_telemetry = row
                            self.telemetry_writer.writerow(row)
                            self.telemetry_file.flush()
                        sync_buf.clear()
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Telemetry read error: {e}")
                time.sleep(0.1)

    def start_recording(self):
        if self.is_recording:
            return
        self._start_video_writer()
        self._open_frame_timestamps_file()
        self.is_recording = True
        print(f"\n[RECORDING STARTED] Saving video to {self.video_path}")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.frame_timestamps_file:
            self.frame_timestamps_file.close()
            self.frame_timestamps_file = None
            self.frame_timestamps_writer = None
        print(f"\n[RECORDING STOPPED] Captured {self.recorded_frame_index} frames")

    def run(self):
        self.telemetry_thread.start()

        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            print(f"Failed to open webcam ID {self.cam_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        print("Starting webcam data collection.")
        print("Press r to start recording, s to stop, q to quit.")
        print(f"Session directory: {self.session_dir}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from webcam. Exiting...")
                    break

                timestamp_ms = int(time.time() * 1000)
                display_frame = frame.copy()

                if self.is_recording:
                    self.video_writer.write(frame)
                    self.frame_timestamps_writer.writerow(
                        [self.recorded_frame_index, timestamp_ms]
                    )
                    self.frame_timestamps_file.flush()
                    self.recorded_frame_index += 1
                    cv2.putText(
                        display_frame,
                        f"RECORDING ({self.recorded_frame_index})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "READY (press r to record)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

                with self.telemetry_lock:
                    current_tel = self.latest_telemetry

                if current_tel:
                    cv2.putText(
                        display_frame,
                        f"TX: {current_tel[2]:.1f} TY: {current_tel[3]:.1f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Ball: {current_tel[4]:.1f}, {current_tel[5]:.1f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "WAITING FOR TELEMETRY...",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("Webcam Data Collection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.start_recording()
                elif key == ord("s"):
                    self.stop_recording()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_recording()
            self.stop_event.set()
            if self.telemetry_thread.is_alive():
                self.telemetry_thread.join(timeout=1.0)
            if self.telemetry_file:
                self.telemetry_file.close()
            cap.release()
            cv2.destroyAllWindows()
            print("Data collection stopped.")
            print(
                f"Output files: {self.video_path}, {self.telemetry_path}, {self.frame_timestamps_path}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Collect webcam video and STM32 telemetry for offline sync."
    )
    parser.add_argument(
        "--port", type=str, default="auto", help="STM32 COM port (e.g. COM3 or auto)"
    )
    parser.add_argument("--baud", type=int, default=2000000, help="Baud rate")
    parser.add_argument("--cam", type=int, default=0, help="Webcam ID")
    parser.add_argument("--width", type=int, default=640, help="Webcam width")
    parser.add_argument("--height", type=int, default=480, help="Webcam height")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate")
    args = parser.parse_args()

    port = args.port
    if port == "auto":
        port = find_stm32_port()
        if not port:
            port = "COM3"
            print(f"Could not auto-detect STM32. Defaulting to {port}")

    collector = WebcamDataCollector(
        port, args.baud, args.cam, args.width, args.height, args.fps
    )
    collector.run()


if __name__ == "__main__":
    main()
