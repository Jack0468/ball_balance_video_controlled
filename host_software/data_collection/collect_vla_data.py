import cv2
import time
import struct
import os
import csv
import argparse
import threading
import serial
from datetime import datetime
from src.utils import find_stm32_port


class VLADataCollector:
    def __init__(self, port, baudrate=2000000, cam_id=0):
        self.port = port
        self.baudrate = baudrate
        self.cam_id = cam_id

        # Base bronze directory
        self.bronze_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "01_bronze"
        )
        os.makedirs(self.bronze_dir, exist_ok=True)

        self.is_recording = False
        self.session_dir = None
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_index = 0

        self.latest_telemetry = None
        self.telemetry_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.telemetry_thread = threading.Thread(target=self.read_serial_telemetry)
        self.telemetry_thread.daemon = True

    def start_recording(self):
        if self.is_recording:
            return

        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = os.path.join(self.bronze_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        video_path = os.path.join(self.session_dir, "rgb_video.mp4")
        # Define codec for mp4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Assuming webcam is 640x480 at 30 fps
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        csv_path = os.path.join(self.session_dir, "telemetry.csv")
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "frame_index",
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

        self.frame_index = 0
        self.is_recording = True
        print(f"\n[RECORDING STARTED] Saving to {self.session_dir}\n")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

        print(
            f"\n[RECORDING STOPPED] Saved {self.frame_index} frames to {self.session_dir}\n"
        )

    def read_serial_telemetry(self):
        print(f"Connecting to STM32 on {self.port} at {self.baudrate} baud...")
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        except Exception as e:
            print(f"Error opening serial port: {e}")
            return

        # Format: mcu_micros(I) + 15 floats(f) = 64 bytes total (excluding sync header)
        struct_format = "<Ifffffffffffffff"
        expected_size = struct.calcsize(struct_format)

        sync_buf = bytearray()
        while not self.stop_event.is_set():
            if ser.in_waiting > 0:
                b = ser.read(1)
                sync_buf.append(b[0])
                if len(sync_buf) > 4:
                    sync_buf.pop(0)

                # Match sync header 0xAABBCCDD
                if bytes(sync_buf) == b"\xaa\xbb\xcc\xdd":
                    data = ser.read(expected_size)
                    if len(data) == expected_size:
                        host_time_ms = int(time.time() * 1000)
                        unpacked = struct.unpack(struct_format, data)

                        with self.telemetry_lock:
                            self.latest_telemetry = (host_time_ms,) + unpacked
                    sync_buf.clear()

    def run(self):
        self.telemetry_thread.start()

        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            print(f"Failed to open webcam ID {self.cam_id}")
            return

        # Set webcam resolution to 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting VLA Data Collection.")
        print("Press 'r' to start recording.")
        print("Press 's' to stop recording.")
        print("Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from webcam. Exiting...")
                    break

                # Grab the most recent telemetry
                with self.telemetry_lock:
                    current_tel = self.latest_telemetry

                # Display status on the UI frame
                display_frame = frame.copy()
                if self.is_recording:
                    cv2.putText(
                        display_frame,
                        f"RECORDING (Frame: {self.frame_index})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

                    if current_tel:
                        # Write frame and telemetry
                        self.video_writer.write(frame)
                        self.csv_writer.writerow([self.frame_index] + list(current_tel))
                        self.frame_index += 1
                else:
                    cv2.putText(
                        display_frame,
                        "READY (Press 'r' to record)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                if current_tel:
                    target_x, target_y = current_tel[2], current_tel[3]
                    touch_x, touch_y = current_tel[4], current_tel[5]
                    cv2.putText(
                        display_frame,
                        f"Ball: {touch_x:.1f}, {touch_y:.1f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Tgt: {target_x:.1f}, {target_y:.1f}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 150, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "WAITING FOR STM32 TELEMETRY...",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("VLA Data Collection", display_frame)

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
            cap.release()
            cv2.destroyAllWindows()
            print("VLA Data Collection stopped.")


if __name__ == "__main__":
    # Ensure src is in the path for imports
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(
        description="Collect Multimodal VLA demonstrations."
    )
    parser.add_argument(
        "--port", type=str, default="auto", help="STM32 COM port (e.g. COM8 or auto)"
    )
    parser.add_argument("--baud", type=int, default=2000000, help="Baud rate")
    parser.add_argument("--cam", type=int, default=0, help="Webcam ID")
    args = parser.parse_args()

    port = args.port
    if port == "auto":
        port = find_stm32_port()
        if not port:
            port = "COM3"
            print(f"Could not auto-detect STM32. Defaulting to {port}")

    collector = VLADataCollector(port, args.baud, args.cam)
    collector.run()
