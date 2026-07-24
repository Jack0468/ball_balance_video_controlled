import cv2
import time
import struct
import numpy as np
import os
import serial
import argparse
import queue
import sounddevice as sd

from ultralytics import YOLO
from ml_vision.core.coordinate_math import HomographyProjector
from src.receivers import USBReceiver, UDPReceiver
from src.mock_receiver import MockReceiver
from src.utils import find_stm32_port
from src.state_machine import TargetStateMachine
from src.openvino_dispatcher import OpenVINOPipeline
from src.audio_utils import numpy_stft

# --- Configuration ---
SERIAL_PORT = "COM3"
SERIAL_BAUD = 2000000 
SAMPLE_RATE = 16_000
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * 1.25)
# ---------------------

def align_speech_to_fixed_length(audio, target_samples=OUTPUT_SEQUENCE_LENGTH):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if peak < 0.03 or rms < 0.003:
        return None

    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return None

    start = max(0, active[0] - int(0.08 * SAMPLE_RATE))
    end = min(len(audio), active[-1] + int(0.12 * SAMPLE_RATE))
    audio = audio[start:end]

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for USB mode")
    parser.add_argument("--port", type=str, default="auto", help="STM32 serial port (e.g. COM3 or 'auto')")
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver instead of USB camera")
    parser.add_argument("--udp_port", type=int, default=5001, help="Port to listen on for UDP video stream")
    parser.add_argument("--head", action="store_true", help="Show video feed with detections for debugging")
    parser.add_argument("--mock", action="store_true", help="Use mock receiver with 0000.jpg")
    args = parser.parse_args()

    # Auto-detect port if 'auto'
    if args.port == "auto":
        detected_port = find_stm32_port()
        args.port = detected_port if detected_port else SERIAL_PORT
        print(f"STM32 Port selected: {args.port}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_ov_dir = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/platform_and_markers_model/weights/best_openvino_model'))
    corrector_xml = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/corrector/best_corrector.xml'))
    audio_xml = os.path.abspath(os.path.join(script_dir, 'ml_audio/models/audio_command_classifier/best_classifier.xml'))

    print(f"Loading OpenVINO YOLO model via ultralytics from: {yolo_ov_dir}")
    yolo_model = YOLO(yolo_ov_dir, task="pose")

    # Initialize OpenVINO Pipeline for Audio and Corrector (we pass dummy for yolo_xml just to satisfy init)
    yolo_xml_dummy = os.path.join(yolo_ov_dir, 'best.xml')
    pipeline = OpenVINOPipeline(yolo_xml_dummy, audio_xml, corrector_xml, device="CPU")
    
    # Initialize Homography Projector
    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    
    state_machine = TargetStateMachine()
    
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None
        
    if args.mock:
        receiver = MockReceiver('ml_vision/data/yolo_raw_dataset/images/0000.jpg')
    else:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480) if args.udp else USBReceiver(camera_id=args.cam_id)
        
    # Audio setup
    audio_chunk_queue = queue.Queue()
    step_samples = int(SAMPLE_RATE * 0.2)
    audio_buffer = np.zeros(OUTPUT_SEQUENCE_LENGTH, dtype=np.float32)
    
    def _audio_callback(indata, frames, time_info, status):
        audio_chunk_queue.put(indata.copy().squeeze())

    audio_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=step_samples, callback=_audio_callback
    )
    audio_stream.start()

    print("Waiting for camera feed...")
    while receiver.get_latest_frame() is None:
        time.sleep(0.1)

    print("Starting OpenVINO Asynchronous Inference Loop...")
    try:
        while True:
            start_t = time.perf_counter()
            
            # --- 1. Audio Dispatch ---
            while not audio_chunk_queue.empty():
                new_chunk = audio_chunk_queue.get_nowait()
                audio_buffer = np.roll(audio_buffer, -len(new_chunk))
                audio_buffer[-len(new_chunk):] = new_chunk
                
                aligned = align_speech_to_fixed_length(audio_buffer)
                if aligned is not None:
                    spec = numpy_stft(aligned)
                    pipeline.dispatch_audio(spec)
                else:
                    # Send dummy so callback fires with low confidence
                    pipeline.audio_history.append(None)
                    if len(pipeline.audio_history) > 3: pipeline.audio_history.pop(0)

            # --- 2. Vision Dispatch ---
            frame = receiver.get_latest_frame()
            display_frame = None
            if args.head and frame is not None:
                display_frame = frame.copy()
                
            if frame is not None:
                # Run YOLO natively using ultralytics OpenVINO backend
                results = yolo_model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
                
                if results and len(results) > 0:
                    res = results[0]
                    if res.boxes is not None:
                        classes = res.boxes.cls.cpu().numpy()
                        boxes = res.boxes.xywh.cpu().numpy()
                        
                        corners = None
                        ball_box = None
                        detected_markers = {}
                        
                        for i, cls in enumerate(classes):
                            c = int(cls)
                            if c == 0: # Platform
                                if res.keypoints is not None and len(res.keypoints.xy) > i:
                                    kpts = res.keypoints.xy[i].cpu().numpy()
                                    if len(kpts) == 4:
                                        corners = kpts
                            elif c == 1: # Ball
                                ball_box = boxes[i]
                            elif c >= 2: # Marker
                                name = yolo_model.names[c].replace('_marker', '')
                                detected_markers[name] = boxes[i]
                                
                        if corners is not None and ball_box is not None:
                            homography_x, homography_y = 0.0, 0.0
                            marker_coords = {}
                            if projector.update_homography(corners):
                                hx, hy = projector.project_point(ball_box[0], ball_box[1])
                                if hx is not None and hy is not None:
                                    homography_x, homography_y = hx, hy
                                    
                                for name, box in detected_markers.items():
                                    mx, my = projector.project_point(box[0], box[1])
                                    if mx is not None and my is not None:
                                        marker_coords[name] = (mx, my)
                                        
                            features = np.array([
                                ball_box[0], ball_box[1], ball_box[2], ball_box[3],
                                corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                                corners[2][0], corners[2][1], corners[3][0], corners[3][1],
                                homography_x, homography_y
                            ], dtype=np.float32)
                            
                            features[0:12:2] /= 640.0
                            features[1:12:2] /= 480.0
                            features[12:] /= 100.0
                            
                            pipeline.dispatch_corrector(features.reshape(1, 14))
                            
                            if display_frame is not None:
                                # Draw platform corners
                                for j in range(4):
                                    p1 = (int(corners[j][0]), int(corners[j][1]))
                                    p2 = (int(corners[(j+1)%4][0]), int(corners[(j+1)%4][1]))
                                    cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
                                    cv2.circle(display_frame, p1, 5, (0, 0, 255), -1)
                                
                                # Draw ball
                                bx, by, bw, bh = ball_box
                                cv2.rectangle(display_frame, (int(bx - bw/2), int(by - bh/2)), 
                                              (int(bx + bw/2), int(by + bh/2)), (0, 165, 255), 2)
                                              
                                # Draw markers
                                for m_name, m_box in detected_markers.items():
                                    mx, my, mw, mh = m_box
                                    cv2.rectangle(display_frame, (int(mx - mw/2), int(my - mh/2)), 
                                                  (int(mx + mw/2), int(my + mh/2)), (255, 0, 255), 2)
                                    cv2.putText(display_frame, m_name, (int(mx - mw/2), int(my - mh/2) - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                                                
                        else:
                            kpts_shapes = [res.keypoints.xy[j].cpu().numpy().shape if res.keypoints is not None and len(res.keypoints.xy) > j else None for j in range(len(classes))]
                            print(f"[DEBUG] Missing! classes: {classes}. corners is None? {corners is None}. ball_box is None? {ball_box is None}")

            # If corrector finished, send it to serial
            cam_coords = pipeline.state.get("corrector_output")
            if cam_coords is not None:
                pipeline.state["corrector_output"] = None
                cam_x, cam_y = cam_coords
                
                cmd = pipeline.get_and_clear_audio_command()
                if cmd is not None:
                    print(f"\n======================================")
                    print(f"🗣️ DETECTED AUDIO COMMAND: '{cmd.upper()}'")
                    print(f"======================================\n")
                    
                state_machine.process_command(cmd, cam_x, cam_y)
                
                target_x, target_y = state_machine.get_target_coords({}) # Pass markers
                
                try:
                    payload = f"{cam_x:.2f},{cam_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode('ascii')
                    if ser:
                        ser.write(payload)
                except Exception as e:
                    pass
                
                end_t = time.perf_counter()
                fps = 1.0 / (end_t - start_t)
                print(f"Targeting '{state_machine.current_target_name}' at X={target_x:.1f} Y={target_y:.1f} | Ball: X={cam_x:.1f} Y={cam_y:.1f} mm | FPS: {fps:.1f}")
                
            if display_frame is not None:
                cv2.imshow("VRI 2026 YOLO Core", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        audio_stream.stop()
        audio_stream.close()
        receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")

if __name__ == '__main__':
    main()
