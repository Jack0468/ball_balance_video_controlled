for 25/06/2026
- Determine the kind stepper motor / driver required for the FPGA implementation 
- Completed: Fixed the vision preprocessor pipeline. Transitioned from Canny edge detection to HSV Color Masking (specifically tuning for the White/Grey platform) and contour detection directly on the binary mask. 
- Completed: Added robust shape-filtering heuristics (4-sided, convex, interior color uniformity variance, frame-border touch exclusions).
- Completed: Added a 5-second camera stabilization startup buffer (later removed for faster iterations).
- Completed: Fixed ZeroDivisionError when running in headless/no-YOLO modes.
- Completed: Reverted slow CLAHE algorithm and instead implemented an ultra-fast Edge Overlap Verification heuristic. The pipeline now verifies candidates by performing a bitwise intersection against a 1-pixel thin morphological gradient edge-map of the HSV mask.

for 1/07/26

work on the polling infractucture of the camera via verilog fpga.

- [x] Create a robust Expert Position Classification Model (ResNet18) to predict `(x, y)` purely from image frames
  - [x] Define a PyTorch Dataset loading from `data/02_silver`
  - [x] Script to train `models.resnet18` targeting normalized `(x, y)` output
  - [x] Option A: Add a subset test evaluator on last 20%
- [x] Resolve YOLO marker detection glare issues
  - [x] Build synthetic dataset generator (`data/03_synthetic_yolo`) using real `02_silver` images with HSV red ball masking
  - [x] Synthesize translucent fake markers to teach YOLO marker localization through extreme glare
- [x] Improve Colab training stability
  - [x] Split monolithic notebook into `colab_train_yolo.ipynb` and `colab_train_expert.ipynb`
- [ ] RETUNE PID: The Python Host-PC control loop running over USB will have a completely different latency profile than the original bare-metal Teensy C++ loop. The constants (kp=0.8, ki=0.2, kd=0.09) will cause oscillations and need aggressive retuning.

13/07/2026
DO NOT BEGIN THIS WORK UNTIL INSTRUCTED.
Here is the straight-to-implementation blueprint for the RRS ball tracking pipeline. This minimizes latency and keeps the real-time embedded logic separate from the heavy Python processing.

Phase 1: iPhone to Laptop (Vision Pipeline)
Protocol: UDP over USB-Tethered Network
Framerate Target: 30 FPS (~33ms per frame)

iPhone (Pyto Script):

Initialize cv2.VideoCapture() and standard Python socket (UDP/IPv4).

Inside the main loop:

Read the frame.

Downscale if necessary (e.g., 640x480 is plenty for ball tracking and saves bandwidth).

Compress to JPEG: _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]).

Send the byte array directly to the laptop's tethered IP address via sock.sendto().

Laptop (Receiver Thread):

Open a UDP socket bound to the tethered IP/Port.

Receive the byte array (sock.recvfrom()).

Decompress instantly: frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).

Push this frame into a thread-safe queue so your PyTorch loop always has the most recent frame (drop older frames if the ML is running slower than 30 FPS).

Phase 2: Laptop ML & Error Calculation
Environment: Python + PyTorch + PySerial

Inference: Pull the latest frame from the queue and run it through your PyTorch model to get the ball's bounding box (x, y, w, h).

Error Calculation: Calculate the pixel difference between the center of the ball and the center of the camera frame (this is your setpoint error).

Example: error_x = target_center_x - frame_center_x

Data Serialization: Do not send raw strings (like "X:120, Y:-40\n") over serial; string parsing on the STM32 wastes clock cycles. Pack the error values into a fixed byte structure using Python's struct library.

Format: [Start Byte] [Error X (16-bit int)] [Error Y (16-bit int)] [End Byte]

Example: payload = struct.pack('<chh', b'<', error_x, error_y)

Transmit: Blast that packed byte array out via pyserial at a high baud rate (e.g., 115200 or 921600).

Phase 3: STM32 Actuator Control (The Real-Time Node)
Environment: Embedded C/C++

UART Reception: Set up a DMA (Direct Memory Access) or Interrupt-driven UART receive buffer. It looks for the start byte (<), reads the next 4 bytes (the two 16-bit integers), and updates a global current_error struct.

The Hardware PID Loop:

Run your PID math strictly on the STM32 using a hardware timer interrupt (e.g., firing exactly every 10ms or 20ms).

The PID algorithm reads the latest current_error from the UART buffer, calculates the Proportional, Integral, and Derivative terms, and computes the new motor output.

Actuation:

The output of the PID loop directly updates the CCR (Capture/Compare Register) values for the PWM timers driving your motor controllers.

Failsafe: Add a timeout. If the STM32 hasn't received a valid serial packet from the laptop in >100ms (meaning the ML crashed or the cable disconnected), immediately set PWM to neutral to stop the robot from spinning out of control.

14/07/2026
- [ ] **DATASET FAILURE**: Discovered that the 02_silver dataset has ±10mm camera shifts across the 3 collection runs. This completely invalidates the static pixel-to-millimeter mapping needed by the ResNet tracker.
- [ ] **PIVOT**: The YOLO pivot is the correct path forward to compute dynamic homography and ignore camera bumps. However, glare in the dataset prevents manual ground-truth labeling.
- [ ] **DECISION**: We will implement a Teacher-Student VLM pipeline (Option A) using Qwen-VL or similar to auto-label the dataset offline. 
- [ ] **IMMEDIATE FOCUS**: Do not modify YOLO pipeline code right now. Focus entirely on the expert model (ResNet18) for position detection first to establish a baseline and evaluate its performance.

15/07/2026

refactor the audio section to align with our project standards.
write the script to integrate the audio model with the video inference in order to control the ball
