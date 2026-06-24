# Machine Learning Vision Pipeline - Tasks & Roadmap

## Phase 1: Tasks We Can Do NOW (Pre-Hardware)

Since we do not have the physical robot built or high-quality dataset images yet, we can work on the following standalone components:

- [x] **Test the Pre-Trained YOLO Pipeline**: Use our current "dodgy images" or a live webcam feed of a random ball to test `extract_frames.py` and benchmark the latency of the pre-trained YOLO model.
- [x] **Refine the Canny Preprocessor**: Test `preprocessor.py` by feeding it images of any square/rectangular object (like a piece of paper or a book) to ensure the contour approximation and perspective warp perfectly flattens the image.
- [x] **FrontPanel API Skeleton**: Write a dummy script that imports the Opal Kelly API and sends fake `(x, y)` coordinates to simulate the USB data transfer.
- [x] **C++ to HLS Refactoring**: Begin converting the old Teensy C++ Inverse Kinematics and PID code to use `ap_fixed` fixed-point arithmetic, preparing it for Vitis HLS synthesis.
- [x] **Coordinate Math**: Write the Python logic to cleanly translate the 2D pixel coordinates into real-world millimeter offsets from the center.

---

## Phase 2: Tasks We Will Need To Do LATER (Hardware Integration)

Once the robot is built and the Opal Kelly FPGA is mounted:

- [ ] **Physical Calibration**: Mount the camera precisely overhead and measure the exact millimeter dimensions of the physical platform to lock in our pixel-to-millimeter scaling ratio.
- [ ] **Lighting & Threshold Tuning**: Fine-tune the `canny_threshold1` and `canny_threshold2` values in `preprocessor.py` based on the actual lighting conditions of the room where the robot is permanently installed.
- [ ] **Custom Model Fine-Tuning**: Collect a high-quality dataset of the specific ball rolling on the final platform. Label the data and use it to fine-tune our YOLO/MobileNet model to prevent false positives.
- [ ] **USB Latency Testing**: Hook up the Host PC to the Opal Kelly FPGA and measure the true end-to-end latency of passing the vision coordinates to the hardware.
- [ ] **Audio Integration**: Link up with the Audio team to map their predicted state (e.g., "go red") to a hardcoded physical `(x, y)` target coordinate for the ball to balance on.
