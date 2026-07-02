for 25/06/2026
- Determine the kind stepper motor / driver required for the FPGA implementation 
- Completed: Fixed the vision preprocessor pipeline. Transitioned from Canny edge detection to HSV Color Masking (specifically tuning for the White/Grey platform) and contour detection directly on the binary mask. 
- Completed: Added robust shape-filtering heuristics (4-sided, convex, interior color uniformity variance, frame-border touch exclusions).
- Completed: Added a 5-second camera stabilization startup buffer (later removed for faster iterations).
- Completed: Fixed ZeroDivisionError when running in headless/no-YOLO modes.
- Completed: Reverted slow CLAHE algorithm and instead implemented an ultra-fast Edge Overlap Verification heuristic. The pipeline now verifies candidates by performing a bitwise intersection against a 1-pixel thin morphological gradient edge-map of the HSV mask.

for 1/07/26

work on the polling infractucture of the camera via verilog fpga.

- [ ] RETUNE PID: The Python Host-PC control loop running over USB will have a completely different latency profile than the original bare-metal Teensy C++ loop. The constants (kp=0.8, ki=0.2, kd=0.09) will cause oscillations and need aggressive retuning.
