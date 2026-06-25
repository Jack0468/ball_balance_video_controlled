for 25/06/2026
- Determine the kind stepper motor / driver required for the FPGA implementation 
- Completed: Fixed the vision preprocessor pipeline. Transitioned from Canny edge detection to HSV Color Masking (specifically tuning for the White/Grey platform) and contour detection directly on the binary mask. 
- Completed: Added robust shape-filtering heuristics (4-sided, convex, interior color uniformity variance, frame-border touch exclusions).
- Completed: Added a 5-second camera stabilization startup buffer.
- Completed: Fixed ZeroDivisionError when running in headless/no-YOLO modes.