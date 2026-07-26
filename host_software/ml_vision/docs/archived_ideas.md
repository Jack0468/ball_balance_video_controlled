# Archived Ideas: ML Vision

This document tracks experimental architectural approaches, algorithms, and ideas that were explored but ultimately set aside in favor of a different approach. We keep this history to prevent redundant work in the future and to document engineering trade-offs.

---

## 1. Pure Unsupervised Clustering for Marker Detection (13/07/2026)

**The Goal:**
Identify the four physical control markers (Red, Green, Grey, Black) on the VRI platform using a purely unsupervised approach, demonstrating "machine learning intelligence" without the need for manual data labeling.

**The Approach:**
We developed `test_marker_clustering.py` to evaluate K-Means spatial-color clustering.
To minimize the massive CPU overhead of running K-Means on hundreds of thousands of pixels, we attempted to isolate the markers using aggressive HSV thresholding:

- **Color Mask:** `Saturation > 50` & `Value > 50` (Isolates Red & Green)
- **Black Mask:** `Value < 40` (Isolates Black marker)
- **Grey Mask:** `Value BETWEEN (70, 150)` & `Saturation < 30` (Isolates Grey marker)

Pixels that passed this mask were mapped into a 5-dimensional feature space `[X, Y, H, S, V]`, normalized, and fed into K-Means (`K=6`).

**The Results & Why it was Archived:**

- **The Grey/Black Thresholding Flaw:** Because the Grey and Black markers have near-zero color saturation, we had to introduce the `Value` (brightness) thresholds. However, this caused the mask to accidentally capture virtually every shadow cast on the platform, the laptop screen bezels, and the dark background floor.
- **Latency Collapse:** The mask preserved over **92,000 pixels** (almost 30% of the entire 640x480 frame). Running K-Means on an array of 92,000 pixels in Python `scikit-learn` took **4.93 seconds** per frame (0.2 FPS).
- **Conclusion:** Purely unsupervised clustering in HSV space lacks the *spatial context* necessary to distinguish a grey marker from a grey shadow. It requires too much raw compute to be viable for our strict 30 FPS, real-time control target on a CPU edge device.

**Pivot:**
We archived this script and pivoted towards exploring Transfer Learning approaches (like a Multi-Task ResNet head) where the neural network intrinsically learns the spatial context to ignore background shadows while remaining computationally efficient.

## 2.  iPhone Camera Setup Guide (UDP over USB)

To achieve low-latency, high-bandwidth camera streaming from the iPhone to the host PC for live ML inference, we utilize a **UDP stream over a direct USB network connection**.

Standard webcam drivers (like UVC or Camo) often throttle bandwidth or introduce processing latency. By using an iOS Python interpreter to encode and blast raw UDP packets directly to the PC over a tethered USB network connection, we can bypass these limitations.

## 1. Establish the USB Network Connection

This step creates a hardwired Ethernet connection between the iPhone and the Windows PC over the USB cable.

1. **Disable Wi-Fi** on your Windows PC temporarily to ensure the UDP traffic routes through the USB connection and not your local router.
2. **Enable Personal Hotspot** on your iPhone (Settings > Personal Hotspot).
3. **Plug the iPhone into the PC** via the USB cable.
4. Windows will automatically detect the iPhone as a new wired Network Adapter (NDIS).
5. Open a command prompt (`cmd`) on Windows and type `ipconfig`.
6. Look for the Ethernet adapter assigned to the Apple Mobile Device. Note down your PC's IP address on this network (it is usually `172.20.10.2` or `192.168.137.1`).

## 2. Setup the iPhone Streaming Script

We use the Pyto app to run native Python scripts directly on iOS to access the camera and network sockets.

1. **Download Pyto**: Install the "Pyto" app from the iOS App Store.
2. **Transfer the Script**: AirDrop, email, or copy the `host_software/ml_vision/iphone_camera_stream.py` script onto your iPhone and open it inside Pyto.
3. **Configure the IP**: Open the script in Pyto and change the `LAPTOP_IP` variable on line 8 to match the IP address you found in Step 1 (e.g., `'172.20.10.2'`).

## 3. Run the Live Inference Loop

Once the network is established and the script is configured, you can start the system.

1. **Start the PC Receiver**: Run the host inference loop in your VS Code terminal on the Windows PC:

   ```bash
   conda run -n ball_balance_env python host_software/ml_vision/host_inference_loop.py --camera udp
   ```

2. **Start the iPhone Stream**: Press the "Play" button in Pyto on your iPhone.

The iPhone will immediately begin capturing frames via OpenCV, compressing them to JPEG, and blasting them to port 5005. The PC will instantly catch them and display the live inference window!
