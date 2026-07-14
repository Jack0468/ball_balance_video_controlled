# iPhone Camera Setup Guide (UDP over USB)

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
