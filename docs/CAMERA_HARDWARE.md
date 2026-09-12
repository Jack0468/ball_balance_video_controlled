# Camera Hardware — Jetson USB Webcam

## Finding (2026-09-09): `receivers.py`'s `USBReceiver` fails to open on this camera at its default settings

**Root cause**: `USBReceiver.__init__` hardcodes `cap.set(cv2.CAP_PROP_FPS, 60)`. This
camera's real capabilities (raw `v4l2-ctl --list-formats-ext` dump below) show **MJPG at
640x480 supports only one discrete interval: 30fps** — 60fps at that resolution doesn't
exist as a mode at all. On this Jetson, OpenCV's backend fallback in `USBReceiver`
(`cv2.CAP_DSHOW` fails on Linux as expected, falls through to `cv2.VideoCapture(camera_id)`
with no explicit backend) resolves to **GStreamer**, whose autonegotiation fails outright
on an unsupported mode rather than clamping to the nearest valid one — surfaced as
`GStreamer warning: Embedded video playback halted; module v4l2src0 reported: Internal
data stream error` / `unable to start pipeline`, not an informative "unsupported fps"
message.

**Confirmed fix**: forcing `cv2.CAP_V4L2` explicitly (bypassing GStreamer) and requesting
`30` fps instead of `60` at `640x480` opens the camera and reads real frames successfully.
Not yet confirmed whether the V4L2-backend-forcing part is strictly necessary or whether
30fps alone would also succeed via the GStreamer default path (30fps is a genuinely valid
mode per the dump below) — worth testing in isolation if a future change wants to avoid
forcing a backend unnecessarily.

**Not yet applied to `receivers.py` itself** — that's shared project code
(`host_software/src/`), so the actual fix (making `fps` a constructor parameter rather
than a hardcoded `60`, defaulting to `60` for existing Windows-laptop callers whose
cameras may genuinely support it, with `run_jetson_standalone.py` passing `30` explicitly
for this camera) is proposed, not yet made.

## Raw capability dump (`v4l2-ctl -d /dev/video0 --list-formats-ext`)

```text
ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture

        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 1280x720
                        Interval: Discrete 0.033s (30.000 fps)
                        Interval: Discrete 0.040s (25.000 fps)
                        Interval: Discrete 0.050s (20.000 fps)
                        Interval: Discrete 0.067s (15.000 fps)
                        Interval: Discrete 0.100s (10.000 fps)
                        Interval: Discrete 0.133s (7.500 fps)
                        Interval: Discrete 0.200s (5.000 fps)
                Size: Discrete 960x720
                        Interval: Discrete 0.033s (30.000 fps)
                        Interval: Discrete 0.040s (25.000 fps)
                        Interval: Discrete 0.050s (20.000 fps)
                        Interval: Discrete 0.067s (15.000 fps)
                        Interval: Discrete 0.100s (10.000 fps)
                        Interval: Discrete 0.133s (7.500 fps)
                        Interval: Discrete 0.200s (5.000 fps)
                Size: Discrete 640x480
                        Interval: Discrete 0.033s (30.000 fps)
                Size: Discrete 800x600
                        Interval: Discrete 0.033s (30.000 fps)
        [1]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 1280x720
                        Interval: Discrete 0.100s (10.000 fps)
                Size: Discrete 960x720
                        Interval: Discrete 0.100s (10.000 fps)
                Size: Discrete 640x480
                        Interval: Discrete 0.033s (30.000 fps)
                Size: Discrete 800x600
                        Interval: Discrete 0.033s (30.000 fps)
```
