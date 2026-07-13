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
