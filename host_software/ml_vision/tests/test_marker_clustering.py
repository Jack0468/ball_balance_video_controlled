import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import time

def main():
    # 1. Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(os.path.join(script_dir, '../data/02_silver/images/1783662370840.jpg'))
    output_path = os.path.abspath(os.path.join(script_dir, 'clustering_output.jpg'))
    
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return
        
    start_t = time.perf_counter()
    
    # 2. Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv_image.shape
    
    H, S, V = cv2.split(hsv_image)
    
    # 3. Multi-Mask Preprocessing
    # a. Color Mask (Red & Green) -> High Saturation, High Value
    mask_color = (S > 50) & (V > 50)
    
    # b. Black Mask -> Very low value
    mask_black = (V < 40)
    
    # c. Grey Mask -> Intermediate value, low saturation
    mask_grey = (V > 70) & (V < 150) & (S < 30)
    
    # Combine masks using Bitwise OR
    final_mask = mask_color | mask_black | mask_grey
    
    # 4. Feature Extraction & Normalization
    # Get the (Y, X) coordinates of all pixels that passed the mask
    y_coords, x_coords = np.where(final_mask)
    
    if len(y_coords) == 0:
        print("Error: Mask eliminated all pixels!")
        return
        
    print(f"Mask eliminated {100 * (1 - len(y_coords) / (w * h)):.2f}% of pixels. Kept {len(y_coords)} pixels.")
    
    # Extract H, S, V values for those pixels
    h_vals = H[y_coords, x_coords]
    s_vals = S[y_coords, x_coords]
    v_vals = V[y_coords, x_coords]
    
    # Stack into a feature array: [X, Y, H, S, V]
    # Normalize features to [0, 1] range so spatial distance is comparable to color distance
    features = np.column_stack([
        x_coords / float(w),
        y_coords / float(h),
        h_vals / 180.0,  # OpenCV Hue goes up to 180
        s_vals / 255.0,
        v_vals / 255.0
    ])
    
    # 5. K-Means Clustering
    # We expect 4 markers + 1 ball + noise, so let's try K=6
    K = 6
    print(f"Running K-Means (K={K}) on {len(features)} pixels...")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    elapsed = (time.perf_counter() - start_t) * 1000.0
    print(f"Clustering finished in {elapsed:.1f}ms")
    
    # 6. Visualization
    output_image = image.copy()
    
    # Assign a random color to each cluster for drawing
    np.random.seed(42)
    cluster_colors = np.random.randint(0, 255, size=(K, 3), dtype=np.uint8)
    
    # Draw every clustered pixel with its assigned cluster color (for debugging the mask)
    for i in range(len(x_coords)):
        c_id = labels[i]
        color = tuple(int(c) for c in cluster_colors[c_id])
        output_image[y_coords[i], x_coords[i]] = color
        
    # Calculate spatial centroids in original pixel coordinates and draw large circles
    centroids = kmeans.cluster_centers_
    for c_id in range(K):
        # Centroid features are [norm_x, norm_y, norm_h, norm_s, norm_v]
        cx = int(centroids[c_id, 0] * w)
        cy = int(centroids[c_id, 1] * h)
        avg_h = int(centroids[c_id, 2] * 180)
        
        color = tuple(int(c) for c in cluster_colors[c_id])
        
        # Draw a big circle around the centroid
        cv2.circle(output_image, (cx, cy), 15, color, 3)
        # Put text labeling the Hue near the centroid to help identify the marker
        cv2.putText(output_image, f"H:{avg_h}", (cx+20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, output_image)
    print(f"Saved clustering visualization to {output_path}")

if __name__ == '__main__':
    main()
