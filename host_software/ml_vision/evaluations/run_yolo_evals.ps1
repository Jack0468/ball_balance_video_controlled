$ErrorActionPreference = "Stop"
Set-Location c:\Users\Admin\Documents\Windows_codespace\VRI_2026\host_software\ml_vision\evaluations

Write-Host "Running evaluate_yolo_homography.py on marker_ball model..."
python evaluate_yolo_homography.py --model_path ../models/yolov8_marker_ball_v1/weights/best.pt

Write-Host "Running evaluate_yolo_homography.py on YOLO pose models..."
python evaluate_yolo_homography.py --model_path ../models/yolov8_platform_pose_v1/weights/best.pt
python evaluate_yolo_homography.py --model_path ../models/yolov8_unified_pose_v1/weights/best.pt

Write-Host "Re-plotting model comparisons..."
python plot_model_comparisons.py

Write-Host "Done!"
