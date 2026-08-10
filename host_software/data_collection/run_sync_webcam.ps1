# Replace the timestamp below with your actual session folder name!
$SESSION = "host_software\data\01_bronze\session_20260810_114330"

python host_software\data_collection\sync_webcam_telemetry.py `
  --frame-timestamps $SESSION\frame_timestamps.csv `
  --telemetry $SESSION\telemetry.csv `
  --output $SESSION\synced_telemetry.csv
