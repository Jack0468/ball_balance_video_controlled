import time
import os


class RealtimeLatencyMonitor:
    def __init__(self, log_interval=100, save_dir="ml_vision/evaluations"):
        self.log_interval = log_interval
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.vision_latencies = []
        self.mlp_latencies = []
        self.audio_latencies = []
        self.serial_latencies = []
        self.total_latencies = []

        self.current_start = 0.0
        self.current_vision_end = 0.0
        self.current_mlp_end = 0.0
        self.current_audio_end = 0.0

        self.frame_count = 0
        self.plot_thread = None

    def start_frame(self):
        self.current_start = time.perf_counter()

    def end_vision(self):
        self.current_vision_end = time.perf_counter()

    def end_mlp(self):
        self.current_mlp_end = time.perf_counter()

    def end_audio(self):
        self.current_audio_end = time.perf_counter()

    def end_frame(self, log_to_console=False):
        end_t = time.perf_counter()

        vision_lat = (self.current_vision_end - self.current_start) * 1000.0
        mlp_lat = (self.current_mlp_end - self.current_vision_end) * 1000.0
        audio_lat = (self.current_audio_end - self.current_mlp_end) * 1000.0
        serial_lat = (end_t - self.current_audio_end) * 1000.0
        total_lat = (end_t - self.current_start) * 1000.0

        self.vision_latencies.append(vision_lat)
        self.mlp_latencies.append(mlp_lat)
        self.audio_latencies.append(audio_lat)
        self.serial_latencies.append(serial_lat)
        self.total_latencies.append(total_lat)

        self.frame_count += 1

        if log_to_console:
            print(
                f"[Latency] Total: {total_lat:.1f}ms | Vision: {vision_lat:.1f}ms | MLP: {mlp_lat:.1f}ms | Audio: {audio_lat:.1f}ms | Serial: {serial_lat:.1f}ms"
            )

        if self.frame_count % self.log_interval == 0:
            self._save_data()

    def _save_data(self):
        import json

        save_path = os.path.join(self.save_dir, "realtime_system_latency.json")
        data = {
            "vision": self.vision_latencies,
            "mlp": self.mlp_latencies,
            "audio": self.audio_latencies,
            "serial": self.serial_latencies,
            "total": self.total_latencies,
        }
        with open(save_path, "w") as f:
            json.dump(data, f)
