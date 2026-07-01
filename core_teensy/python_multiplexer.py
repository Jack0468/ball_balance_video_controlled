import serial
import time
import struct
import numpy as np

class PythonMultiplexer:
    def __init__(self, port, baudrate=2000000):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.frame_width = 160
        self.frame_height = 120
        self.frame_size = self.frame_width * self.frame_height * 2 # RGB565 (2 bytes per pixel)
        self.sync_header = b'\xAA\xBB\xCC\xDD'
        
        # State variables
        self.target_x = 0.0
        self.target_y = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        
    def read_frame(self):
        """Reads a QQVGA frame from the Teensy Serial stream."""
        # 1. Align to sync header
        sync_buffer = bytearray()
        while True:
            if self.ser.in_waiting > 0:
                sync_buffer.extend(self.ser.read(1))
                if len(sync_buffer) >= 4:
                    if sync_buffer[-4:] == self.sync_header:
                        break
        
        # 2. Read the full frame buffer
        frame_data = self.ser.read(self.frame_size)
        if len(frame_data) == self.frame_size:
            # Optionally convert RGB565 to standard format using numpy here for ml_video
            return frame_data
        return None

    def update_ml_outputs(self, target_var, current_pos):
        """
        Updates the target and current positions from the ML modules.
        target_var: Tuple (x, y) from ml_audio
        current_pos: Tuple (x, y) from ml_video
        """
        self.target_x, self.target_y = target_var
        self.current_x, self.current_y = current_pos
        
    def send_coordinates_to_teensy(self):
        """Transmits the multiplexed physical coordinates to the Teensy PID loop."""
        # Format: T<target_x>,<target_y>C<current_x>,<current_y>\n
        command = f"T{self.target_x:.2f},{self.target_y:.2f}C{self.current_x:.2f},{self.current_y:.2f}\n"
        self.ser.write(command.encode('utf-8'))

    def run_loop(self):
        print("Starting Multiplexer loop...")
        try:
            while True:
                # 1. Receive camera frame from Teensy
                frame = self.read_frame()
                
                # 2. In a real system, pass 'frame' to ml_video, and get audio from ml_audio.
                # For baseline architecture skeleton, we simulate dummy coordinates.
                # self.current_x, self.current_y = ml_video.process(frame)
                # self.target_x, self.target_y = ml_audio.get_target()
                
                # 3. Send physical coordinates back to Teensy for PID actuation
                self.send_coordinates_to_teensy()
                
        except KeyboardInterrupt:
            print("Multiplexer stopped.")
            self.ser.close()

if __name__ == "__main__":
    # Replace 'COM3' or '/dev/ttyACM0' with the actual Teensy port
    multiplexer = PythonMultiplexer(port='COM3')
    multiplexer.run_loop()
