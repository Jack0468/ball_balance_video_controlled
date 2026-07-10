import sys
import threading
import collections
import struct
import csv

def analyze_log(filename):
    y_err = []
    x_err = []
    
    print(f"Analyzing CSV log: {filename}")
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    y_err.append(float(row['error_y']))
                    x_err.append(float(row['error_x']))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    if not x_err:
        print("No valid error data found in CSV.")
        return

    print(f"Total frames: {len(y_err)} (approx {len(y_err)*0.033:.2f} seconds at 30Hz)")
    
    # Check if settled
    y_in_deadband = [abs(e) < 3.0 for e in y_err]
    x_in_deadband = [abs(e) < 3.0 for e in x_err]
    
    def max_streak(lst):
        m = 0
        cur = 0
        for v in lst:
            if v:
                cur += 1
                m = max(m, cur)
            else:
                cur = 0
        return m
        
    print(f"Y axis max streak in deadband: {max_streak(y_in_deadband)} frames")
    print(f"X axis max streak in deadband: {max_streak(x_in_deadband)} frames")
    
    def find_settling_time(err_list):
        for i in range(len(err_list)):
            if all(abs(e) < 3.0 for e in err_list[i:]):
                return i
        return -1
        
    y_settle = find_settling_time(y_err)
    x_settle = find_settling_time(x_err)
    
    print(f"Y settled at frame: {y_settle} (approx {y_settle*0.033:.2f}s)")
    print(f"X settled at frame: {x_settle} (approx {x_settle*0.033:.2f}s)")
    
    def count_crossings(err_list):
        crossings = 0
        for i in range(1, len(err_list)):
            if err_list[i] * err_list[i-1] < 0:
                crossings += 1
        return crossings
        
    print(f"Y zero crossings: {count_crossings(y_err)}")
    print(f"X zero crossings: {count_crossings(x_err)}")

    def find_peaks(err_list):
        peaks = []
        for i in range(1, len(err_list)-1):
            if abs(err_list[i]) > abs(err_list[i-1]) and abs(err_list[i]) > abs(err_list[i+1]):
                if abs(err_list[i]) >= 3.0:
                    peaks.append(abs(err_list[i]))
        return peaks
        
    y_peaks = find_peaks(y_err)
    x_peaks = find_peaks(x_err)
    
    print(f"Y peaks (mm): {[round(p, 1) for p in y_peaks[:10]]}... (total {len(y_peaks)})")
    print(f"X peaks (mm): {[round(p, 1) for p in x_peaks[:10]]}... (total {len(x_peaks)})")


def live_plot(port):
    try:
        import serial
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Error: Missing required packages for live plotting.")
        print("Please run: pip install pyserial matplotlib")
        return

    print(f"Connecting to {port} at 2000000 baud...")
    try:
        ser = serial.Serial(port, 2000000, timeout=1)
    except Exception as e:
        print(f"Failed to connect to {port}: {e}")
        return

    MAX_POINTS = 300
    
    # Data deques
    y_errors = collections.deque(maxlen=MAX_POINTS)
    y_roll = collections.deque(maxlen=MAX_POINTS)
    x_errors = collections.deque(maxlen=MAX_POINTS)
    x_pitch = collections.deque(maxlen=MAX_POINTS)
    
    def read_serial():
        struct_format = "<Ifffffffffffffff"
        expected_size = struct.calcsize(struct_format)
        sync_buf = bytearray()
        
        while True:
            try:
                if ser.in_waiting > 0:
                    b = ser.read(1)
                    sync_buf.append(b[0])
                    if len(sync_buf) > 4:
                        sync_buf.pop(0)
                        
                    # Match sync header 0xAABBCCDD
                    if bytes(sync_buf) == b'\xAA\xBB\xCC\xDD':
                        data = ser.read(expected_size)
                        if len(data) == expected_size:
                            unpacked = struct.unpack(struct_format, data)
                            # unpacked: [0]=mcu_micros, [1]=target_x, [2]=target_y
                            # [3]=touch_x, [4]=touch_y, [5]=error_x, [6]=error_y
                            # [7]=pitch, [8]=roll, ...
                            
                            x_errors.append(unpacked[5])
                            y_errors.append(unpacked[6])
                            x_pitch.append(unpacked[7]) # pitch (controls X)
                            y_roll.append(unpacked[8])  # roll (controls Y)
                            
                        sync_buf.clear()
            except Exception as e:
                pass

    t = threading.Thread(target=read_serial, daemon=True)
    t.start()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title(f'Live Binary Telemetry Plot - {port}')
    
    # Y Error Plot
    line_y_err, = ax1.plot([], [], 'b-', label='Y Error (mm)')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axhline(3.0, color='r', linestyle='--', alpha=0.5, label='Deadband')
    ax1.axhline(-3.0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, MAX_POINTS)
    ax1.set_ylim(-90, 90)
    ax1.set_title('Y-Axis Error')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Y Output Angle Plot
    line_y_roll, = ax2.plot([], [], 'r-', label='Roll Angle')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlim(0, MAX_POINTS)
    ax2.set_ylim(-15, 15)
    ax2.set_title('Y-Axis Output Angle')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # X Error Plot
    line_x_err, = ax3.plot([], [], 'b-', label='X Error (mm)')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.axhline(3.0, color='r', linestyle='--', alpha=0.5, label='Deadband')
    ax3.axhline(-3.0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlim(0, MAX_POINTS)
    ax3.set_ylim(-90, 90)
    ax3.set_title('X-Axis Error')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # X Output Angle Plot
    line_x_pitch, = ax4.plot([], [], 'r-', label='Pitch Angle')
    ax4.axhline(0, color='black', linewidth=1)
    ax4.set_xlim(0, MAX_POINTS)
    ax4.set_ylim(-15, 15)
    ax4.set_title('X-Axis Output Angle')
    ax4.legend(loc='upper right')
    ax4.grid(True)

    def update(frame):
        if len(y_errors) > 0:
            rng_y = range(len(y_errors))
            line_y_err.set_data(rng_y, list(y_errors))
            line_y_roll.set_data(rng_y, list(y_roll))
            
        if len(x_errors) > 0:
            rng_x = range(len(x_errors))
            line_x_err.set_data(rng_x, list(x_errors))
            line_x_pitch.set_data(rng_x, list(x_pitch))
            
        return line_y_err, line_y_roll, line_x_err, line_x_pitch

    print("Opening live plot... Close the window to stop.")
    ani = animation.FuncAnimation(fig, update, interval=33, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    ser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_log.py <logfile.csv OR COM_PORT>")
        sys.exit(1)
        
    target = sys.argv[1]
    
    if target.upper().startswith("COM") or target.startswith("/dev/"):
        live_plot(target)
    else:
        analyze_log(target)
