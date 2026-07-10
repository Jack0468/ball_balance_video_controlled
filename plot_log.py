import re
import sys
import threading
import collections

def analyze_log(filename):
    y_errors = []
    x_errors = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        if lines[i].startswith("P: "):
            if i+1 < len(lines) and lines[i+1].startswith("error[i]:"):
                m = re.search(r"error\[i\]:\s+([-\d.]+)", lines[i+1])
                if m:
                    err = float(m.group(1))
                    x_errors.append(err)

    if not x_errors:
        print("No errors found.")
        return

    y_err = x_errors[0::2]
    x_err = x_errors[1::2]
    
    print(f"Total frames: {len(y_err)} (approx {len(y_err)*0.01:.2f} seconds)")
    
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
    
    print(f"Y settled at frame: {y_settle} (approx {y_settle*0.01:.2f}s)")
    print(f"X settled at frame: {x_settle} (approx {x_settle*0.01:.2f}s)")
    
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

    print(f"Connecting to {port}...")
    try:
        ser = serial.Serial(port, 115200, timeout=1)
    except Exception as e:
        print(f"Failed to connect to {port}: {e}")
        return

    MAX_POINTS = 300
    
    # Y-axis data
    y_errors = collections.deque(maxlen=MAX_POINTS)
    y_P = collections.deque(maxlen=MAX_POINTS)
    y_I = collections.deque(maxlen=MAX_POINTS)
    y_D = collections.deque(maxlen=MAX_POINTS)
    
    # X-axis data
    x_errors = collections.deque(maxlen=MAX_POINTS)
    x_P = collections.deque(maxlen=MAX_POINTS)
    x_I = collections.deque(maxlen=MAX_POINTS)
    x_D = collections.deque(maxlen=MAX_POINTS)
    
    current_axis = 0 # 0 for Y, 1 for X
    
    def read_serial():
        nonlocal current_axis
        current_pid = None
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                # Check for PID components
                if line.startswith("P:"):
                    m = re.match(r"P:\s*([-\d.]+)\s+I:\s*([-\d.]+)\s+D:\s*([-\d.]+)\s+V:\s*([-\d.]+)", line)
                    if m:
                        current_pid = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
                        
                # Check for error
                elif line.startswith("error[i]:"):
                    m = re.search(r"error\[i\]:\s+([-\d.]+)", line)
                    if m and current_pid is not None:
                        err = float(m.group(1))
                        if current_axis == 0:
                            y_errors.append(err)
                            y_P.append(current_pid[0])
                            y_I.append(current_pid[1])
                            y_D.append(current_pid[2])
                            current_axis = 1
                        else:
                            x_errors.append(err)
                            x_P.append(current_pid[0])
                            x_I.append(current_pid[1])
                            x_D.append(current_pid[2])
                            current_axis = 0
                        current_pid = None
                        
            except Exception as e:
                pass

    t = threading.Thread(target=read_serial, daemon=True)
    t.start()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title(f'Live PID Tuning - {port}')
    
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

    # Y PID Plot
    line_y_p, = ax2.plot([], [], 'r-', label='P', alpha=0.8)
    line_y_i, = ax2.plot([], [], 'g-', label='I', alpha=0.8)
    line_y_d, = ax2.plot([], [], 'orange', label='D', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlim(0, MAX_POINTS)
    ax2.set_ylim(-150, 150)
    ax2.set_title('Y-Axis PID Output')
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

    # X PID Plot
    line_x_p, = ax4.plot([], [], 'r-', label='P', alpha=0.8)
    line_x_i, = ax4.plot([], [], 'g-', label='I', alpha=0.8)
    line_x_d, = ax4.plot([], [], 'orange', label='D', alpha=0.8)
    ax4.axhline(0, color='black', linewidth=1)
    ax4.set_xlim(0, MAX_POINTS)
    ax4.set_ylim(-150, 150)
    ax4.set_title('X-Axis PID Output')
    ax4.legend(loc='upper right')
    ax4.grid(True)

    def update(frame):
        if len(y_errors) > 0:
            rng_y = range(len(y_errors))
            line_y_err.set_data(rng_y, list(y_errors))
            line_y_p.set_data(rng_y, list(y_P))
            line_y_i.set_data(rng_y, list(y_I))
            line_y_d.set_data(rng_y, list(y_D))
            
        if len(x_errors) > 0:
            rng_x = range(len(x_errors))
            line_x_err.set_data(rng_x, list(x_errors))
            line_x_p.set_data(rng_x, list(x_P))
            line_x_i.set_data(rng_x, list(x_I))
            line_x_d.set_data(rng_x, list(x_D))
            
        return line_y_err, line_y_p, line_y_i, line_y_d, line_x_err, line_x_p, line_x_i, line_x_d

    print("Opening live plot... Close the window to stop.")
    ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    ser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_log.py <logfile.log OR COM_PORT>")
        sys.exit(1)
        
    target = sys.argv[1]
    
    # Check if the argument is a COM port or Linux serial device
    if target.upper().startswith("COM") or target.startswith("/dev/"):
        live_plot(target)
    else:
        analyze_log(target)
