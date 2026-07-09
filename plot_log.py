import re
import sys

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

if __name__ == "__main__":
    analyze_log(sys.argv[1])
