import time

class PIDController:
    def __init__(self, kp=0.8, ki=0.2, kd=0.09, kp_adj=0.4, ki_adj=0.25, kd_adj=0.23):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.kp_adj = kp_adj
        self.ki_adj = ki_adj
        self.kd_adj = kd_adj
        
        self.max_output = 83.5
        self.max_angle = 12.5
        
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        
        self.prev_ball_x = 0.0
        self.prev_ball_y = 0.0
        
        self.prev_time = time.time()
        self.last_detected_time = time.time()

    def calculate_angles(self, setpoint_x, setpoint_y, ball_x, ball_y):
        t = time.time()
        dt = t - self.prev_time
        
        if dt > 0.075 or dt <= 0.0:
            self.prev_time = t
            return 0.0, 0.0
            
        self.last_detected_time = t
        
        # Calculate velocity (damping)
        ball_vel_x = (ball_x - self.prev_ball_x) / (dt * 50)
        ball_vel_y = (ball_y - self.prev_ball_y) / (dt * 50)
        self.prev_ball_x = ball_x
        self.prev_ball_y = ball_y
        
        error_x = ball_x - setpoint_x
        error_y = ball_y - setpoint_y
        
        # X Axis PID
        self.integral_x += error_x * dt
        self.integral_x = max(-50, min(50, self.integral_x)) # Anti-windup
        deriv_x = (error_x - self.prev_error_x) / dt
        self.prev_error_x = error_x
        
        if abs(error_x) < 25:
            out_x = self.kp_adj * error_x + self.ki_adj * self.integral_x + self.kd_adj * deriv_x
        else:
            out_x = self.kp * error_x + self.ki * self.integral_x + self.kd * deriv_x
            
        # Y Axis PID
        self.integral_y += error_y * dt
        self.integral_y = max(-50, min(50, self.integral_y)) # Anti-windup
        deriv_y = (error_y - self.prev_error_y) / dt
        self.prev_error_y = error_y
        
        if abs(error_y) < 25:
            out_y = self.kp_adj * error_y + self.ki_adj * self.integral_y + self.kd_adj * deriv_y
        else:
            out_y = self.kp * error_y + self.ki * self.integral_y + self.kd * deriv_y
            
        # Constrain and scale to angles
        out_x = max(-self.max_output, min(self.max_output, out_x))
        out_y = max(-self.max_output, min(self.max_output, out_y))
        
        angle_x = out_x * (self.max_angle / self.max_output)
        angle_y = out_y * (self.max_angle / self.max_output)
        
        self.prev_time = t
        return angle_x, angle_y

    def reset(self):
        self.integral_x = 0
        self.integral_y = 0
