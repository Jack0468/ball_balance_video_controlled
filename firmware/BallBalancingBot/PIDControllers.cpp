#include <Arduino.h>
#include <math.h>
#include "PIDControllers.h"

// PID Constants
// PID Constants
#define kp 0.25           //.8
#define ki 0.1           //.2
#define kd 0.05           //.09
#define kv 0.0           //.05
#define kp_adj 0.3       // restored to original
#define ki_adj 0.3       // restored to original
#define kd_adj 0.05      // restored to original
#define max_output 73.5  // max X distance away from the center
#define max_angle 12.5   // max tilt angle

extern bool enable_binary_telemetry;
extern double current_ball_x;
extern double current_ball_y;
extern bool ball_detected;

bool enable_binary_telemetry = true;
double current_ball_x = 0;
double current_ball_y = 0;
bool ball_detected = false;

#pragma pack(push, 1)
struct TelemetryPacket {
    uint32_t sync_header;
    uint32_t mcu_micros;
    float target_x;
    float target_y;
    float touch_x;
    float touch_y;
    float error_x;
    float error_y;
    float pitch;
    float roll;
    float theta_a;
    float theta_b;
    float theta_c;
    float integral_x;
    float integral_y;
    float deriv_x;
    float deriv_y;
};
#pragma pack(pop)

//Variables needed for PID control
double error[2] = { 0, 0 }, error_prev[2], error_raw_prev[2] = { 0, 0 }, integ[2] = { 0, 0 }, deriv[2] = { 0, 0 }, deriv_prev[2] = { 0, 0 }, output[2], output_angles[2];  // error/error_prev for P, integ for I, deriv for D
// variables for velocity damping
double ball_vel[2] = {0,0}, p_prev[2] = {0,0};
double predict_time = 0.3; 

//Runs X and Y PID Controllers to balance ball at a specific point
void pid_balance(double setpoint_x, double setpoint_y) {

  static unsigned long t_prev = 0;              // Records previous time (uses static so the variable gets remembered through each loop iteration)
  static unsigned long last_detected_time = 0;  // Track when ball was last detected
  unsigned long t = millis();                   // Records current time using millis() (which counts total time since the program started)
  double dt = (t - t_prev) / 1000.0;            // Converts milliseconds to seconds

  // Handles first run or unreasonable time gaps
  if (t_prev == 0 || dt > 0.500) {
    t_prev = t;
    //Serial.println((String) "dt: " + dt + ". Time gap too large or first run. Resetting timing.");
    return;  // Ends function
  }

  // Only runs if minimum sample time has passed (0.020 seconds or 20 milliseconds for 50Hz)
  if (dt < 0.020) {
    return;
  }

    coords p = get_coords();           // retrieves ball's position from touchscreen
    bool detected = false;
    
    // Pure data collection: Use the touchpad as the primary input!
    if (p.z > 0.5) {
      detected = true;
      current_ball_x = p.x_mm;
      current_ball_y = p.y_mm;
    } else {
      detected = false;
    }
    ball_detected = detected;

    if (detected) {
      last_detected_time = t;

      // Target position guardrail: Never allow the state machine to target closer than 14mm to the edge!
      // Physical edges: 85mm (X) and 75mm (Y). Max targets: 71mm (X) and 61mm (Y).
      setpoint_x = constrain(setpoint_x, -71.0, 71.0);
      setpoint_y = constrain(setpoint_y, -61.0, 61.0);

      // --- SETPOINT SLEW RATE LIMITER ---
      // Moves the internal PID target smoothly towards the requested setpoint
      // at a maximum velocity of 80mm/second. This prevents massive instantaneous
      // errors from throwing the ball off the table when crossing the center.
      static double internal_setpoint_x = 0;
      static double internal_setpoint_y = 0;
      
      double max_slew_velocity = 80.0; // mm per second
      double max_step = max_slew_velocity * dt;
      
      // Slew X
      if (setpoint_x > internal_setpoint_x) {
          internal_setpoint_x += max_step;
          if (internal_setpoint_x > setpoint_x) internal_setpoint_x = setpoint_x;
      } else if (setpoint_x < internal_setpoint_x) {
          internal_setpoint_x -= max_step;
          if (internal_setpoint_x < setpoint_x) internal_setpoint_x = setpoint_x;
      }
      
      // Slew Y
      if (setpoint_y > internal_setpoint_y) {
          internal_setpoint_y += max_step;
          if (internal_setpoint_y > setpoint_y) internal_setpoint_y = setpoint_y;
      } else if (setpoint_y < internal_setpoint_y) {
          internal_setpoint_y -= max_step;
          if (internal_setpoint_y < setpoint_y) internal_setpoint_y = setpoint_y;
      }

      // --- VIRTUAL WALL (Edge Protection) ---
      if (current_ball_x > 71.0) internal_setpoint_x = 56.0;
      else if (current_ball_x < -71.0) internal_setpoint_x = -56.0;
      
      if (current_ball_y > 61.0) internal_setpoint_y = 46.0;
      else if (current_ball_y < -61.0) internal_setpoint_y = -46.0;

      // Predictive velocity control
      ball_vel[0] = (current_ball_y - p_prev[0]) / (dt*50);
      ball_vel[1] = (current_ball_x - p_prev[1]) / (dt*50);

      p_prev[0] = current_ball_y;
      p_prev[1] = current_ball_x;


      for (int i = 0; i < 2; i++) {
        //if (i == 1) continue; // Skip X axis during Y tuning
        //if (i == 0) continue; // Skip Y axis during X tuning

        error_prev[i] = error[i];
        double error_raw = (i == 0) ? (current_ball_y - internal_setpoint_y) : (current_ball_x - internal_setpoint_x);  // Calculates error based on ball position
        double error_current = error_raw;

        // REMOVED 3.0mm DEADBAND
        // The deadband was causing a fight with the State Machine!
        // If the PID goes to sleep at 2.9mm error, the State Machine EWMA
        // will never mathematically hit strictly < 3.0mm, causing a permanent stall!
        // We must let the PID controller constantly fight to hit exactly 0.0.

        error[i] = error_current; 
        
        // Calculate derivative on Measurement (ball velocity) rather than Error!
        // This prevents "Derivative Kick" where the continuously moving slew-target injects massive fake velocity into the D-term!
        static double p_prev_raw[2] = {0, 0};
        double current_pos = (i == 0) ? current_ball_y : current_ball_x;
        double raw_deriv = (current_pos - p_prev_raw[i]) / dt;
        p_prev_raw[i] = current_pos;
        
        raw_deriv = isnan(raw_deriv) || isinf(raw_deriv) ? 0 : raw_deriv;
        
        // Because we are running at 30Hz instead of 100Hz, the time step is 3.3x longer.
        // To maintain the same physical filter time-constant (0.1 seconds), we must increase alpha from 0.10 to 0.35.
        // If we don't, the D-term will lag 3x further behind reality, which makes the robot 'cooked' and unstable!
        deriv[i] = (0.35 * raw_deriv) + (0.65 * deriv_prev[i]);
        deriv_prev[i] = deriv[i];
        
        double v = constrain(ball_vel[i], -1000, 1000); // chooses ball velocity from earlier depending on axis
        integ[i] += error[i] * dt;                                                       // Calculates integral term by summing up error * dt
        integ[i] = constrain(integ[i], -50, 50);                                         // Reduced to prevent windup throwing it off the edge

        if (abs(error[i]) < 25) {
          output[i] = kp_adj * error[i] + ki_adj * integ[i] + kd_adj * deriv[i];
          if (!enable_binary_telemetry) Serial.println((String) "P: " + (kp_adj * error[i]) + " I: " + (ki_adj * integ[i]) + " D: " + (kd_adj * deriv[i]) + " V: " + (kv * v));
        }
        else {
          output[i] = kp * error[i] + ki * integ[i] + kd * deriv[i] - kv*v; // Forms output by adding P, I, and D terms. Currently an arbitrary value
          if (!enable_binary_telemetry) Serial.println((String) "P: " + (kp * error[i]) + " I: " + (ki * integ[i]) + " D: " + (kd * deriv[i]) + " V: " + (kv * v));
        }

        output_angles[i] = constrain(output[i], -max_output, max_output) * (max_angle / max_output);  // scales down PID output and maps it to an angle
        if (!enable_binary_telemetry) Serial.println((String) "error[i]: " + error[i] + " .error_prev[i]: " + error_prev[i] + " .dt: " + dt);
      }

      // SPIKE FILTERING
      // static double prev_output_angles[2] = {output_angles[0], output_angles[1]};  // Store previous angles for spike detection
      // double spike_threshold = 7.0; 
      // static int spike_count[2] = { 0, 0 };  // Track consecutive spikes

      // for (int i = 0; i < 2; i++) {
      //   if (abs(output_angles[i] - prev_output_angles[i]) > spike_threshold) {
      //     spike_count[i]++;
      //     if (spike_count[i] < 3) {
      //       Serial.println((String) "Spike detected on axis " + i +  ": " + output_angles[i] + " -> " + prev_output_angles[i]);  
            
      //       // Allow up to 3 consecutive spikes
      //       output_angles[i] = prev_output_angles[i];  // Reject spike
            
      //     }
      //     // After 3 spikes, allow the change (might be a valid large correction)
      //   } else {
      //     spike_count[i] = 0;  // Reset counter on normal operation
      //   }
      //   prev_output_angles[i] = output_angles[i];
      // }
      
      // Update motor targets
      move_to_angle(output_angles[0], -output_angles[1], 80);
      
      // Dynamically adjust max speeds to smooth out discrete 30Hz movements!
      speed_controller();

      // Send telemetry packet if enabled
      if (enable_binary_telemetry) {
          TelemetryPacket pkt;
          pkt.sync_header = 0xDDCCBBAA; // 0xAABBCCDD packed little-endian
          pkt.mcu_micros = micros();
          pkt.target_x = setpoint_x; // Log the true state machine target
          pkt.target_y = setpoint_y;
          pkt.touch_x = p.x_mm;
          pkt.touch_y = p.y_mm;
          pkt.error_x = error[1];
          pkt.error_y = error[0];
          pkt.pitch = output_angles[1];
          pkt.roll = output_angles[0];
          
          pkt.theta_a = steps_to_angle(pos[0]);
          pkt.theta_b = steps_to_angle(pos[1]);
          pkt.theta_c = steps_to_angle(pos[2]);
          
          pkt.integral_x = integ[1];
          pkt.integral_y = integ[0];
          pkt.deriv_x = deriv[1];
          pkt.deriv_y = deriv[0];
          
          Serial.write((uint8_t*)&pkt, sizeof(TelemetryPacket));
      }

    }

    else {
      // Check if ball has been undetected for 3 seconds (3000 milliseconds)
      if (t - last_detected_time >= 3000) {
        integ[0] = integ[1] = 0;  // Reset integral terms after 3 seconds
        move_to_angle(0,0,80);
        //Serial.println("Ball not detected for 3 seconds - resetting integral terms");
      }
    }
    t_prev = t;  // resets value of t_prev
}

//uses the PID controller to move the ball to a point for a specified amount of time (in milliseconds)
void move_to_point(double setpoint_x, double setpoint_y, unsigned long delay) {
  unsigned long t_prev = millis();
  while (millis() - t_prev < delay) {
    pid_balance(setpoint_x, setpoint_y);
  }
}


