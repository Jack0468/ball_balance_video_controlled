#include "DataCollectionStateMachine.h"
#include <math.h>

DataCollectionStateMachine::DataCollectionStateMachine() {
    state = PHASE_0_CENTERING;
    state_start_time_ms = millis();
    last_random_update_ms = millis();
    last_sweep_update_ms = millis();
    last_brownian_update_ms = millis();
    
    phase0_duration_ms = 10000;  // 10 seconds for centering
    phase1_duration_ms = 300000; // 300 seconds
    phase2_duration_per_pattern_ms = 50000; // 50 seconds
    phase5_duration_ms = 200000; // 200 seconds
    
    ewma_err_x = 100.0;
    ewma_err_y = 100.0;
    
    waiting_at_start = true;
    
    in_recovery = false;
    state_before_recovery = PHASE_0_CENTERING;
    
    current_pattern_idx = 0;
    
    int idx = 0;
    for (int x = -40; x <= 40; x += 40) {
        for (int y = -30; y <= 30; y += 30) {
            start_points[idx].x = x;
            start_points[idx].y = y;
            idx++;
        }
    }
    current_start_idx = 0;
    
    Point2D dirs[8] = {
        {0, 1}, {0, -1}, {-1, 0}, {1, 0},
        {1, 1}, {-1, -1}, {-1, 1}, {1, -1}
    };
    for (int i = 0; i < 8; i++) {
        directions[i] = dirs[i];
    }
    current_dir_idx = 0;
    sweep_distance = 0;
    current_edge_idx = 0;
    
    target_x = 0.0;
    target_y = 0.0;
}

void DataCollectionStateMachine::getNextTarget(double &out_x, double &out_y, bool &is_done) {
    unsigned long now = millis();
    unsigned long elapsed_in_state = now - state_start_time_ms;
    
    extern bool ball_detected;
    
    // Track the last time we had a solid reading
    if (ball_detected) {
        last_ball_detected_ms = now;
    }
    
    // If ball goes missing for > 1500ms, automatically hijack the state machine into recovery mode!
    // This perfectly debounces touchscreen ADC noise or bouncing.
    if ((now - last_ball_detected_ms > 1500) && !in_recovery) {
        in_recovery = true;
        state_before_recovery = state;
        state = PHASE_0_CENTERING;
        ewma_err_x = 100.0;
        ewma_err_y = 100.0;
        state_start_time_ms = now;
    }
    
    is_done = false;
    
    if (state == PHASE_0_CENTERING) {
        target_x = 0.0;
        target_y = 0.0;
        
        extern double current_ball_x;
        extern double current_ball_y;
        double err_x = abs(current_ball_x - target_x);
        double err_y = abs(current_ball_y - target_y);
        
        // Update EWMA error metrics (alpha = 0.015 gives roughly 8 seconds of settling memory)
        if (ball_detected) {
            ewma_err_x = 0.015 * err_x + 0.985 * ewma_err_x;
            ewma_err_y = 0.015 * err_y + 0.985 * ewma_err_y;
        }
        
        // Once the moving average drops below 3.0mm, we consider it completely settled
        if (ewma_err_x < 3.0 && ewma_err_y < 3.0) {
            if (in_recovery) {
                state = state_before_recovery;
                in_recovery = false;
            } else {
                // TEMP: Skip straight to Edges for data collection!
                state = PHASE_4_EDGES;
            }
            state_start_time_ms = now;
            
            // Reset EWMA for the next state
            ewma_err_x = 100.0;
            ewma_err_y = 100.0;
        }
    }
    else if (state == PHASE_1_RANDOM) {
        extern double current_ball_x;
        extern double current_ball_y;
        double err_x = abs(current_ball_x - target_x);
        double err_y = abs(current_ball_y - target_y);
        
        // Update EWMA error metrics (alpha = 0.015 gives roughly 8 seconds of settling memory)
        if (ball_detected) {
            ewma_err_x = 0.015 * err_x + 0.985 * ewma_err_x;
            ewma_err_y = 0.015 * err_y + 0.985 * ewma_err_y;
        }
        
        // Change target as soon as the raw error is within 15mm, to keep it moving continuously!
        if (err_x < 15.0 && err_y < 15.0) {
            // Pick a new target.
            double new_tx = random(-600, 600) / 10.0;
            double new_ty = random(-450, 450) / 10.0;
            
            // Final constraint to keep it within the PID physical guardrails
            target_x = constrain(new_tx, -70.0, 70.0);
            target_y = constrain(new_ty, -60.0, 60.0);
            
            // Reset EWMA for the next target
            ewma_err_x = 100.0;
            ewma_err_y = 100.0;
        }
        
        if (elapsed_in_state > phase1_duration_ms) {
            state = PHASE_2_PATTERNS;
            state_start_time_ms = now;
            current_pattern_idx = 0;
            waiting_at_start = true;
            ewma_err_x = 100.0;
            ewma_err_y = 100.0;
        }
    }
    else if (state == PHASE_2_PATTERNS) {
        if (waiting_at_start) {
            if (current_pattern_idx == 0) {
                target_x = 0.0; target_y = 0.0;
            } else {
                target_x = 5.0; target_y = 0.0; // Start of spiral
            }
            
            extern double current_ball_x;
            extern double current_ball_y;
            double err_x = abs(current_ball_x - target_x);
            double err_y = abs(current_ball_y - target_y);
            
            if (ball_detected) {
                ewma_err_x = 0.015 * err_x + 0.985 * ewma_err_x;
                ewma_err_y = 0.015 * err_y + 0.985 * ewma_err_y;
            }
            
            if (ewma_err_x < 3.0 && ewma_err_y < 3.0) {
                waiting_at_start = false;
                state_start_time_ms = now; // Restart the phase timer so we get the full duration!
                ewma_err_x = 100.0;
                ewma_err_y = 100.0;
            }
            return;
        }
        
        // Frame count equivalent for smooth math: 30 frames per second
        double t_frames = (elapsed_in_state / 1000.0) * 30.0;
        double t = t_frames * 0.05;
        
        if (current_pattern_idx == 0) { // figure8
            target_x = 50.0 * sin(t);
            target_y = 30.0 * sin(2*t);
        } else if (current_pattern_idx == 1) { // spiral
            double radius = 5.0 + 40.0 * ((double)elapsed_in_state / phase2_duration_per_pattern_ms);
            target_x = radius * cos(t * 1.5);
            target_y = radius * sin(t * 1.5);
        }
        
        if (elapsed_in_state > phase2_duration_per_pattern_ms) {
            current_pattern_idx++;
            state_start_time_ms = now; // reset timer for next pattern
            waiting_at_start = true;
            ewma_err_x = 100.0;
            ewma_err_y = 100.0;
            if (current_pattern_idx >= 2) {
                state = PHASE_3_SWEEPS;
                state_start_time_ms = now;
            }
        }
    }
    else if (state == PHASE_3_SWEEPS) {
        if (current_start_idx >= 9) {
            state = PHASE_4_EDGES;
            state_start_time_ms = now;
            waiting_at_start = true;
            ewma_err_x = 100.0;
            ewma_err_y = 100.0;
            current_edge_idx = 0;
            return;
        }
        
        double start_x = start_points[current_start_idx].x;
        double start_y = start_points[current_start_idx].y;
        
        if (waiting_at_start) {
            target_x = start_x;
            target_y = start_y;
            
            extern double current_ball_x;
            extern double current_ball_y;
            double err_x = abs(current_ball_x - target_x);
            double err_y = abs(current_ball_y - target_y);
            
            if (ball_detected) {
                ewma_err_x = 0.015 * err_x + 0.985 * ewma_err_x;
                ewma_err_y = 0.015 * err_y + 0.985 * ewma_err_y;
            }
            
            if (ewma_err_x < 3.0 && ewma_err_y < 3.0) {
                waiting_at_start = false;
                sweep_distance = 0.0;
                last_sweep_update_ms = now;
                ewma_err_x = 100.0;
                ewma_err_y = 100.0;
            }
            return;
        }
        
        double dir_dx = directions[current_dir_idx].x;
        double dir_dy = directions[current_dir_idx].y;
        
        // Normalize
        double mag = sqrt(dir_dx*dir_dx + dir_dy*dir_dy);
        dir_dx /= mag;
        dir_dy /= mag;
        
        // Every 10 frames (~333ms), move 5mm
        if (now - last_sweep_update_ms >= 333) {
            sweep_distance += 5.0;
            last_sweep_update_ms = now;
        }
        
        target_x = start_x + dir_dx * sweep_distance;
        target_y = start_y + dir_dy * sweep_distance;
        
        if (abs(target_x) > 85.0 || abs(target_y) > 65.0 || sweep_distance >= 100.0) {
            sweep_distance = 0.0;
            current_dir_idx++;
            waiting_at_start = true; // Wait for ball to return to start point!
            ewma_err_x = 100.0;
            ewma_err_y = 100.0;
            
            if (current_dir_idx >= 8) {
                current_dir_idx = 0;
                current_start_idx++;
            }
        }
    }
    else if (state == PHASE_4_EDGES) {
        // Trace the perimeter of the physical PID guardrail: (-70, -60) to (70, -60) to (70, 60) to (-70, 60) and back
        Point2D edge_points[4] = {
            {-70.0, -60.0},
            {70.0, -60.0},
            {70.0, 60.0},
            {-70.0, 60.0}
        };

        if (waiting_at_start) {
            target_x = edge_points[0].x;
            target_y = edge_points[0].y;
            
            extern double current_ball_x;
            extern double current_ball_y;
            double err_x = abs(current_ball_x - target_x);
            double err_y = abs(current_ball_y - target_y);
            
            if (ball_detected) {
                ewma_err_x = 0.015 * err_x + 0.985 * ewma_err_x;
                ewma_err_y = 0.015 * err_y + 0.985 * ewma_err_y;
            }
            
            if (ewma_err_x < 5.0 && ewma_err_y < 5.0) {
                waiting_at_start = false;
                current_edge_idx = 1; // start moving to next point
                last_sweep_update_ms = now;
                sweep_distance = 0.0;
            }
            return;
        }
        
        int prev_idx = (current_edge_idx - 1 + 4) % 4;
        double start_x = edge_points[prev_idx].x;
        double start_y = edge_points[prev_idx].y;
        double end_x = edge_points[current_edge_idx].x;
        double end_y = edge_points[current_edge_idx].y;
        
        double dir_dx = end_x - start_x;
        double dir_dy = end_y - start_y;
        double edge_len = sqrt(dir_dx*dir_dx + dir_dy*dir_dy);
        dir_dx /= edge_len;
        dir_dy /= edge_len;
        
        // Every 30 frames (~1000ms), move 10mm -> 10mm/s for a slower, thorough trace
        if (now - last_sweep_update_ms >= 100) {
            sweep_distance += 1.0;
            last_sweep_update_ms = now;
        }
        
        target_x = start_x + dir_dx * sweep_distance;
        target_y = start_y + dir_dy * sweep_distance;
        
        if (sweep_distance >= edge_len) {
            sweep_distance = 0.0;
            current_edge_idx++;
            // Do 3 full laps (12 edges) to spend "quite some time" here
            if (current_edge_idx >= 12) {
                // If we completed 3 full laps, finish!
                state = PHASE_5_BROWNIAN;
                state_start_time_ms = now;
                last_brownian_update_ms = now;
            }
        }
    }
    else if (state == PHASE_5_BROWNIAN) {
        if (now - last_brownian_update_ms >= 50) { // Update every 50ms (20Hz)
            double jump_x = random(-60, 61) / 10.0; // -6.0 to 6.0 mm jump
            double jump_y = random(-60, 61) / 10.0;
            
            target_x += jump_x;
            target_y += jump_y;
            
            target_x = constrain(target_x, -70.0, 70.0);
            target_y = constrain(target_y, -60.0, 60.0);
            
            last_brownian_update_ms = now;
        }
        
        if (elapsed_in_state > phase5_duration_ms) {
            state = PHASE_DONE;
            is_done = true;
        }
    }
    else if (state == PHASE_DONE) {
        is_done = true;
    }
    
    out_x = target_x;
    out_y = target_y;
}
