#include "DataCollectionStateMachine.h"
#include <math.h>

DataCollectionStateMachine::DataCollectionStateMachine() {
    state = PHASE_1_RANDOM;
    state_start_time_ms = millis();
    last_random_update_ms = millis();
    last_sweep_update_ms = millis();
    
    phase1_duration_ms = 100000; // 100 seconds
    phase2_duration_per_pattern_ms = 50000; // 50 seconds
    
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
    
    target_x = 0.0;
    target_y = 0.0;
}

void DataCollectionStateMachine::getNextTarget(double &out_x, double &out_y, bool &is_done) {
    unsigned long now = millis();
    unsigned long elapsed_in_state = now - state_start_time_ms;
    is_done = false;
    
    if (state == PHASE_1_RANDOM) {
        if (now - last_random_update_ms >= 3333) { // 100 frames at 30Hz ~ 3.33 seconds
            target_x = random(-600, 600) / 10.0;
            target_y = random(-450, 450) / 10.0;
            last_random_update_ms = now;
        }
        
        if (elapsed_in_state > phase1_duration_ms) {
            state = PHASE_2_PATTERNS;
            state_start_time_ms = now;
            current_pattern_idx = 0;
        }
    }
    else if (state == PHASE_2_PATTERNS) {
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
            if (current_pattern_idx >= 2) {
                state = PHASE_3_SWEEPS;
                state_start_time_ms = now;
            }
        }
    }
    else if (state == PHASE_3_SWEEPS) {
        if (current_start_idx >= 9) {
            state = PHASE_DONE;
            is_done = true;
            out_x = target_x;
            out_y = target_y;
            return;
        }
        
        double start_x = start_points[current_start_idx].x;
        double start_y = start_points[current_start_idx].y;
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
        
        if (abs(target_x) > 65.0 || abs(target_y) > 50.0 || sweep_distance >= 100.0) {
            sweep_distance = 0.0;
            current_dir_idx++;
            if (current_dir_idx >= 8) {
                current_dir_idx = 0;
                current_start_idx++;
            }
        }
    }
    else if (state == PHASE_DONE) {
        is_done = true;
    }
    
    out_x = target_x;
    out_y = target_y;
}
