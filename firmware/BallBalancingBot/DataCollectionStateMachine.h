#ifndef DATA_COLLECTION_STATE_MACHINE_H
#define DATA_COLLECTION_STATE_MACHINE_H

#include <Arduino.h>

enum DataCollectionState {
    PHASE_1_RANDOM,
    PHASE_2_PATTERNS,
    PHASE_3_SWEEPS,
    PHASE_DONE
};

struct Point2D {
    double x;
    double y;
};

class DataCollectionStateMachine {
public:
    DataCollectionStateMachine();
    void getNextTarget(double &out_x, double &out_y, bool &is_done);

private:
    DataCollectionState state;
    unsigned long state_start_time_ms;
    unsigned long phase1_duration_ms;
    unsigned long phase2_duration_per_pattern_ms;
    
    int current_pattern_idx;
    
    Point2D start_points[9];
    int current_start_idx;
    
    Point2D directions[8];
    int current_dir_idx;
    double sweep_distance;
    
    double target_x;
    double target_y;
    
    unsigned long last_random_update_ms;
    unsigned long last_sweep_update_ms;
};

#endif
