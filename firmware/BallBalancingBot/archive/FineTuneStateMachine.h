#ifndef FINE_TUNE_STATE_MACHINE_H
#define FINE_TUNE_STATE_MACHINE_H

#include <Arduino.h>

// Randomised command-sequence driver.
//
// On the first call to getNextTarget() it generates a random ordering of
// commands with random gaps between them, spanning a fixed run duration.
// It prints the full schedule (so you can see the order chosen this run),
// then fires each command at its scheduled timestamp, printing as it goes.
// After the run duration it returns the ball to centre and holds there.
//
// Commands:
//   go <colour>  -> drive to that colour's coordinate (clamped to guardrails)
//   forward/backward/left/right -> step 20mm in a cartesian direction (clamped)
//   stop / hold  -> latch the CURRENT ball position as the target so it settles
class DataCollectionStateMachine {
public:
    DataCollectionStateMachine();
    void getNextTarget(double &out_x, double &out_y, bool &is_done);

private:
    enum Command {
        CMD_GO_GREEN = 0,
        CMD_GO_RED,
        CMD_GO_BLUE,
        CMD_GO_GREY,
        CMD_GO_YELLOW,
        CMD_FORWARD,
        CMD_BACKWARD,
        CMD_LEFT,
        CMD_RIGHT,
        CMD_STOP,
        CMD_HOLD,
        NUM_COMMANDS   // must stay last: used as the count for random()
    };

    struct ScheduledCommand {
        unsigned long time_ms;  // fire time, measured from run start (0..RUN_DURATION_MS)
        Command       cmd;
    };

    static const int           MAX_COMMANDS    = 200;
    static const unsigned long RUN_DURATION_MS = 300000UL;  // 5 minute run
    static const unsigned long MIN_GAP_MS      = 1500UL;   // shortest gap between commands
    static const unsigned long MAX_GAP_MS      = 7000UL;   // longest gap (lets a target settle)

    ScheduledCommand schedule[MAX_COMMANDS];
    int              num_commands;
    int              next_command_idx;

    bool             run_started;
    bool             finished;
    unsigned long    run_start_ms;

    double target_x;
    double target_y;

    void        generateSchedule(unsigned long now);
    void        printSchedule();
    void        fireCommand(const ScheduledCommand &sc);
    const char* commandName(Command c);
};

#endif // DATA_COLLECTION_STATE_MACHINE_H