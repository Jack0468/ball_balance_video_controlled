#include "FineTuneStateMachine.h"

// Physical guardrails (PID safe area). Everything is clamped to these.
static const double X_MIN = -70.0, X_MAX = 70.0;
static const double Y_MIN = -60.0, Y_MAX = 60.0;

// Step size for the directional commands.
static const double STEP_MM = 20.0;

// Colour coordinates as given. Note several sit outside the guardrails
// (e.g. green x=100, grey x=-110); those get clamped when the command fires.
static const double GREEN_X  = 100.0,  GREEN_Y  = -50.0;
static const double RED_X    = -50.0,  RED_Y    = -30.0;
static const double BLUE_X   = 10.0,   BLUE_Y   = 50.0;
static const double GREY_X   = -110.0, GREY_Y   = 60.0;
static const double YELLOW_X = 100.0,  YELLOW_Y = 45.0;

DataCollectionStateMachine::DataCollectionStateMachine() {
    num_commands     = 0;
    next_command_idx = 0;
    run_started      = false;
    finished         = false;
    run_start_ms     = 0;
    target_x         = 0.0;
    target_y         = 0.0;
}

const char* DataCollectionStateMachine::commandName(Command c) {
    switch (c) {
        case CMD_GO_GREEN:  return "GO GREEN";
        case CMD_GO_RED:    return "GO RED";
        case CMD_GO_BLUE:   return "GO BLUE";
        case CMD_GO_GREY:   return "GO GREY";
        case CMD_GO_YELLOW: return "GO YELLOW";
        case CMD_FORWARD:   return "FORWARD (+Y)";
        case CMD_BACKWARD:  return "BACKWARD (-Y)";
        case CMD_LEFT:      return "LEFT (-X)";
        case CMD_RIGHT:     return "RIGHT (+X)";
        case CMD_STOP:      return "STOP (hold here)";
        case CMD_HOLD:      return "HOLD (hold here)";
        default:            return "UNKNOWN";
    }
}

void DataCollectionStateMachine::generateSchedule(unsigned long now) {
    // Seed from micros(). For stronger entropy tie in a floating analog pin,
    // e.g. randomSeed(micros() ^ analogRead(A0)), if you have a spare pin.
    randomSeed(17);

    num_commands = 0;
    unsigned long t = 0;
    while (t < RUN_DURATION_MS && num_commands < MAX_COMMANDS) {
        schedule[num_commands].time_ms = t;
        schedule[num_commands].cmd     = (Command)random(0, NUM_COMMANDS);
        num_commands++;
        t += random((long)MIN_GAP_MS, (long)MAX_GAP_MS + 1);
    }

    run_start_ms     = now;
    run_started      = true;
    finished         = false;
    next_command_idx = 0;
}

void DataCollectionStateMachine::printSchedule() {
    Serial.println();
    Serial.println(F("===== COMMAND SCHEDULE (this run) ====="));
    for (int i = 0; i < num_commands; i++) {
        Serial.print(F("  #"));
        if (i < 10) Serial.print(' ');
        Serial.print(i);
        Serial.print(F("   t="));
        Serial.print(schedule[i].time_ms / 1000.0, 2);
        Serial.print(F("s   "));
        Serial.println(commandName(schedule[i].cmd));
    }
    Serial.print(F("        t="));
    Serial.print(RUN_DURATION_MS / 1000.0, 2);
    Serial.println(F("s   RETURN TO CENTRE (0, 0)"));
    Serial.println(F("======================================="));
    Serial.println();
}

void DataCollectionStateMachine::fireCommand(const ScheduledCommand &sc) {
    // Current measured ball position, provided by the main program.
    extern double current_ball_x;
    extern double current_ball_y;

    switch (sc.cmd) {
        case CMD_GO_GREEN:  target_x = GREEN_X;  target_y = GREEN_Y;  break;
        case CMD_GO_RED:    target_x = RED_X;    target_y = RED_Y;    break;
        case CMD_GO_BLUE:   target_x = BLUE_X;   target_y = BLUE_Y;   break;
        case CMD_GO_GREY:   target_x = GREY_X;   target_y = GREY_Y;   break;
        case CMD_GO_YELLOW: target_x = YELLOW_X; target_y = YELLOW_Y; break;

        // Directional steps are relative to the current target.
        case CMD_FORWARD:   target_y += STEP_MM; break;  // +Y
        case CMD_BACKWARD:  target_y -= STEP_MM; break;  // -Y
        case CMD_LEFT:      target_x -= STEP_MM; break;  // -X
        case CMD_RIGHT:     target_x += STEP_MM; break;  // +X

        // Stop/hold latch wherever the ball currently is so it settles there.
        case CMD_STOP:
        case CMD_HOLD:
            target_x = current_ball_x;
            target_y = current_ball_y;
            break;

        default: break;
    }

    // Clamp to guardrails. This is also what stops "forward" from walking the
    // ball off the edge of the platform: it just saturates at the boundary.
    target_x = constrain(target_x, X_MIN, X_MAX);
    target_y = constrain(target_y, Y_MIN, Y_MAX);

    // Real-time log of the command actually firing + resulting clamped target.
    Serial.print('[');
    Serial.print(sc.time_ms / 1000.0, 2);
    Serial.print(F("s] "));
    Serial.print(commandName(sc.cmd));
    Serial.print(F(" -> ("));
    Serial.print(target_x, 1);
    Serial.print(F(", "));
    Serial.print(target_y, 1);
    Serial.println(')');
}

void DataCollectionStateMachine::getNextTarget(double &out_x, double &out_y, bool &is_done) {
    unsigned long now = millis();

    // First call: build and print the random schedule for this run.
    if (!run_started) {
        generateSchedule(now);
        printSchedule();
        target_x = 0.0;
        target_y = 0.0;
    }

    unsigned long elapsed = now - run_start_ms;

    if (elapsed >= RUN_DURATION_MS) {
        // Run finished: return to centre and hold there.
        if (!finished) {
            finished = true;
            Serial.print('[');
            Serial.print(RUN_DURATION_MS / 1000.0, 2);
            Serial.println(F("s] RUN COMPLETE -> returning to centre (0, 0)"));
        }
        target_x = 0.0;
        target_y = 0.0;
    } else {
        // Fire every command whose scheduled time has now arrived.
        while (next_command_idx < num_commands &&
               elapsed >= schedule[next_command_idx].time_ms) {
            fireCommand(schedule[next_command_idx]);
            next_command_idx++;
        }
    }

    // Never set is_done: we hold at centre rather than dumping the ball.
    // (If you'd rather end the program and tip the ball off, set is_done = true
    //  inside the "finished" block above instead.)
    is_done = false;

    out_x = target_x;
    out_y = target_y;
}