# STM32 Telemetry-Back — Proposed Protocol (Track 2)

**Status: proposal, not merged.** Everything in this folder is staged for the firmware
owner (Electrical Engineer) to review — `firmware/` is outside `agent_ml_multimodal`'s
write boundary (`.agents/agent_ml_multimodal.md`), so this is a reference implementation,
not a live change.

## Why this exists

Two gaps found while planning the Jetson port (2026-08-18):

1. **`BallBalancingBot.ino` (current source-of-truth firmware) never sends anything back
   to the host.** `loop()` only calls `rl_balance()`; `SerialCoords.cpp` is receive-only.
   `host_software/evaluations/evaluate_system_control.py`'s `REQUIRED_COLUMNS` needs
   `theta_a/b/c` (control-effort metric) — today that only exists via the separate,
   legacy `ExpertEvaluationFirmware` build's binary telemetry, not this one. So a
   Jetson-hosted Track 1 run, using this firmware, would produce a CSV missing the
   columns needed for full four-metric comparison.
2. **`RemoteStepControl.cpp/h`** (already written, not yet wired into `loop()`) is also
   receive-only. Phase B (control-net inference moved to the Jetson) needs the motors'
   *real, lagged* step position each cycle as a closed-loop input — `control_net.py`'s own
   docstring is explicit that feeding it anything else (e.g. the last commanded target)
   runs the trained policy out of distribution. Confirmed by reading `RLControl.cpp`
   directly: its own `actual_steps` input comes from `motorA.currentPosition()` /
   `motorB.currentPosition()` / `motorC.currentPosition()` (AccelStepper's real-time
   position) — **not** the `pos[]` array, which only holds the last commanded target.
   Any telemetry-back implementation must read the same source or it will silently feed
   the Jetson-side control net the wrong signal.

Both gaps share one fix: add an outbound telemetry channel. Designed as a **standalone
module** (`Telemetry.cpp/.h`, proposed below) called once per `loop()` iteration,
independent of which control path is active (`rl_balance()`, `remote_step_control_update()`,
or a future Phase-B path) — so it doesn't require invasively modifying either
`SerialCoords.cpp` or `RemoteStepControl.cpp`, just one call added to `BallBalancingBot.ino`'s
existing `loop()`.

**Not designed from scratch.** While staging this, `firmware/MLVisionPIDControl/PIDControllers.cpp`
was found to already implement exactly this idea — a packed `TelemetryPacket` struct sent
over `Serial` from inside `pid_balance()`, gated by an `enable_binary_telemetry` flag, using
the same `0xAABBCCDD`-style sync header. `MLVisionPIDControl.ino` is a PID-based firmware
variant not previously catalogued in `.agents/AGENTS.md`'s "at least six `BallBalancingBot.ino`
duplicates" flag — it wasn't caught because that audit was filename-based
(`BallBalancingBot.ino`) and this file is named differently. Worth folding into that
existing firmware-duplication audit, separately from this proposal. The struct/sync-header
design below deliberately matches that precedent's conventions (packed struct, sync header
as the struct's own first field, `micros()` not `millis()`) rather than inventing a
divergent format — the field *set* still differs, because the RL control law has no
PID-specific diagnostics (`error`/`integral`/`deriv` terms) to report, and needs
`actual_step_a/b/c` for a reason the PID variant doesn't have (Phase B's closed-loop
control net on the Jetson).

**Open discrepancy, flagged not resolved:** `PIDControllers.cpp`'s own telemetry logs
`steps_to_angle(pos[i])` — the commanded-target buffer — not `motorX.currentPosition()`
(the real, lagged position). This proposal deliberately uses `currentPosition()` instead,
because `RLControl.cpp` itself uses that as the control net's `actual_steps` input, but
this is a real inconsistency between firmware variants worth the firmware owner's attention,
not something silently normalized here.

## Wire format

Binary, matching `firmware/MLVisionPIDControl/PIDControllers.cpp`'s established
`TelemetryPacket` + `0xAABBCCDD`-style sync-header convention (also referenced generally in
`.agents/AGENTS.md`'s "Telemetry sync protocol" section):

```text
struct TelemetryPacket (packed, sent as one Serial.write()):
[uint32 sync_header]       0xDDCCBBAA in memory -> AA BB CC DD on the wire (little-endian)
[uint32 mcu_micros]        firmware-side timestamp, micros() not millis() (matches precedent)
[float target_x_mm]
[float target_y_mm]
[float touch_x_mm]         from get_coords()/SerialCoords.cpp (or 0.0 if not tracked)
[float touch_y_mm]
[int32 actual_step_a]      motorA.currentPosition() -- see discrepancy note above
[int32 actual_step_b]      motorB.currentPosition()
[int32 actual_step_c]      motorC.currentPosition()
[float theta_a_deg]        steps_to_angle(actual_step_a)  -- MotorControl.cpp, already exists
[float theta_b_deg]        steps_to_angle(actual_step_b)
[float theta_c_deg]        steps_to_angle(actual_step_c)
```

`theta_a/b/c` are derived purely for `evaluate_system_control.py` schema compatibility —
they are not needed for control itself (the RL control net and `RemoteStepControl.cpp`
both work directly in step-space). Deriving them here (server-side, at the point closest
to the ground-truth `currentPosition()` reading) avoids any ambiguity from re-deriving
them later on the Jetson from a possibly-stale step count.

Sent once per `loop()` iteration on the same `Serial` the STM32 already uses for RX
(`SerialCoords.cpp`/`RemoteStepControl.cpp` both default to `Serial` at 2,000,000 baud) —
full-duplex, so this doesn't interfere with inbound coordinate/step traffic.

## Jetson-side consumer

`host_software/ml_jetson_vla/runtime/telemetry_logger.py` parses this stream and writes
exactly `evaluate_system_control.py`'s `REQUIRED_COLUMNS` schema
(`host_timestamp_ms, target_x, target_y, touch_x, touch_y, theta_a, theta_b, theta_c`) to
CSV, so a Jetson-hosted run drops into `evaluate_system_control.py --runs label=csv ...`
with zero changes to that tool.

## What's NOT in scope here

Wiring `remote_step_control_update()` into `BallBalancingBot.ino`'s `loop()` (swapping it
in for `rl_balance()`, already documented as a one-line change in `RemoteStepControl.h`'s
own comment) is a **Phase B** action, not part of this proposal. This telemetry module is
useful on its own under the *current* `rl_balance()`/`SerialCoords.cpp` setup — it's what
makes Track 1's Jetson runs fully comparable via the standard four metrics, independent of
whether/when Phase B happens.
