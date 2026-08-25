# Firmware Spec — Jetson Port (Phase A)

For the firmware/electrical engineer. Hardware bring-up is starting, so this is scoped
tight: one required item, two decisions we need your call on, and an explicit list of
what's *not* being asked for right now.

## TL;DR

- **Nothing is required to get the Jetson driving the platform at all.** The current
  firmware (`stm32_ml_control_and_vision/BallBalancingBot.ino`, `SerialCoords.cpp` +
  `RLControl.cpp`) already accepts exactly the wire format the Jetson sends
  (`ball_x,ball_y,target_x,target_y\n` ASCII @ 2,000,000 baud) — unchanged from how the
  laptop talks to it today. Plug the Jetson in over USB in place of the laptop and it
  should just work.
- **One thing is needed for full comparative evaluation**: telemetry-back (STM32 → host).
  Without it we can still measure 3 of the 4 standard metrics (steady-state error,
  settling time, task success rate — all derivable from the Jetson's own vision+target
  data). The 4th, control effort, needs `theta_a/b/c` (motor angles), which only the STM32
  knows. This is the one ask below.
- **Nothing about motor control changes.** The STM32 keeps computing control on-device
  (`RLControl.cpp`) exactly as today. We are not asking you to change how the plate moves.

## The ask: telemetry-back

Add an outbound send, once per `loop()` iteration, reporting what the STM32 currently
knows: the ball/target position it received, the motors' real step position, and their
angle equivalent.

**A working reference for this already exists in this codebase** —
`firmware/MLVisionPIDControl/PIDControllers.cpp` sends exactly this shape of packet
(`TelemetryPacket` struct + `0xAABBCCDD`-style sync header) from inside `pid_balance()`,
gated by `enable_binary_telemetry`. We noticed this got flipped on there recently — if
that's active testing, would be good to sync so we're not duplicating effort.

A drop-in-ready version for the RL firmware (`stm32_ml_control_and_vision/`) is staged at
`Telemetry.h` / `Telemetry.cpp` in this same folder — full wire format in
`TELEMETRY_PROTOCOL.md`. Summary:

```text
struct TelemetryPacket (packed, one Serial.write() per loop):
  uint32 sync_header     0xDDCCBBAA in memory -> AA BB CC DD on the wire
  uint32 mcu_micros
  float  target_x_mm, target_y_mm
  float  touch_x_mm, touch_y_mm
  int32  actual_step_a, actual_step_b, actual_step_c   -- motorX.currentPosition()
  float  theta_a_deg, theta_b_deg, theta_c_deg          -- steps_to_angle(actual_step_X)
```

**Integration point**: one call, `telemetry_send(touch_x, touch_y, target_x, target_y)`,
added to `BallBalancingBot.ino`'s `loop()` after `rl_balance()` runs. `Telemetry.cpp`
doesn't touch `SerialCoords.cpp` or `RLControl.cpp` — it only reads from `get_coords()`
and `motorA/B/C.currentPosition()`, both already available.

**Not compiled/tested here** — `Telemetry.cpp` was written outside the actual Arduino
sketch tree (can't resolve `MotorControl.h`/`Arduino.h` from here), so it needs a real
compile once dropped into the sketch directory. Please sanity-check it builds before
trusting it.

## Two things we need your call on

**1. Which firmware is "the" firmware for this?** `MLVisionPIDControl.ino` (PID-based,
telemetry already built) and `stm32_ml_control_and_vision/BallBalancingBot.ino` (RL-based,
current documented source of truth, no telemetry) are two different control laws. The
Jetson port is targeting the RL one, since that's what's documented as live — but if
`MLVisionPIDControl` is what you're actually running on the bench right now, tell us and
we'll retarget the spec to match instead of duplicating work against the wrong base.

**2. `currentPosition()` vs. `pos[]` for angle reporting.** `RLControl.cpp` treats
`motorX.currentPosition()` (real, lagged stepper position) as ground truth — it's what the
RL policy was trained against. `PIDControllers.cpp`'s own telemetry instead logs
`steps_to_angle(pos[i])` — the *commanded target*, not the real position. These aren't
interchangeable: `pos[]` can be ahead of where the motor physically is mid-move. Our
`Telemetry.cpp` proposal uses `currentPosition()` to match `RLControl.cpp`'s own
convention — flagging in case there's a reason the PID firmware does it the other way that
we're missing.

## Explicitly not asked for right now

- Wiring `RemoteStepControl.cpp` into `loop()` (accepting precomputed step targets from
  the Jetson, bypassing `RLControl.cpp`'s on-device inference). That file already exists
  but isn't called anywhere — it's a real future step (moving control-net inference onto
  the Jetson) but deliberately deferred until Phase A is running and validated. Don't wire
  it in yet.
- Any change to `RLControl.cpp`, `MotorControl.cpp`, or the control law itself.
- Any change to `SerialCoords.cpp`'s inbound wire format — the Jetson sends the exact same
  thing the laptop does.
