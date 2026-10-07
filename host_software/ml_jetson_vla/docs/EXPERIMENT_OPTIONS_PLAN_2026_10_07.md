# Experiment options plan (2026-10-07)

**Status: plan and code scaffold. No model has been trained, no hardware has been run, no result is claimed here.**
Code: `host_software/ml_jetson_vla/experiments/` (entry point `run_experiment.py`). Tests: `tests/test_experiments_cpu.py`.

## 0. Architecture under test

- **Slow layer (vision head):** Qwen2.5-VL-3B grounding of a colour marker on the current frame, then the existing pixel-to-telemetry-mm chain. ~1.7 s per call (`MULTI_HEAD_ARCHITECTURE_SPEC.md` risk 6).
- **Orchestrator:** free-text command to `TargetRequest`. The fast layer never receives a colour word.
  - **Option S** (`StateMachineOrchestrator`, implemented): colour commands map through `COLOR_COMMANDS` to a grounding label. `hold`/`stop` give a fixed hold point at the ball. `forward/backward/left/right` nudge the hold point by 15% of the platform, clamped to 90% of half-range (signs and constants mirror `src/state_machine.py`).
  - **Option L** (`LanguageOrchestrator`, stub): raises `NotImplementedError` unless a `parse_fn` is injected. No model is picked or loaded.
- **Fast layer:** `FastLayer.plan(target_mm, state_mm, frame)` returns a `ChunkPlan`. Every chunk is safety-checked inside `plan()`.
- **Output:** the firmware angle line `A,<a>,<b>,<c>` in degrees, limit +/-11.025 deg. A chunk with any value outside the limit (or non-finite) is rejected whole and nothing is sent. Nothing is clamped.

## 1. Option A: ACT_FAST (fast layer) with option S (orchestrator)

**What it is.** ACT (`lerobot` ACTPolicy, 51.6M params, ResNet18 backbone) with the target added as two extra state elements. `observation.state` becomes `[touch_x, touch_y, target_x, target_y]`. The 4-dim config comes from `bench_action_models.build_act_feature_spec(with_target=True)`.

**New versus pretrained.**
- Pretrained: ResNet18 backbone initialisation, if enabled (the bench sets `pretrained_backbone_weights=None`, so none is used).
- New: the whole ACT head, and the 2 target state dimensions. No general-purpose ACT checkpoint exists.

**What must be trained.**
1. The LeRobot converter must emit the target. **Gap:** `convert_to_lerobot.py` currently writes `observation.state` as 2-dim touch only, with no target column. This must be extended before any training. Not done in this task.
2. Train ACT on the converted dataset with a **session-level** train/eval split, written to a file before training (risk 1).
3. Action normalisation: the scaffold does not apply dataset stats. The trained checkpoint must carry stats and the wrapper must apply them. **Blocking** for any hardware test that uses real weights.

**Hardware tests (in order; see section 4 for commands).**

| Test | Measures | Pass (proposed, user to set) |
|---|---|---|
| H0 replay on Orin, `--device cuda`, no sends | ACT p50/p90 inference on Orin in the real schedule; target age; stall fraction | p90 inference <= 0.5 x chunk duration (chunk 10 = 167 ms); stall fraction <= 2% of replans excluding the first chunk |
| H3 live with trained checkpoint and `hold` | Serial path, 30 Hz send, plate holds under the fast layer | 0 out-of-range lines sent (hard gate); no stop from `consecutive_rejects`; plate stays inside platform |
| H4 live with `go_green`, grounding running | End-to-end target reach | See section 3 |

## 2. Option B: SMOLVLA_DIRECT

**What it is.** Pretrained `lerobot/smolvla_base`, pinned at `d9f33c94a60fb382c90dea2164c96845bd955e28`. It receives the instruction text ("go green"), the frame, and the state, and does its own grounding.

**Note on decision 1.** The spec says the fast layer never receives a colour word. SMOLVLA_DIRECT by design receives the instruction text, which is the one exception. Confirm this is intended.

**New versus pretrained.**
- Pretrained: the whole VLM and action head, at the pinned revision.
- New: the 6-to-3 output reduction (a placeholder, the first 3 of the padded action vector), the state padding (2 real dims padded to the checkpoint's state dim with zeros), and the single frame sent to every camera key.

**What must be trained.** Fine-tuning on the same converted dataset as option A, as a like-for-like comparison (skill point 3). The output reduction is an open decision and must be fixed before fine-tuning. Action normalisation must be checked against lerobot's pre/post processors. Not verified in this task.

**Hardware tests.**

| Test | Measures | Pass (proposed) |
|---|---|---|
| H5 replay on Orin, `--device cuda` | SmolVLA p50/p90 per chunk (bench measured 924 ms p50 at chunk 50, 1.08 Hz replan) | Chunk-50 p90 < 1667 ms (chunk duration) before it is used live. Otherwise stalls are structural |
| H6 live, after fine-tune | As H4 | As H4 |

## 3. Shared pass/fail criteria (all numbers user-changeable)

| Criterion | Proposed value | Basis |
|---|---|---|
| Out-of-range lines sent | 0 (hard gate, not tunable) | Firmware safety bound |
| Stall fraction (excluding first chunk) | <= 2% | Chunk must land before the previous one runs out |
| Target age p90 | <= 2000 ms | Matches `STALE_TARGET_S` in `bench_hybrid_qwen_act.py` |
| Task success | ball within 10 mm of target for >= 1 s, within a 10 s trial | Proposed; not from `EVALUATION_STRATEGY.md`, which must be checked |
| Trials per option before any ranking | >= 20, across >= 5 seeds for any trained model | Skill point 4 (a 3-seed ranking was overturned at 5) |
| Four standard metrics | Report steady-state error, settling time, control effort, task success via `evaluate_system_control.py` | No new metric invented |

**Metric input gap.** `evaluate_system_control.py` requires `target_x, target_y, touch_x, touch_y, theta_a/b/c` and a `host_timestamp_ms` column. Live mode writes `<out>.telemetry.csv` in that format. Column names were checked against the tool's `REQUIRED_COLUMNS` and `TIMESTAMP_CANDIDATES`.

## 4. Known limits (read before any run)

1. **Replay timing is not hardware timing.** It uses the session clock advanced by measured inference. Stall and target-age numbers are a model of the async loop.
2. **Target age grows between commands.** Marker targets are grounded once per command, and fixed targets are dated from the command. A 60 s replay of the first logged session with the zero stub gave a target-age median of about 4.9 s and a p90 of about 8.9 s. Periodic re-grounding is needed before a staleness pass can mean anything. Not implemented.
3. **Replan margin must exceed one tick.** The check runs once per 33 ms control tick. A margin below one tick forces stalls even with zero inference cost. The default is 40 ms. The first replay with a 20 ms margin gave a 21% stall fraction with a zero-cost stub; the 30 Hz grid plus 40 ms margin gave 1%.
4. **Rejected chunks cause replan churn.** A rejected chunk contributes no steps, so the next replan triggers immediately. Replay with a constant 12 deg stub produced 1797 rejected chunks in 60 s. Live mode stops after `--max-consecutive-rejects` (default 3).
5. **Touch frame.** Uplink touch is hundredths of a mm (`TouchProbe.h`) and is divided by 100 in `parse_uplink_line`. Replay uses the logged telemetry columns directly. `valid=0` samples (no ball) are counted but are not treated as a position.
6. **Stale touch readings** are used as logged (user decision). No masking.
7. **Environment.** `lerobot` 0.4.4 is installed in `ball_balance_env`, but `import lerobot.policies` fails in that interpreter (`groot/groot_n1.py` dataclass default-order error). ACT and SmolVLA therefore cannot be constructed on this Windows machine. The bench documents the Docker image `arm2-lerobot:r36.4.0` as the runtime. The replay of ACT was attempted and failed for this reason. The stub replay was run in its place.

## 5. Hardware test order and commands

Run only in a later session with the lab set up. None of these has been run. Paths assume the repo root. Replace `<PORT>` with the STM32 USB CDC device and `<HOMOGRAPHY>` with a saved 3x3 mm-to-px matrix. The firmware baud is 2000000.

**H0. Orin timing, no sends (ACT, random weights, no motors).**
```
python3 host_software/ml_jetson_vla/experiments/run_experiment.py --option act_fast_statemachine --mode replay --session host_software/data/01_bronze/session_jetson_track4_20260915_151627 --device cuda --chunk-len 10 --max-sim-seconds 60 --out results/h0_act_orin_replay.json
```

**H1. Serial path, no model, zero chunk (sends `A,0.0000,0.0000,0.0000` at 30 Hz).**
```
python3 host_software/ml_jetson_vla/experiments/run_experiment.py --option act_fast_statemachine --mode live --i-understand-this-drives-motors --stub-value 0 --serial-port <PORT> --baud 2000000 --homography-npy <HOMOGRAPHY> --instruction hold --max-run-s 5 --out results/h1_stub_zero_live.json
```
Expect 0 rejects. Expect a telemetry CSV at `results/h1_stub_zero_live.json.telemetry.csv`.

**H2. Reject path on hardware (stub 12 deg; nothing should be sent).**
```
python3 host_software/ml_jetson_vla/experiments/run_experiment.py --option act_fast_statemachine --mode live --i-understand-this-drives-motors --stub-value 12 --serial-port <PORT> --baud 2000000 --homography-npy <HOMOGRAPHY> --instruction hold --max-run-s 5 --out results/h2_stub_reject_live.json
```
Expect `stop_reason: consecutive_rejects` and `lines_sent: 0`.

**H3. Option S, trained ACT checkpoint, fixed target (after training and normalisation are in place).**
```
python3 host_software/ml_jetson_vla/experiments/run_experiment.py --option act_fast_statemachine --mode live --i-understand-this-drives-motors --checkpoint <ACT_CKPT_DIR> --serial-port <PORT> --baud 2000000 --homography-npy <HOMOGRAPHY> --instruction hold --max-run-s 10 --out results/h3_act_hold_live.json
```

**H4. Option S, colour target (needs the Qwen backend on the Jetson).**
```
python3 host_software/ml_jetson_vla/experiments/run_experiment.py --option act_fast_statemachine --mode live --i-understand-this-drives-motors --checkpoint <ACT_CKPT_DIR> --serial-port <PORT> --baud 2000000 --homography-npy <HOMOGRAPHY> --instruction go_green --max-run-s 10 --out results/h4_act_green_live.json
```

**H5. SmolVLA replay timing on Orin (after the output reduction is decided).**
```
python3 host_software/ml_jetson_vla/experiments/run_experiment.py --option smolvla_direct --mode replay --session host_software/data/01_bronze/session_jetson_track4_20260915_151627 --device cuda --max-sim-seconds 60 --out results/h5_smolvla_orin_replay.json
```

**Option L (`act_fast_llm`):** no hardware test. It refuses (exit 3) until a parser is wired in.

## 6. Still a placeholder

- ACT weights: random unless `--checkpoint` is given. Every output JSON carries `weights_status`.
- SmolVLA output: 6-to-3 placeholder reduction. State padding placeholder. No fine-tune.
- Action normalisation: not applied in either layer.
- Option L: stub only.
- Audio command: not wired. Live takes a single `--instruction`.
- Converter: no target column yet.
- Grounding in live: once per command, not periodic.
- Live loop: written, not run. Tested only through its parts (serial format, uplink parse, scheduler).
