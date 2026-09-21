# Arm 2 Minimal-Baseline Scope — Effort/Generality, Not Peak Performance (2026-09-18)

**2026-09-18, later same day — see `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` for the
follow-up that extends this doc's Qwen-only scope to real code supporting multiple
general-purpose VLM candidates (`core/vlm_backends.py`, `core/minimal_vlm_policy.py`,
per-candidate smoke tests, an offline scorer against real Track 4 data, and a real executed
verification run).** That doc does not change anything below -- the firmware gap (§3), the
specialization-track parking (§5), and the minimal-vs-specialization distinction all still
apply exactly as stated here. It only widens "which model" while keeping "how" identical.

**Status: scoping/documentation only. No implementation code. No firmware changes.** This
doc records a real strategic correction the user made to today's Arm 2 work and exists so
the next person (or the next session) doesn't re-derive, misfile, or accidentally undo it.
Read this before touching `MULTI_HEAD_OUTPUT_DESIGN.md`, `MULTI_HEAD_ARCHITECTURE_SPEC.md`,
`core/qwen_multihead_policy.py`, `ACTION_CHUNK_CONTROL_SUBSYSTEM_BOOTSTRAP.md`, or the
Stage 0/1 fine-tuning data-staging pipeline — all of that is real, useful, and **not what
this doc is asking for**.

## 1. The correction, precisely

The project's actual comparison goal for Arm 2 is **general-purpose, minimum-coding-effort
deployment** vs. Arm 1's **high-specialization, purpose-built small model** — an
**effort/generality axis**, not a peak-performance axis.

Fine-tuning/specializing the large model for this exact task is real, separate,
**secondary-track** work — useful infrastructure worth building eventually, but it does not
answer today's actual comparison question, and none of it should be presented as if it does.

Both arms are still judged, whatever their internal approach, on the project's standard four
metrics (`docs/EVALUATION_STRATEGY.md`: steady-state error, settling time, control effort,
task success rate — plus the 2026-09-18 extended metrics computed by the same
`compute_metrics()` in `host_software/evaluations/evaluate_system_control.py`). The
effort/generality axis is about *how each arm is built*, not a different scoring rubric —
both still get run through `evaluate_system_control.py --runs label=csv label=csv ...`
against the same pre-recorded audio sequence (`EVALUATION_STRATEGY.md`'s "Standardized
Evaluation Sequence"). The comparison's *point* is that Arm 1 gets there by heavy
specialization and Arm 2 gets there — or doesn't, or gets there slower/worse — with
essentially none. That gap, in whichever direction it falls, is the finding.

Most of today's prior Arm-2 work is real, useful, and **not this**:
`MULTI_HEAD_OUTPUT_DESIGN.md`, `MULTI_HEAD_ARCHITECTURE_SPEC.md`, `core/qwen_multihead_policy.py`,
`ACTION_CHUNK_CONTROL_SUBSYSTEM_BOOTSTRAP.md` (`core/action_chunk_bootstrap.py`), and the
Stage 0/1 fine-tuning data-staging pipeline are all specialization-track infrastructure —
kept, not deleted, but parked (§5) until the minimal baseline described here has actually
been run and reported.

## 2. The minimal-baseline architecture

**Qwen2.5-VL-3B, standalone, on the Jetson AGX Orin: own camera → prompt (image + current
instruction) → parsed structured text output → real motor angles.**

This bypasses Track 1's execution pipeline **entirely** — no reuse of the small-class vision
CNN, the `ml_audio` classifier, the Fusion state machine, or the RL-trained control net
(`RLControl.cpp`) for actuation, even in this "minimal effort" baseline. This is not a new
rule invented for this doc — it is `ARCHITECTURE.md`'s own original design for Arm 2, stated
directly (read in full 2026-09-18, not paraphrased):

> "**Jetson AGX Orin, large pretrained model** — an adapted version of the lab partner's
> Qwen-based model (NOT the small in-house `RT1LiteVLA` in `ml_multimodal/`), running
> **standalone** on the Jetson AGX Orin 64GB Developer Kit model: p3730: own camera, own
> inference, own control loop."
> — `ARCHITECTURE.md` §"The comparison being built," item 2

And its confound-avoidance rationale, also stated directly rather than reconstructed:

> "Standalone keeps this arm's development, and eventually its evaluation runs, fully
> decoupled from FPGA/optical-platform readiness."
> — `ARCHITECTURE.md` §"Why 'standalone' for the Jetson arm, not FPGA-relayed sensor data"

The same section's broader point — that arm 2 exists specifically so hardware platform
*isn't* the confound between it and arm 1 — extends directly to this correction: reusing any
piece of Track 1's execution pipeline would introduce a second, uncontrolled confound
(whose *architecture* actually handled the task) on top of the one the project has already
gone to the trouble of removing (hardware platform, by deploying arm 1 to the Jetson too).
A minimal-effort Arm 2 baseline that quietly leans on Arm 1's control net would not be
measuring "general-purpose, low-effort large-model deployment" at all — it would be
measuring Arm 1's control net with a slower target-picker bolted in front of it.

### Considered and rejected — do not re-propose

Earlier in this session, before this correction, a shortcut was proposed: have Qwen decide
*only* the target (in whatever coordinate space), and hand that target to Track 1's existing
RL control net for actuation. **The user explicitly rejected this.** It is not an
acceptable minimal baseline, for the reason above — it reuses an Arm-1 execution component
and reintroduces exactly the confound `ARCHITECTURE.md`'s standalone design was meant to
avoid. Recorded here so it doesn't resurface as a "simpler" fallback later.

## 3. The firmware dependency this creates (blocked — not resolved by this doc)

Qwen2.5-VL-3B has no native action output of any kind — this is `MULTI_HEAD_ARCHITECTURE_SPEC.md`'s
own finding (§0, point 1 area / the model-iteration-constraints discussion): it is "a plain
vision-language model — it generates text tokens (`model.generate(...)` → decoded string),
full stop." A minimal-effort baseline still needs *some* mechanism to get from that text
output to real motor angles.

**The honest minimum**: prompt Qwen to emit a structured text output (e.g. `theta_a/b/c`, or
a target position, in a parseable format) plus a small parser script. Explicitly **no
custom-trained heads, no fine-tuning** — that parser is real engineering, of a categorically
smaller kind than training custom heads, but it is still a real deliverable, not a triviality
to wave past.

That output has nowhere to go on the current robot. `firmware/stm32_ml_control_and_vision/BallBalancingBot/SerialCoords.cpp`
was read directly (2026-09-18, not assumed from memory) — its actual wire protocol, PC → MCU:

```
V,<seq>,<x_mm>,<y_mm>,<tgt_x_mm>,<tgt_y_mm>[,<valid>]   preferred
<x_mm>,<y_mm>,<tgt_x_mm>,<tgt_y_mm>[,<valid>]           legacy, still OK
<x_mm>,<y_mm>                                            legacy, target=(0,0)
```

This is `(ball_x, ball_y, target_x, target_y)` in millimetres — it is the input format
Track 1's expert pipeline (vision + audio + Fusion feeding the on-board RL control net)
sends. **There is no field, tag, or code path in `SerialCoords.cpp` today that accepts
`(theta_a, theta_b, theta_c)` or any other pre-computed motor-angle/step target from an
external source.** `handle_line()` parses exactly the four (or two) float fields above and
nothing else; anything else on the line is either ignored or treated as junk.

This is exactly what item 6 of today's earlier TODO list proposed, and which is **still
explicitly on hold pending the user's direct sign-off.** Until now that gap was assumed to
block only the specialization track (a custom-trained action head needs somewhere to send
its output too). **It does not — it equally blocks the minimal baseline described in §2.**
Whether Qwen's output is a fine-tuned regression head or a parsed line of prompted text, it
is still a motor-angle-shaped value with no accepting endpoint on the robot today.

**This doc does not resolve that gap and does not touch `SerialCoords.cpp` or any other
firmware file.** It states the dependency so the minimal Arm 2 baseline isn't reported as
"just needs a parser script" when it also needs a firmware decision that is explicitly the
user's to make, not an agent's.

## 4. Realistic expectations — report these, don't engineer around them

VLM text generation is slow. Prior research this session on OpenVLA/Qwen2.5-VL-class models
on Jetson-class hardware puts per-inference latency in the multi-second range — nothing like
Track 1's roughly 30Hz control loop (`SerialCoords.cpp`'s own comment: "At 30 Hz control
cadence, 150 ms = ~4 missed frames").

This is an **expected, reportable finding of the comparison itself** — it is exactly what
"minimum effort" costs, stated plainly rather than hidden. It is not a problem to solve by,
e.g., adding action chunking or a fast secondary control tier — doing that would cross back
into the specialization track (§5) and would defeat the point of measuring what a genuinely
low-effort deployment looks like. If the minimal baseline's control loop is slow and jittery
compared to Arm 1, that gap **is the comparison's answer**, not a defect in the baseline.

## 5. Specialization-track infrastructure — parked for later, not part of the minimal baseline

Real, useful, already built or in progress. Kept as-is. Not touched by this doc. Do not read
any of it as describing the minimal baseline in §2 — it describes a different, secondary
question (how well can this model be made to *specialize* for the task, given real training
investment).

| Artifact | What it is |
|---|---|
| `docs/MULTI_HEAD_OUTPUT_DESIGN.md` | Plan for a grounding head + state-fusion head + action-chunk regression head bolted onto Qwen2.5-VL-3B — synthesis/recommendation only, no code. |
| `docs/MULTI_HEAD_ARCHITECTURE_SPEC.md` | Concrete layer shapes/param counts/attachment points for the same three-head design, checked against real checkpoint config and `convert_to_lerobot.py`'s feature schema. |
| `core/qwen_multihead_policy.py` | Architecture skeleton implementing the three heads (Head A = Qwen's native VLM path, Head B = state fusion, Head C = action-chunk regression). Explicitly "NOT TRAINING CODE" per its own docstring — no loss function, optimizer, data loader, or PPO/BC loop. |
| `docs/ACTION_CHUNK_CONTROL_SUBSYSTEM_BOOTSTRAP.md` + `core/action_chunk_bootstrap.py` | Design + a real bootstrap run validating the action-chunking/open-loop-execution mechanism against all 10 Track 4 sessions. Reuses Head B/C shapes from the item above. No hardware, no firmware. |
| Stage 0 — `data_processing/convert_to_lerobot.py` (regime-tagged) | Builds the real LeRobotDataset (`observation.image`/`observation.state`/`action`/`task`) that any BC fine-tuning of the multi-head design would train against. |
| Stage 1 — `deployment/stage_finetune_data_for_colab.ps1` | Zips raw session video + uploads to Drive for Stage 2/3's (not-yet-built) Colab fine-tuning notebooks, extending the zip+Drive pattern from `bootstrap_model_comparison_colab.ipynb`. |

None of this is wasted — it is the right infrastructure for the secondary "how far can Arm 2
be specialized" question, once the primary effort/generality comparison in §2 has actually
been run. It just isn't what answers today's question, and shouldn't be cited as if it does.

## 6. What's genuinely still needed to build the minimal baseline

(a) **The firmware decision** (§3) — blocked, explicitly the user's call, not this agent's
    to make or act on.
(b) **A prompting scheme + parser script** — not yet written. This is the next real
    deliverable once (a) is unblocked. Not built as part of this task (documentation only,
    per scope).
(c) **Reuse of the already-working Qwen2.5-VL-3B-on-Jetson deployment** as the model-loading
    foundation — `deployment/qwen_vl_smoke_test.py` already confirmed this loads and runs on
    the real hardware. That part doesn't need to change; the minimal baseline's model-loading
    code should extend it rather than duplicate it, consistent with this project's standing
    "reuse existing tooling" convention.

## 7. Out of scope for this document

- No firmware changes (§3 is a dependency note, not an implementation).
- No implementation code for the prompting/parsing layer (§6b) — a future task builds it.
- No edits to any specialization-track doc/code in §5 — they stay exactly as they are.
