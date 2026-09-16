# Multi-Head Qwen2.5-VL-3B Architecture Spec (Arm 2) — 2026-09-15

**Status: concrete architecture design, no training/firmware code.** Takes
`MULTI_HEAD_OUTPUT_DESIGN.md`'s recommendation (grounding head + state-fusion head +
action-chunk regression head) from concept to real layer shapes, parameter counts,
attachment points, and a data-flow contract against `convert_to_lerobot.py`'s actual,
already-built feature schema — per the `model-iteration-constraints` skill's requirements
for designing a new model architecture. Everything below was checked against real files in
this repo (`config.json`, `vla_architecture.py`, `dataset.py`, `RLControl.cpp`,
`session_recorder.py`, `motor_geometry.py`, `convert_to_lerobot.py`), not asserted from
memory of Qwen2.5-VL's public spec — where a number is derived rather than read verbatim,
that's stated explicitly.

Per this task's scope: no training code, no Colab notebook, no PPO/BC loop, no firmware
changes, no `PolicyCommand` edit. Those are items 4/5/6, not this doc.

## 0. What the `model-iteration-constraints` skill flagged, and how it shaped this design

Read in full before designing (see skill body). Four of its eight points materially changed
what's below, not just background reading:

1. **"Read locked constraints first."** `CLAUDE.md`'s FPGA BRAM (612.5KB) / DSP (220 slices)
   limits do **not** bind this design — `CLAUDE.md` explicitly carves the Jetson comparison
   arms out of the 120Hz/FPGA latency requirement, and arm 2 never touches the FPGA. The
   constraint that *does* bind is `ARCHITECTURE.md`'s own stated Jetson ceiling: **~4B params,
   INT4-quantized, via MLC-LLM/TensorRT-LLM**. Qwen2.5-VL-3B computes out to ~3.74B params
   (§1 below) — under that ceiling only if quantization actually happens; the checkpoint on
   disk is BF16-native (`config.json`: `"torch_dtype": "bfloat16"`), and `deployment/` is
   still "a scaffold — there is no model to point it at yet" per `ARCHITECTURE.md`. This
   design treats INT4 deployment as unresolved infrastructure, not a solved prerequisite —
   see §6.
2. **"Check hardware-deployment compatibility before investing training time."** This is the
   single biggest risk this design surfaces (§6): `deployment/`'s stated toolchain
   (MLC-LLM/TensorRT-LLM) is built around serving a stock HF causal-LM `generate()` graph.
   Bolting a custom action-query token + regression head onto Qwen's hidden states is a
   different graph shape than either toolchain's default text-generation path expects. This
   project has hit exactly this failure mode before (`ml_system_parameter_budget.md` §5.8 —
   GELU had no hls4ml support, discovered by reading the converter source, not assumed) —
   the fix pattern there (check the real converter/runtime source before committing) applies
   here too. **Recommend a scoped feasibility check of custom-head export through the actual
   chosen toolchain before BC training starts**, not after a checkpoint exists.
3. **"Isolate the variable you're actually testing."** Arm 2 vs. Arm 1 already differs in
   hardware-platform-adjacent ways the project is actively controlling for (deploying arm 1
   to the Jetson too, per `ARCHITECTURE.md`). This design adds a second axis that needs the
   same discipline: any future regression-head-vs-flow-matching-head ablation (plan §3.3's
   phase-2 stretch goal) must hold the BC training data fixed across both, exactly the
   small-vs-medium-vision-class trap this project already has one open instance of. Recorded
   as an explicit constraint on any future Head C variant, §7.
4. **"Never rank from a small seed count."** Directly shapes the chunk-size-N
   recommendation in §5.4 below: this project's own history (a 3-seed vision ranking flipped
   twice, once by an unrelated shared hyperparameter) means N cannot be picked from one or
   two trial runs and then locked in. §5.4 gives a starting range, not a chosen value.

Points 5-8 (evaluation isolation, dtype hygiene, export hygiene, versioning) are addressed
inline where they apply (§4.3, §6, §8) rather than repeated here.

## 1. Backbone: Qwen2.5-VL-3B, real dimensions from the local checkpoint config

Read directly from `host_software/ml_jetson_vla/models/qwen2_5_vl_3b_instruct/config.json`
(present on disk, not guessed):

| Component | Dimension | Value |
|---|---|---|
| Language backbone hidden size | `hidden_size` | 2048 |
| Language backbone layers | `num_hidden_layers` | 36 |
| Language backbone attention heads / KV heads | `num_attention_heads` / `num_key_value_heads` | 16 / 2 (GQA) |
| Language backbone MLP intermediate size | `intermediate_size` | 11008 |
| Vocab size (tied embed/lm_head) | `vocab_size` | 151936 |
| Vision encoder hidden size | `vision_config.hidden_size` | 1280 |
| Vision encoder depth | `vision_config.depth` | 32 |
| Vision encoder intermediate size | `vision_config.intermediate_size` | 3420 |
| Vision→language projection ("merger") output | `vision_config.out_hidden_size` | 2048 |
| Patch / spatial merge | `patch_size=14`, `spatial_merge_size=2` | — |
| Native checkpoint dtype | `torch_dtype` | bfloat16 |

**Total parameter count is computed here, not read from a params file (none present
locally)** — using standard Qwen2/SwiGLU-MLP + GQA-attention counting formulas against the
dimensions above:

- LM per-layer: attention (q/k/v/o with GQA: `2048×2048 + 2048×256 + 2048×256 + 2048×2048`
  ≈ 9.4M) + SwiGLU MLP (`3 × 2048×11008` ≈ 67.6M) ≈ **77.1M/layer × 36 layers ≈ 2.78B**
- Tied embedding/`lm_head`: `151936 × 2048` ≈ **0.31B** (counted once, tied)
- Vision tower per-block: attention (`~6.55M`) + SwiGLU MLP (`3 × 1280×3420` ≈ 13.1M) ≈
  **19.7M/block × 32 blocks ≈ 0.63B**, plus a small patch-embed (~1.5M) and merger
  projection (`1280×4 → 2048`, ~14.7M)
- **Total ≈ 3.09B (LM) + 0.65B (vision) ≈ 3.74B**, consistent with Qwen's own public "3.75B"
  figure for this checkpoint — cross-checked, not the source of truth. Treat the exact digit
  as approximate; if an exact count is ever needed, get it from `sum(p.numel() for p in
  model.parameters())` on the loaded checkpoint, not this document.

**Grounding head (Head A) = this backbone used as-is.** Native VLM path: vision tower →
merger → interleaved with text token embeddings → 36-layer decoder → tied `lm_head`. No new
engineering; this is what `qwen_vl_smoke_test.py` already exercises end-to-end
(`model.generate(**inputs)` → decoded string), confirmed working code, not a plan.

**Fine-tuning mode: frozen backbone + LoRA, not full fine-tune.** Two independent reasons,
both concrete:
- Preserves the pretrained grounding capability that is arm 2's entire reason for existing
  over a from-scratch model (plan §3.1's framing, restated because it drives this choice
  directly).
- **Resource math, not just a preference.** Full fine-tuning 3.74B params in FP32 with Adam
  needs weights (4B×4B ≈ 15GB) + gradients (15GB) + two Adam moment buffers (30GB) ≈ **~60GB
  just for optimizer state**, before activations — at or past the Jetson AGX Orin 64GB's
  entire *unified* (CPU+GPU shared) memory budget, leaving nothing for activations, the
  vision tower's own KV/attention buffers, or the OS. LoRA changes this math by roughly two
  orders of magnitude (§3 below) and is the only version of "fine-tune Qwen on this hardware"
  that's arithmetically plausible without a separate training box — but see §6 for the
  unresolved question of whether training even happens on-device at all.

## 2. Head B — State fusion

### 2.1 What the plan recommended vs. what `convert_to_lerobot.py` actually built (real gap, found by reading both)

`MULTI_HEAD_OUTPUT_DESIGN.md` §3.2 recommends reusing `RLControl.cpp`'s proven 9-dim state
contract (`[ball_x, ball_y, x_error, y_error, filtered_vel_x, filtered_vel_y,
actual_step_A/B/C]`). But the **already-built, already-real** `convert_to_lerobot.py`
(`_build_features()`, read this session) defines `observation.state` as:

```python
"observation.state": {"dtype": "float32", "shape": (2,), "names": ["touch_x_mm", "touch_y_mm"]}
```

Two dims, not nine — just ball position, no error term, no velocity, no motor-angle
feedback. This is a real, concrete interface gap between the plan and the actual data
pipeline that Task 3 (this doc) is scoped to design against, in the same spirit as
`MULTI_HEAD_OUTPUT_DESIGN.md` §4's `PolicyCommand` finding — surfaced, not fixed here (fixing
it means editing `convert_to_lerobot.py`, outside this task's scope).

**What's actually recoverable without new hardware capture, if the converter is later
extended** (all raw material already exists in `telemetry.csv`, read directly this session
via `session_recorder.py`'s `TELEMETRY_CSV_FIELDS`):
- `target_x`, `target_y` are already columns → error terms (`touch_x - target_x`, etc.) are
  a pure derived computation, no new capture.
- `host_timestamp_ms` is already a column → finite-differenced velocity is computable, but
  requires the converter to become a **stateful, per-session windowed** transform (needs the
  previous frame's `touch_x/y` and timestamp) instead of today's stateless per-row mapper —
  a real implementation change, not free.
- `theta_a`, `theta_b`, `theta_c` **are already captured per frame** — confirmed by reading
  `run_jetson_standalone.py` (line ~570): `theta_a = steps_to_angle(touch_snapshot["motor_a"])`,
  where `motor_a` comes from the `T,...` uplink's `motorA.currentPosition()` — i.e. this is
  **real measured/actual (lag-affected) motor position feedback**, the same conceptual
  quantity as `RLControl.cpp`'s `actual_steps`, just in degrees instead of raw steps
  (`motor_geometry.py`'s `steps_to_angle`, confirmed a plain `0.1125°/step` linear map, no
  origin offset needed — see that module's docstring). **But it currently only lands in the
  `action` feature, not `observation.state`.**

**Non-obvious catch if Phase 2 duplicates `theta_a/b/c` into state, flagged here because it's
easy to get subtly wrong:** `theta_a/b/c` at frame *t* is the actual measured angle at *t* —
if that same value is used both as a component of `state(t)` (Head B's input) and, un-shifted,
as the ground-truth label for `action` at the *same* frame *t*, action-chunking must still
only ever target *strictly future* frames relative to the state used to predict them
(chunk = `[t+1 .. t+N]`, never including `t` itself). This isn't literally a leak under that
rule, but it's the kind of off-by-one that's easy to get wrong when wiring a windowed dataset
loader, and worth a unit test rather than an assumption once Phase 2 is built.

### 2.2 Recommended design: build Phase 1 now, extend to Phase 2 only if BC training shows the 2-dim state is insufficient

This matches this project's own repeated "start simple, escalate only if needed" pattern
(cited in the plan: small vision class before medium, BC before RL). Concretely:

**Phase 1 (buildable today, matches `convert_to_lerobot.py` exactly):**

```
state_dim = 2   # [touch_x_mm, touch_y_mm]
Head B = nn.Sequential(nn.Linear(2, 2048), nn.LayerNorm(2048))
```

Params: `Linear(2,2048)` = `2×2048 + 2048 (bias)` = 6,144, plus `LayerNorm(2048)`'s own
affine `weight`/`bias` = `2×2048` = 4,096 → **10,240 total**. `LayerNorm` (not present in
`RT1LiteVLA`'s `state_embed`, added here) because this projects directly into Qwen's own
hidden space, which is layer-normed throughout the backbone — an un-normalized raw `Linear`
output at Qwen's scale is a plausible source of training instability that `RT1LiteVLA` never
had to deal with (it projects into its own from-scratch 512-dim Transformer, not a
pretrained one).

**Phase 2 (if needed, requires `convert_to_lerobot.py` changes not in scope here):**

```
state_dim = 9   # [touch_x, touch_y, err_x, err_y, vel_x, vel_y, theta_a, theta_b, theta_c]
Head B = nn.Sequential(nn.Linear(9, 2048), nn.LayerNorm(2048))
```

Params: `9×2048 + 2048` = **20,480**. Note this is *not* unit-identical to `RLControl.cpp`'s
own 9-dim vector — the motor-position term is degrees here (`theta_a/b/c`) vs. raw step
counts there — so it's the same conceptual slots, not a byte-identical port. Feature scale
differences across dims (mm-range position/error, mm/s-range velocity, ~±10°-range angle) are
a real reason to standardize each input feature (z-score against training-set statistics)
before this `Linear`, not just feed raw units into a freshly-initialized layer.

## 3. Head C — Action-chunk regression head

### 3.1 Attachment point

A single trainable **action-query embedding** (`nn.Parameter`, shape `[2048]`) is appended to
Qwen's input sequence (after the vision + text tokens, before the position where generation
would normally start), mirroring `RT1LiteVLA`'s own "dedicated token → pool its final hidden
state" pattern (`vla_architecture.py`'s `state_embed` token at sequence position 0, whose
transformer output is what `action_head` actually consumes — read directly, not
paraphrased). After one forward pass through the 36-layer backbone, this token's final
hidden state (`[B, 2048]`) is pulled out and used as Head C's vision+language input,
alongside Head B's `[B, 2048]` state embedding.

**Why a dedicated token over pooling the last text-position hidden state:** the last text
token's hidden state is optimized (via pretraining) to predict the *next vocabulary token*,
not to summarize the scene for a downstream regression head — a fresh learned query token
gives the fine-tuning process a representation slot with no pretrained bias toward next-token
prediction to fight against. This mirrors why `RT1LiteVLA` uses a dedicated state token rather
than pooling the ResNet feature map directly.

### 3.2 Internal shape — plain regression, reusing `RT1LiteVLA.action_head`'s pattern, chunked

Per the plan §3.3's explicit recommendation (reuse the working pattern, not flow-matching,
until proven inadequate):

```
fused = concat([pooled_query_hidden (2048), head_b_state_embed (2048)])   # [B, 4096]

Head C = nn.Sequential(
    nn.Linear(4096, 512), nn.ReLU(),   # new: down-project Qwen-scale fusion to RT1LiteVLA's scale
    nn.Linear(512, 128),   nn.ReLU(),  # == RT1LiteVLA.action_head layer 1, verbatim shape
    nn.Linear(128, N * 3),             # == RT1LiteVLA.action_head layer 2, action_dim generalized to N*3
)
# reshape [B, N*3] -> [B, N, 3]
# bounding (new, see §3.3): output = tanh(raw) * ANGLE_LIMIT_DEG
```

The first `Linear(4096, 512)` is new — `RT1LiteVLA.action_head` was never designed to
consume a 4096-dim input; everything after it is `RT1LiteVLA.action_head`'s exact,
already-trained shape (`ml_multimodal/core/vla_architecture.py` lines 62-65), just with the
final layer's output width generalized from `action_dim=3` to `N*3` for chunking.

**Param count** (worked example at `N=10`, a placeholder pending §5.4's tuning, not a final
choice):

| Layer | Shape | Params |
|---|---|---|
| Fusion down-project | `Linear(4096, 512)` | 2,097,664 |
| Action-head layer 1 (reused pattern) | `Linear(512, 128)` | 65,664 |
| Action-head layer 2 (reused pattern, chunked) | `Linear(128, 30)` | 3,870 |
| Action-query token | `nn.Parameter([2048])` | 2,048 |
| **Head C + query token total** | | **≈2,169,246** |

Combined with Head B Phase 1 (10,240, including `LayerNorm`'s affine params — see §2.2):
**2,179,486 new trainable params** (confirmed by running `qwen_multihead_policy.py`'s
`self_test()` against the actual module, not hand-arithmetic alone — the hand-computed
figure this doc first carried was off by 4,096 params from forgetting `LayerNorm`'s own
`weight`/`bias`, caught by that run). ~0.058% of the 3.74B backbone — genuinely small,
consistent with the plan's framing, and cheap enough to train on Jetson-class hardware even
before considering LoRA.

### 3.3 Output bounding — resolves plan §5 open question 3

The plan flagged that `RT1LiteVLA.action_head` has no bounding activation at all (raw linear
output), unlike `RLControl.cpp`, which clips to `[-1,1]` then scales by `MAX_MOTOR_STEP=98`
steps. **Recommendation: bound Head C's output with `tanh(x) * ANGLE_LIMIT_DEG`.**

`ANGLE_LIMIT_DEG` starting value: **11.025°**, derived directly from `RLControl.cpp`'s
`MAX_MOTOR_STEP = 98.0f` via `motor_geometry.py`'s confirmed linear map
(`steps_to_angle(98) = 0.1125°/step × 98 = 11.025°`) — i.e. arm 1's own proven, currently-
running operating envelope, converted to degrees. **Flagged explicitly: this is arm 1's
*validated operating range*, not necessarily the platform's full mechanical limit** — it may
be more conservative than what the hardware can safely do. Confirm against
`docs/PROJECT_LOGBOOK.md` / the Electrical Engineer before treating it as a hard bound rather
than a reasonable, cited starting point.

### 3.4 Discretized-action-token alternative

Not re-litigated here — plan §3.4 already considered and deprioritized RT-2/OpenVLA-style
autoregressive token-decoded actions for exactly the same "few tokens/sec" reason the
project's own research spike ruled out single-rate monolithic VLAs. Nothing in this doc's
deeper layer-shape work changes that reasoning.

## 4. Data flow — from `convert_to_lerobot.py`'s schema to a forward pass

```
LeRobotDataset row (per frame):
  observation.image : uint8 RGB (H, W, 3)   -- decoded from rgb_video.mp4
  observation.state : float32 (2,)          -- [touch_x_mm, touch_y_mm]   (Phase 1)
  action             : float32 (3,)          -- [theta_a_deg, theta_b_deg, theta_c_deg]
  task               : str                   -- real recognized audio command, or "(no command yet)"
       |
       v
Qwen processor (EXACT call shape already exercised by qwen_vl_smoke_test.py, real code):
  processor.apply_chat_template([{role:"user", content:[{type:"image", image:...},
                                                          {type:"text", text: task}]}], ...)
  process_vision_info(...) -> image_inputs
  processor(text=[...], images=image_inputs, ..., return_tensors="pt")
       -> input_ids, attention_mask, pixel_values, image_grid_thw
       |
       v
[+] append action-query embedding (2048,) to the input embedding sequence
       |
       v
Qwen2_5_VLForConditionalGeneration backbone (frozen + LoRA, §1/§6)
       |
       +--> Head A: standard lm_head path over text positions (unchanged, native) --
       |    used for logging/debug only in Phase 1 (plan §3.1(a); (b) is a separate
       |    scoping decision, not made here)
       |
       +--> pooled_query_hidden = last_hidden_state[:, query_token_position, :]   [B, 2048]
                    |
                    +---(concat)---+
                                   |
observation.state --> Head B -----+--> fused [B, 4096] --> Head C --> [B, N, 3] theta_a/b/c chunk
   [B, 2]  (Phase 1)   [B, 2048]
```

**Chunk formation is NOT free from the schema as stored on disk** — the LeRobot parquet
holds one `action` row of shape `(3,)` per frame, not pre-chunked `(N,3)` windows.
**Unverified assumption, flagged explicitly per this project's documentation convention:**
consuming a temporal window per training sample is expected to use LeRobotDataset's
`delta_timestamps` mechanism (the standard way ACT/diffusion-policy-style chunked policies
are trained against `lerobot` datasets) — but this was **not** verified against the
installed `lerobot==0.4.4` source the way `convert_to_lerobot.py`'s author verified
`add_frame()`/`create()`/`finalize()` (via `inspect.signature`/`inspect.getsource`, per that
script's own docstring). Confirm `delta_timestamps`' real signature against the installed
version before building the training-side data loader — don't assume it from general
`lerobot` familiarity, matching the standard this project has already held itself to once on
this exact file.

## 5. Open questions and unconfirmed assumptions (explicit, not settled)

1. **SmolVLA vs. Qwen2.5-VL-3B** — still unresolved, carried over from
   `MULTI_HEAD_OUTPUT_DESIGN.md` §0. This doc designs against Qwen2.5-VL-3B per that doc's
   stated operative-decision framing, but the user has not confirmed it.
2. **Custom-head deployment compatibility through MLC-LLM/TensorRT-LLM** — §0 point 2 /
   §6. The single biggest unverified assumption in this design. Needs a scoped smoke test,
   not assumed to work post-training.
3. **INT4 quantization of a checkpoint carrying two new custom heads** — does the
   quantization pipeline treat the new `Linear` heads the same way it treats backbone
   weights, or do they need to stay FP16/BF16 while only the backbone quantizes? Not
   researched here.
4. **Chunk size `N`** — a starting experimental range, not a value: this project's real
   control/telemetry rate is 30Hz (`session_recorder.py`'s default `fps=30`, matching
   `RLControl.cpp`'s `CONTROL_DT` = 1/30s exactly) and the settling-time metric
   (`EVALUATION_STRATEGY.md`) requires re-entering a 20mm radius and holding it for 500ms.
   A chunk of `N=10` frames at 30Hz ≈ 333ms — comparable to, but under, that 500ms settle
   window, offered as a **starting point for experimentation** (~5-15 frame range), not
   SmolVLA's uncritiqued default of ~50 (which the plan already flagged as not derived from
   this project's own requirements). Per skill point 4: needs ≥5 seeds before any N choice is
   treated as decided.
5. **Phase 1 vs. Phase 2 state dim** — recommend building Phase 1 (2-dim, matches the real
   schema today) first; escalate to Phase 2 (9-dim-equivalent, requires
   `convert_to_lerobot.py` changes) only if BC training shows the 2-dim state is
   insufficient. Not decided which will actually be needed.
6. **Whether `theta_a/b/c` is best understood as lag-affected feedback or a cleaner
   target** — `motor_geometry.py`'s docstring and `run_jetson_standalone.py`'s call site
   both point to "real measured position," analogous to `RLControl.cpp`'s `actual_steps`,
   but this hasn't been confirmed against the live control loop's actual read/command timing
   the way `RLControl.cpp`'s own design notes explicitly call out the lag as load-bearing.
   Worth confirming before treating the BC imitation target as clean ground truth.
7. **LoRA rank/depth** (`r=16`, last 12 of 36 LM layers, ~9.98M adapter params by the same
   counting method as §1 — worked as: attention ≈204.8K/layer + MLP ≈626.7K/layer ≈
   831.5K/layer × 12 layers) is a reasoned starting point based on standard LoRA-efficiency
   practice, not empirically tuned against this task.
8. **Where fine-tuning actually runs** — on-Jetson (unified 64GB memory, LoRA makes this
   plausible per §1's math) vs. off-device on a separate GPU box with only quantized
   inference deployed to the Jetson. Neither `ARCHITECTURE.md` nor
   `LARGE_VLA_RESEARCH_SPIKE.md` decides this for a Qwen backbone specifically (the prior
   SmolVLA framing's "LoRA/Colab fine-tuning" plan is noted there as no longer applying as
   written). Affects nothing in this doc's shapes, but affects items 4/5's actual training
   setup.
9. **Head A joint-vs-frozen training** (plan §5 open question 5) — this design defaults to
   frozen-backbone-plus-LoRA with Head A used for logging only (§3.1(a)), deferring the
   joint multi-task question entirely; not resolved here.
10. **Any future Head C variant (e.g. a flow-matching upgrade) must be trained on identical
    data to the regression head it's compared against** before attributing any accuracy
    delta to the head architecture — per skill point 3, restated here because it's the most
    likely place this project would otherwise repeat its existing small-vs-medium-vision
    data-confound situation.

## 6. Deployment-compatibility risk (skill point 2 — check before training investment)

`ARCHITECTURE.md`'s `deployment/` directory is described as a "quantization/export pipeline
(INT4, MLC-LLM or TensorRT-LLM)... currently a scaffold." Both of those toolchains are built
primarily to serve a stock HF causal-LM `generate()` graph efficiently — that is exactly what
`qwen_vl_smoke_test.py` exercises today (`model.generate(**inputs)`), and exactly what this
design's Head A still does. **Heads B and C are not part of that graph.** They need Qwen's
*hidden states* pulled out mid-forward-pass (not the sampled/decoded output tokens), run
through new `Linear` layers, on every control tick — a materially different serving pattern
than "run the LLM, decode text." Whether MLC-LLM/TensorRT-LLM's Python runtime bindings
expose per-token hidden states cleanly enough to splice in a custom head, or whether this
forces a hand-rolled inference path (losing whatever throughput benefit the quantized runtime
was providing), is **not known and not researched in this session.** Recommend a small,
scoped feasibility spike — export/run just the "get hidden states out, feed a dummy
`Linear`" path through whichever toolchain is chosen — *before* sinking BC training time into
the full multi-head checkpoint, matching this skill's explicit "check before training time,
not after a checkpoint exists" guidance and this project's own precedent
(`ml_system_parameter_budget.md` §5.8: the GELU/hls4ml gap was found by reading the real
converter source before committing to an architecture, not discovered post-hoc).

**Dtype note (skill point 6):** `qwen_vl_smoke_test.py` loads the model with
`torch_dtype=torch.float16` explicitly, but `config.json` declares the checkpoint's native
`torch_dtype` as `bfloat16`. This is a real, findable mismatch — bf16 and fp16 have different
exponent/mantissa splits, and forcing a bf16-trained 3.7B-param checkpoint into fp16 is a
plausible (not yet confirmed) source of activation overflow in the large-value regime typical
of big LLMs. Whichever dtype the multi-head checkpoint trains in, Heads B/C's new `Linear`
layers must be created in that same dtype (or explicitly cast at the fusion point) rather than
silently defaulting to `nn.Linear`'s FP32 default while the backbone runs in FP16/BF16 — the
exact "new derived computation, dtype not checked explicitly" pattern this skill's item 6
warns about. Not fixed here (this doc's scope is the design, not the smoke test), but flagged
since it directly informs Heads B/C's own dtype handling.

## 7. Evaluation-isolation note (skill point 5)

Once BC training exists (items 4/5, not this task), held-out evaluation for arm 2 must be
**session-level**, not frame-level — a Track 4 bronze session is one continuous
`rgb_video.mp4`, and adjacent frames within it are highly correlated (same lighting, same
approach trajectory). A frame-level train/val split would put near-duplicate frames on both
sides of the split, an easy way to silently leak. `dataset-integrity-check` is the skill to
run once real training data exists, to confirm this holds rather than assume it.

## 8. Versioning note (skill point 8)

The base checkpoint on disk is `models/qwen2_5_vl_3b_instruct/`. Any multi-head fine-tuned
checkpoint should land under a distinctly-versioned path (e.g.
`models/qwen2_5_vl_3b_multihead_v1/`) rather than overwriting the base checkpoint in place —
the base weights remain the reference point every future fine-tuning attempt is compared
against, mirroring how `ml_vision` treats `shared_vision_backbone_v1` as a fixed reference
rather than a mutable file.

## 9. Code skeleton

`host_software/ml_jetson_vla/core/qwen_multihead_policy.py` — module/class shapes matching
§1-§3 above, no forward-pass math beyond shape wiring, no loss function, no training loop.
Written to make the shapes in this doc checkable against real code, not as a deliverable
implementation.

## 10. What this doc does not do

Does not decide SmolVLA-vs-Qwen2.5-VL-3B (§5.1). Does not modify `convert_to_lerobot.py`,
`PolicyCommand`, or any firmware. Does not write BC/RL training code (items 4/5). Does not
run the §6 deployment feasibility spike it recommends — that's flagged as necessary future
work, not performed here.
