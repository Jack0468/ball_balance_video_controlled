# Multi-Head Output Design (Arm 2, Qwen2.5-VL-3B) — Plan, 2026-09-15

**Status: planning/synthesis only — no code, no firmware, nothing implemented here.** This
doc pulls together research that already exists across three other docs plus one real
source file, and states one concrete recommendation for item 3 (a separate follow-up task)
to build against. Where this doc asserts something as fact, it cites where that fact was
already established; where it's new synthesis, it says so explicitly.

**Why this is back on the table now:** arm 2's confirmed large-model candidate as of today's
session is **Qwen2.5-VL-3B-Instruct** (smoke-tested at
`host_software/ml_jetson_vla/deployment/qwen_vl_smoke_test.py`), a plain vision-language
model — it generates text tokens (`model.generate(...)` → decoded string), full stop. It has
no continuous action output of any kind. Something has to attach an action-producing path to
it. That "something" is a multi-head design.

## 0. Open inconsistency this doc found, not resolved here

`LARGE_VLA_RESEARCH_SPIKE.md`'s newest section (2026-09-15, "dual-tier framing replaced")
recommends **SmolVLA** as "the confirmed large-model candidate" and does not mention
Qwen2.5-VL-3B as a backbone choice anywhere in that section. Per this task's own routing
(today, later in the same day), the user has since redirected arm 2 toward Qwen2.5-VL-3B
specifically — a *different* model than SmolVLA, not a synonym for it. `ARCHITECTURE.md`
and `LARGE_VLA_RESEARCH_SPIKE.md` as currently written do not reflect this second pivot.
This doc treats the Qwen2.5-VL-3B redirect as the operative decision (per direct task
routing) and designs against it, but flags that **`LARGE_VLA_RESEARCH_SPIKE.md` itself now
needs its own dated addendum reconciling "SmolVLA, action-chunking, single model" with
"Qwen2.5-VL-3B, plain VLM, no native action head" — not done as part of this task**, since
this task's scope is the multi-head plan, not editing that doc's decision record.

One plausible reconciliation, stated here as a hypothesis and not confirmed: Qwen2.5-VL-3B
is being evaluated as a closer architectural proxy for the lab partner's actual model (which
`CLAUDE.md`/`ARCHITECTURE.md` both describe as "Qwen-derived") than SmolVLA ever was, since
SmolVLA's SigLIP+SmolLM2 stack has no relationship to Qwen at all. If that's the real reason,
SmolVLA may still be a live fallback rather than fully superseded — worth confirming with the
user rather than assumed.

## 1. What the existing research already establishes (synthesis of prior work, cited)

1. **A two-role split (slow grounding + fast action) is the established shape for this class
   of problem**, not a novel idea being proposed here. `LARGE_VLA_RESEARCH_SPIKE.md` cites
   NVIDIA Isaac GR00T N1's production System-2 (VLM grounding) / System-1 (diffusion-
   transformer action) split, LiteVLA-H's measured dual-rate numbers on our exact hardware
   class, and FASTER's theoretical argument that a single slow monolithic VLA cannot hit
   real-time control rates.
2. **The split doesn't require two separately-trained models.** The 2026-09-15 revision in
   that same doc found that *action chunking* — one forward pass predicts a short sequence of
   N future actions, executed open-loop across N control ticks — already decouples the
   control-tick rate from the backbone's own slow inference latency, which is what the
   dual-tier split existed to solve in the first place. SmolVLA, π0, and ACT all ship this
   way. **This finding is backbone-agnostic** — nothing about it depends on SmolVLA's
   specific SigLIP+SmolLM2 stack, so it carries over unchanged to a Qwen2.5-VL-3B backbone.
3. **π0/SmolVLA's own architecture, read carefully, already *is* a multi-head design** — this
   is the load-bearing synthesis point of this doc, and it wasn't stated explicitly as such in
   `LARGE_VLA_RESEARCH_SPIKE.md`. SmolVLA = SigLIP vision encoder + SmolLM2-135M language
   backbone (a head/path that understands the instruction+scene) + a **separate flow-matching
   action head** attached on top, producing the action chunk. That is exactly the shape this
   task asks for, just never labeled "multi-head" in the existing doc. The concrete
   recommendation below is this same shape, re-pointed at Qwen2.5-VL-3B in place of SmolVLA's
   own vision-language stack.
4. **The terminal action-head output convention is already decided, not open**: `theta_a/b/c`
   motor angles in degrees, not raw step-space. Confirmed two independent ways:
   - `ARCHITECTURE.md` states explicitly that arm 2/3's output is expected to be
     `theta_a/b/c`, needing `angle_to_steps()`
     (`firmware/stm32_ml_control_and_vision/BallBalancingBot/MotorControl.h`) downstream —
     unlike arm 1's control net, which needs no conversion because it was trained directly on
     raw step-space targets.
   - Read directly from code this session: `host_software/ml_multimodal/core/dataset.py`
     (`VLADataset.__getitem__`, lines 47-55) already builds its action tensor from
     `action_theta_a/b/c` fields, and `host_software/ml_multimodal/core/vla_architecture.py`'s
     `RT1LiteVLA.action_head` (lines 62-65) is already trained to regress exactly that
     3-vector. The in-house VLA already treats `theta_a/b/c` as the terminal output — the new
     design should match this, not invent a different convention.
5. **A concrete, working, single-head action-regression pattern already exists in this
   codebase** and is directly reusable as the *internal shape* of the new action head, even
   though `RT1LiteVLA` itself is not a multi-head design (see §2 caveat below):
   `FiLM(vision, language) → concat state token → 2-layer Transformer → pool → Linear(512,128)
   → ReLU → Linear(128, action_dim)`. This is real, already-trained
   (`ml_multimodal/models/vla_v1/best_vla.pth`) code, not a proposal.
6. **The LeRobot data schema this track already committed to (Task 2/3 of the research spike)
   carries a proprioceptive state vector alongside the action vector per frame**
   (`data/chunk-*/*.parquet`: "state vector, action vector" per timestep,
   `LARGE_VLA_RESEARCH_SPIKE.md` §"Data conversion to LeRobot format"). This confirms the
   action head should consume a state input, not vision+language alone — matching how
   π0/SmolVLA's own action expert is documented to work upstream.
7. **Arm 1's `RLControl.cpp` state contract is a proven, already-validated 9-dim
   observation shape** (`ARCHITECTURE.md`, read directly from firmware): `[ball_x, ball_y,
   x_error, y_error, filtered_vel_x, filtered_vel_y, actual_step_A, actual_step_B,
   actual_step_C]`. This is a stronger, already-working reference than `RT1LiteVLA`'s own
   state input, which is a much simpler 2-dim `[state_x, state_y]` (`dataset.py` line 43-45,
   `vla_architecture.py`'s `state_dim=2` default) — `RT1LiteVLA` never carries velocity or
   actual (lag-affected) motor position, both of which `RLControl.cpp`'s own design notes
   flag as load-bearing (the policy was trained on lagged state specifically).

## 2. Caveat on what's genuinely new synthesis here vs. already established

Everything in §1 is cited from existing docs/code, not re-derived. What's new in this
document specifically:

- Mapping the "grounding head + action head" split explicitly onto **Qwen2.5-VL-3B** as the
  backbone — the research spike only ever discussed this split in the context of
  Jetson-PI/π0.5 (dual-tier framing) and then SmolVLA (single-model framing); it never
  discussed Qwen2.5-VL-3B as a backbone to attach a custom action head to, because
  Qwen2.5-VL-3B was previously only considered as a *fully autoregressive outer-tier
  candidate on its own* (`AGENTS.md`'s original 3-candidate survey, "few tokens/sec as a
  discrete planner, not per-frame control") — never as a backbone with a *separate* head
  bolted on.
- The `PolicyCommand` interface gap identified in §4.
- The recommendation to reuse arm 1's 9-dim state contract over `RT1LiteVLA`'s simpler 2-dim
  one (§1 item 7 draws the comparison; the recommendation to prefer the richer one is this
  doc's call, made explicit in §3).
- The discretized-action-token alternative and why it's deprioritized (§3.4).
- The chunk-size and output-bounding open questions (§5).

## 3. Recommended multi-head shape for Qwen2.5-VL-3B

### 3.1 Head A — Grounding/language head (native to Qwen2.5-VL-3B, not new engineering)

Qwen2.5-VL-3B's own vision encoder + language backbone + `lm_head`, used as-is or LoRA-
adapted (not fully retrained — retraining would waste the pretrained grounding capability
that is the entire reason arm 2 uses a large pretrained VLM instead of a from-scratch model,
per `ARCHITECTURE.md`'s framing of arm 2 as depending on the lab partner's pretrained
weights). Role: fuse the camera frame and the instruction into a scene/target understanding.

Two sub-options for what this head's output is used for, **not resolved here, flag for item
3**:
- (a) Retained as real output for logging/debugging/interpretability only ("target: red
  marker, top-left of platform") — the action head (below) is what actually drives motors.
- (b) Used to replace the closed 5-word audio vocabulary entirely — Qwen's real language
  understanding could accept free-form instructions where arm 1's audio classifier cannot.
  This would be a genuine capability advantage of arm 2 over arm 1, not just a research
  artifact, but changes what "instruction" means as an input to `Policy.act()` — needs its
  own scoping pass, not decided by this doc.

### 3.2 Head B — State fusion (new, small)

A small encoder (mirror `RT1LiteVLA`'s `state_embed: nn.Linear(state_dim, 512)` pattern,
§1 item 5) that projects a proprioceptive state vector into the same embedding space as
Qwen's hidden states, for the action head to condition on.

**Recommendation: use arm 1's proven 9-dim `RLControl.cpp` contract (§1 item 7), not
`RT1LiteVLA`'s simpler 2-dim one.** It's already validated in production (RL-trained,
running today), already carries the velocity/lag terms `RLControl.cpp`'s own notes call
load-bearing, and the LeRobot conversion pipeline (§1 item 6) needs a concrete state-vector
shape decided regardless — reusing a proven shape is lower-risk than either copying
`RT1LiteVLA`'s simpler one or inventing a third.

### 3.3 Head C — Action-chunk regression head (new, the core deliverable for item 3)

Attaches to a pooled representation of Qwen's final hidden layer (or a dedicated learned
"action query" token appended to the input sequence, whose final hidden state gets pulled
out for this head — the same "dedicated token → pool → project" pattern `RT1LiteVLA` already
uses for its state token, §1 item 5) **plus** Head B's fused state embedding.

**Recommended internal shape (phase 1): a plain regression head, not flow-matching.**
Reuse `RT1LiteVLA.action_head`'s exact pattern — `Linear → ReLU → Linear` — rather than
building a flow-matching/diffusion action head like π0/SmolVLA's own. Rationale: nothing in
this codebase has ever implemented flow-matching; the regression pattern is already written,
already trained, already validated end-to-end (`ml_multimodal/models/vla_v1/best_vla.pth`).
Matches this project's own repeated "start simple, escalate only if needed" pattern (small
vision class before medium/YOLO class per `CLAUDE.md`'s Architecture Decisions table; BC
before RL per `EVALUATION_STRATEGY.md`). A flow-matching upgrade is a legitimate phase-2
stretch goal if regression proves inadequate (multimodal/ambiguous action distributions),
not a phase-1 requirement.

**Output shape: `[N, 3]` (a chunk of N future `theta_a/b/c` triples), not `[3]`.** This is
the action-chunking mechanism itself (§1 item 2) — a single `[3]` output would put Qwen back
on the per-tick critical path, reintroducing the exact "few tokens/sec, not real-time"
problem `AGENTS.md`'s original survey already ruled out. `N` is not fixed by this doc; see
§5.

### 3.4 Alternative considered and deprioritized: discretized action tokens

RT-2/OpenVLA-style designs bin each continuous action dimension and repurpose low-frequency
vocabulary tokens as action tokens, decoded autoregressively through Qwen's existing
`lm_head` — no new head at all, in principle. Considered and **not recommended as the
primary path**: autoregressive token-by-token decoding is exactly the "few tokens/sec"
bottleneck this whole track's research already used to rule out single-rate monolithic
VLAs (§1 item 1-2). Using it for the action path specifically would reintroduce the same
problem at the component level that was already solved at the system level by moving to
action chunking. Recorded here so it isn't silently dropped, matching this project's
documentation convention of stating what was considered, not just what was chosen.

## 4. Interface gap found by reading the real scaffold code (not assumed)

`host_software/ml_jetson_vla/core/policy_interface.py`'s `PolicyCommand` dataclass
(read directly this session) has exactly three payload fields:

```python
target_x_mm: float
target_y_mm: float
step_targets: Optional[tuple] = None  # (stepA, stepB, stepC), Phase B only
```

`step_targets` is documented as **raw step-space** (arm 1's Phase-B shape, no angle
conversion needed — matches `RLControl.cpp`'s direct step-space output). **There is no field
for `theta_a/b/c` motor angles anywhere in this interface.** Per §1 item 4, arm 2/3's
terminal action-head output is angles, not steps, and needs `angle_to_steps()` applied
downstream (`MotorControl.h`) — a different output shape than what `step_targets` already
represents. Concretely: `PolicyCommand` as it stands today **cannot carry arm 2's action-head
output** without a new field (something like `motor_angles_deg: Optional[tuple]`, distinct
from both `target_x_mm/y_mm` — which is a ball-space target, not a motor angle — and
`step_targets` — which is already in step-space, arm 1's shape).

This is flagged as a finding for item 3 to act on, **not resolved here** — no code changes
made, per this task's scope. Worth noting it's adjacent to, but distinct from, the on-hold
firmware item (item 6, "STM32 accepts angles instead of its current format") — that item is
about the firmware's *input* format; this finding is about whether the Jetson-side *Python*
interface even has a place to put an angle before it reaches firmware at all. Fixing this
interface gap does not require deciding item 6.

## 5. Open questions, explicitly not resolved by this doc

1. **SmolVLA vs. Qwen2.5-VL-3B** — §0's inconsistency. Needs the user to confirm which is
   actually the standing decision before item 3 starts real engineering.
2. **Chunk size N** — SmolVLA's own default (~50) is cited in `LARGE_VLA_RESEARCH_SPIKE.md`
   as a fact about SmolVLA, not derived from this project's own control requirements. Should
   be scoped against `EVALUATION_STRATEGY.md`'s settling-time metric (a chunk that's too long
   risks the ball drifting out of tolerance mid-chunk before the next replan; too short
   reintroduces the inference-latency bottleneck) — not copied blindly from SmolVLA's number.
3. **Output bounding** — `RLControl.cpp` (arm 1) bounds its output (`tanh`, scaled by 98)
   before treating it as a target; `RT1LiteVLA.action_head` (§1 item 5) has **no bounding
   activation at all** — raw unbounded linear output. This is an existing, apparently
   unnoticed gap in the in-house baseline, not something introduced by this doc. Item 3
   should decide whether the new action head bounds its output to the platform's known-safe
   motor-angle range (mirroring arm 1's proven pattern) or leaves it unbounded like
   `RT1LiteVLA` currently does.
4. **What Head A's grounding output is actually used for** — §3.1(a) vs (b), not decided.
5. **Whether Head A (language) and Head C (action) should be trained jointly (multi-task
   loss) or Head A frozen entirely once Head C is added** — not researched this session,
   would affect whether the existing pretrained grounding capability degrades during action
   fine-tuning.
6. **Whether FASTER's phase-mismatch concern is actually resolved by chunking** — carried
   over unresolved from `LARGE_VLA_RESEARCH_SPIKE.md`'s own open item (2026-09-15 section):
   action chunking raises effective Hz but an external event landing mid-chunk still waits
   out the rest of the chunk. Applies identically here regardless of backbone; not re-solved
   by switching from SmolVLA to Qwen2.5-VL-3B.
7. **RT1LiteVLA as comparison point, not a component to reuse verbatim** — per `CLAUDE.md`'s
   own framing, `RT1LiteVLA` is a secondary/optional lightweight baseline (closed 5-word
   vocabulary, off-the-shelf frozen ResNet18, stubbed RL stage), not arm 2. Its action-head
   *pattern* is reusable (§1 item 5, §3.3); its overall architecture is not what arm 2 should
   become.

## 6. What this doc does not do

No model code, no training loop, no head implementation. No firmware changes or proposals
(item 6 stays on hold, untouched). Does not decide SmolVLA-vs-Qwen2.5-VL-3B. Does not add
the `motor_angles_deg` field to `PolicyCommand` (§4) — that's an implementation change
belonging to item 3, flagged here, not made here.
