# Arm 2 minimal-baseline sweep, run directly on the Jetson (2026-09-22)

Companion to `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` (read its §8 erratum first: the scorer this
uses is the corrected v2 one). This is the plain-script alternative to the Colab notebook
(`deployment/arm2_colab_sweep.ipynb`, left untouched as a working alternative). Same experiment: 4
candidates x 3 prompt variants (`baseline`, `oriented`, `oriented_aruco`), 6 evenly-subsampled
colour-command frames per session x the 10 reference sessions = 60 frames, `max_new_tokens=24`,
corrected scoring, per-candidate fault isolation, validation gate, per-call checkpointing.

Why on-device: Colab's T4/A100 is not the Orin's Ampere GPU; latency/memory numbers from the
deployment target are what this comparison is supposed to measure, and the 10
`session_jetson_track4_*` sessions were collected here (no 666 MB bundle).

**Nothing here has run on the Jetson.** Everything below was verified against mock backends on the dev
machine only (section 6). The first real execution is the user's.

## 1. The shape

- Code: `deployment/run_arm2_sweep_jetson.py` (new, ~600 lines, a thin stage sequencer) importing the
  unchanged engine `deployment/colab_sweep.py`. Runs from a **full repo checkout** (`git pull`).
- Stages (`--stage`): `probe` (what is installed / which candidate can run here), `frames`,
  `hfauth`, `smoke`, `validate`, `sweep`, `export`, `all` (default: frames -> hf auth -> smoke ->
  gate -> sweep -> export).
- `--candidates KEY [KEY ...]` splits the run across environments; separate invocations share one
  checkpoint (`--results-dir` + `--run-label`), one at a time.
- Candidate keys: `qwen2_5_vl_3b`, `internvl2_5_4b`, `paligemma2_3b_mix:prompt`,
  `paligemma2_3b_mix:detect`, `moondream2:query`, `moondream2:point`; optional named-only fallback
  `internvl3_5_4b_hf` (InternVL3.5-4B via native transformers; a **different model generation**, report
  it as such).
- `--use-mock` = fake models, no GPU/downloads, for exercising the plumbing.

## 2. Environment decision (the risky part)

### Evidence, all from this repo

1. `JETSON_ENV_SETUP.md`: Track 1's environment pins `numpy<2` (apt's `python3-opencv` is built
   against the numpy 1.x ABI) and `onnxruntime==1.18.0`; installing `lerobot` once silently upgraded
   numpy underneath it and broke `import cv2` / `import onnxruntime` (fixed by re-pinning
   `numpy==1.26.4`). The class of bug: **an unconstrained `pip install` moved a shared package.**
2. `JETSON_ENV_SETUP.md`: the offline flash installed no JetPack SDK components (no `/usr/local/cuda`,
   cuDNN, TensorRT); GPU torch comes from NVIDIA's Jetson AI Lab wheel index and is the hard-won,
   expensive-to-reproduce part of any GPU environment here. Whatever the existing Qwen environment did
   to get a working CUDA torch, **do not rebuild that stack** in a second environment.
3. `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` §8.4: Moondream2's remote code (revision 2025-06-21) is
   reported broken on `transformers>=5`; the dev-machine Qwen run used `transformers==5.17.0`. The Colab
   notebook resolved this in one throwaway environment with `transformers>=4.56,<5` and never verified
   that Qwen reproduces under it (that is what its gate is for). InternVL2.5's remote code was written
   for ~4.37-4.4x; whether it runs on 4.56+ is likewise unknown until smoke.
4. `vlm_backends.py` / `colab_sweep.py`: a candidate that fails to load is caught, recorded and skipped
   (fault isolation) -- a *runtime* failure never corrupts an environment. Only *installs* do.

### What I could not do

Inspect the Jetson's existing Qwen environment: this environment has no access to the device (project
rule: hardware steps are the user's). So the decision below is conditional on one probe the user runs
(step 2), not on a guess about what is installed there. My expectation, stated as an expectation: that
environment was created recently with a current transformers (>=5), i.e. the Moondream2 conflict is real.

### The decision

**Two environments, never a single shared mutated one, and never four.**

- **`Q` = the existing Qwen environment. Untouched except for constraint-frozen, additive installs of
  missing pure-wheel plumbing (`av`, `pandas`, ...), and only if the probe says they are missing.** It
  runs Qwen2.5-VL only: the validation gate and the Qwen sweep. Reason: the gate exists to decide whether
  *this machine* reproduces the local reference; running it in the environment already smoke-tested live
  removes "I just changed transformers" as a confound, and the known-good environment stays a known-good
  fallback.
- **`X` = one new environment, created as a directory clone of `Q`** (`cp -a`, reuses the CUDA torch
  stack with no re-download, and `Q` stays pristine), then `transformers>=4.56,<5` (+ `timm`, `einops`)
  installed under a frozen-constraints file that excludes only the transformers-linked packages
  (`transformers`, `tokenizers`, `huggingface_hub`, `safetensors`, `hf-xet`). It runs InternVL2.5,
  PaliGemma2 and Moondream2 together: Moondream2's `<5` requirement sets the pin for the group and none
  of the three has a documented conflict with the others.
- **Why not one shared environment**: mutating the only known-good GPU environment with a
  transformers major-version change plus new packages is exactly the failure class in (1), with no
  rollback, and it would entangle the gate with the change under test.
- **Why not one environment per candidate**: each would need its own CUDA torch stack (GBs over WiFi,
  the fragile part per (2)) for no benefit; the three have no known mutual conflicts.

**Shortcut, decided mechanically by the probe (not by preference):** if `--stage probe` in `Q` shows
`transformers` already in `[4.56, 5)` (or at least `<5` with Qwen2.5-VL importable), then Moondream2 is
not blocked there and all four candidates can run from `Q` with additive installs only; skip `X`
entirely. Even then, keep the gate-first order.

### The mechanism that makes installs safe (applies to every install below)

`pip install -c <frozen-constraints> --only-binary=:all: ...`. A constraints file of exact `==` pins
of everything currently installed makes pip **refuse** (loudly) any change to an installed package
instead of silently upgrading it -- the lerobot/numpy failure mode. `--only-binary=:all:` prevents an
aarch64 source build from half-installing. An audit `diff` afterwards proves nothing moved. Do not
`source $X/bin/activate` in the clone (its scripts still point at `Q`); always call `$X/bin/python`,
which also makes it impossible to run in the wrong environment by accident.

If `Q` is **not** a venv (probe prints `venv=False`): create `X` with
`python3 -m venv --system-site-packages $X` instead of cloning; the venv's own site-packages shadow
the inherited `transformers`, the CUDA torch is inherited, and nothing outside `X` is touched.

**OpenCV note**: the sweep needs `cv2` *with* `aruco` and `av`. If `Q` inherits apt's `python3-opencv`
(4.5.4, needs `numpy<2`), keep `numpy<2` frozen and do not pip-install any cv2. If `Q` has no cv2/aruco,
install `opencv-contrib-python-headless` under the same frozen constraints (pip's resolver will
backtrack to a numpy-1.x-compatible wheel or refuse; do not override it). `estimate_homography_from_aruco`
supports both the legacy (4.5.x) and new (>=4.7) ArUco APIs; sub-pixel homography differences between
OpenCV builds are far inside the gate's tolerances, and step 6 (mock run) checks them on-device.

## 3. Hugging Face auth for PaliGemma2 (gated), no `google.colab.userdata`

**Decision: one-time interactive login on the Jetson (cached token), with `HF_TOKEN` as an override.**
1. In any browser, open `https://huggingface.co/google/paligemma2-3b-mix-448` logged in, accept the
   Gemma license; create a **read** token at `https://huggingface.co/settings/tokens`.
2. On the Jetson, once, in `X`: `$X/bin/python -c "from huggingface_hub import login; login()"` and paste
   the token (input hidden). It is stored in `~/.cache/huggingface/token`.
3. No code change was needed: `colab_sweep.get_hf_token()` already falls back to the `HF_TOKEN` env var
   outside Colab, and `check_hf_access()` passes `token=None` to `hf_hub_download`, which resolves to the
   cached login. `export HF_TOKEN=...` also works and takes precedence. The driver's `hfauth`/sweep stages
   print `GATED ACCESS OK`/`UNAVAILABLE` and skip (not crash) the two PaliGemma2 specs if access is missing.

Do not put the token in the repo, a script or a shell history file you commit.

## 4. Ordered commands

Variables (adjust the three paths that are unknown to me):
```bash
export REPO=$HOME/VRI_2026                     # the git checkout on the Jetson
export Q=$HOME/venvs/qwen                      # <-- your EXISTING, smoke-tested Qwen venv
export X=$HOME/venvs/arm2_extra                # new, created in step 3 only if the probe says so
export R=$HOME/arm2_jetson_sweep_results       # checkpoints/results, deliberately OUTSIDE the git tree
export D=$REPO/host_software/ml_jetson_vla/deployment
```

**Step 0 -- code (git pull).** Everything this needs was committed and pushed in `515230e` (the Arm 2
files had been untracked until that commit), so `git log -1` on the Jetson must show `515230e` or
later. Only `data/01_bronze/*` (gitignored; already on the device) and the Qwen checkpoint
(`models.dvc`) do not arrive via git. On the Jetson:
```bash
cd $REPO && git pull && git log -1 --oneline
ls $D/run_arm2_sweep_jetson.py $D/colab_sweep.py $D/score_minimal_baseline_offline.py \
   $D/arm2_minimal_baseline_prompt_ab_scoring_20260918_RESCORED_v2.json \
   $REPO/host_software/ml_jetson_vla/core/vlm_backends.py $REPO/host_software/ml_jetson_vla/core/minimal_vlm_policy.py
ls -d $REPO/host_software/data/01_bronze/session_jetson_track4_* | wc -l     # >= 10 (extras are ignored by default)
df -h $HOME; du -sh $Q                                                       # room for a clone + ~20 GB of checkpoints (rough)
sudo nvpmodel -q                                                             # note it; MAXN for representative latency
```

**Step 1 -- read-only probe of the existing Qwen environment (changes nothing).**
```bash
$Q/bin/python $D/run_arm2_sweep_jetson.py --stage probe --results-dir $R
```
Read: `venv=`; `transformers` version (>=5 -> `X` is needed, expected; in `[4.56,5)` -> shortcut);
`cuda_available=True`, capability `[8, 7]` (if CUDA is False, this is not the environment to use --
stop); any `MISSING` for `av`/`pandas`/`cv2`/`accelerate`/`qwen_vl_utils`; `cv2.aruco True`; the
per-candidate table at the bottom. **Send me this output if anything is surprising.**

**Step 2 -- only if step 1 printed `[BLOCKED-ALL]` (or `accelerate`/`qwen_vl_utils` MISSING): additive
installs into `Q`, frozen-constraints form.**
```bash
$Q/bin/python -m pip freeze --exclude-editable | grep -E '^[A-Za-z0-9_.-]+==' > ~/arm2_q_constraints.txt
$Q/bin/python -m pip install -c ~/arm2_q_constraints.txt --only-binary=:all: -r $D/requirements-jetson-arm2-additive.txt
$Q/bin/python -m pip freeze --exclude-editable | grep -E '^[A-Za-z0-9_.-]+==' | diff ~/arm2_q_constraints.txt - | grep '^<'
#   ^ must print NOTHING. Anything printed = an existing package changed; stop and tell me.
$Q/bin/python $D/run_arm2_sweep_jetson.py --stage probe --results-dir $R      # BLOCKED-ALL must be gone
```
If pip reports a conflict, that is the constraints doing their job: do not loosen them; send me the message.

**Step 3 -- the second environment `X` (skip if the shortcut applies).**
```bash
cp -a "$Q" "$X"                     # clone (or: python3 -m venv --system-site-packages $X  if Q is not a venv)
$X/bin/python -m pip freeze --exclude-editable | grep -E '^[A-Za-z0-9_.-]+==' \
  | grep -viE '^(transformers|tokenizers|huggingface[-_]hub|safetensors|hf[-_]xet)==' > ~/arm2_x_constraints.txt
$X/bin/python -m pip install -c ~/arm2_x_constraints.txt --only-binary=:all: -r $D/requirements-jetson-arm2-transformers-lt5.txt
$X/bin/python $D/run_arm2_sweep_jetson.py --stage probe --results-dir $R
#   expect: prefix=$X (proves the clone relocated and is not secretly using Q), transformers 4.x,
#           cuda_available=True, moondream2 "ok", av/pandas/cv2 present, numpy unchanged vs Q
$Q/bin/python -c "import transformers, numpy; print('Q intact:', transformers.__version__, numpy.__version__)"
```

**Step 4 -- HF login (once, from `X`; section 3).**
```bash
$X/bin/python -c "from huggingface_hub import login; login()"
```

**Step 5 -- checkpoint downloads, on-device (resumable; do them before the sweep so a network problem is
not mistaken for a model problem).**
```bash
$X/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
for repo, rev in [("google/paligemma2-3b-mix-448", None),            # gated: needs step 4
                  ("OpenGVLab/InternVL2_5-4B", None),
                  ("vikhyatk/moondream2", "9a7d4024050840e001defacec2b00727e89149e6")]:   # pinned (tag 2025-06-21)
    print(repo, "->", snapshot_download(repo, revision=rev))
EOF
ls $REPO/host_software/ml_jetson_vla/models/qwen2_5_vl_3b_instruct >/dev/null && echo "Qwen checkpoint present at the default path" \
  || echo "Qwen not at the default path: pass --qwen-model-dir <the path your smoke test used> to EVERY invocation (it is part of the checkpoint identity), or 'dvc pull', or let it download Qwen/Qwen2.5-VL-3B-Instruct at the pinned commit"
```

**Step 6 -- data + plumbing check with mock models (no GPU, ~a minute; do not skip: it exercises this
device's OpenCV ArUco homography and PyAV decode on the real sessions).**
```bash
$Q/bin/python -u $D/run_arm2_sweep_jetson.py --use-mock --stage all --results-dir /tmp/arm2_mock --run-label mock 2>&1 | tail -40
```
Expect `reference-locked, 10 sessions`, `60 frames prepared`, `'identical': True`, all six mock candidates
`complete`, oracle errors of a few mm with `HIT` on colour frames. Anything else: stop.

**Step 7 -- Qwen: smoke, then the validation gate, then the Qwen sweep (in `Q`, in tmux so an SSH drop
cannot kill it; the checkpoint would survive a kill anyway).**
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks        # optional; MAXN + max clocks for representative latency. Record it.
tmux new -s arm2                                 # (or: nohup ... &)
$Q/bin/python -u $D/run_arm2_sweep_jetson.py --candidates qwen2_5_vl_3b --stage all \
   --results-dir $R --run-label jetson_run1 2>&1 | tee -a $R/qwen_run.log
```
`--stage all` = 2 smoke frames -> **gate** (60 frames x baseline/oriented, compared with the local CPU
reference: parse rate >= 0.95, mean legacy error within 25 mm, mean corrected error within 12 mm, median
pixel deviation <= 25 px, same model-input size) -> the remaining Qwen calls (`oriented_aruco`) -> export.
The gate prints, per variant, hits/mean on this machine next to the reference's own (baseline 40/60 @
27.9 mm, oriented 21/60 @ 43.7 mm). **If it prints `PIPELINE VALIDATION: FAILED` the script stops before the
sweep. Do not `--force-continue`; send me `$R/qwen_run.log` and the checkpoint.** Not expected to be
bit-exact (different kernels than the CPU bf16 run); see the failure message for the suspect order
(dtype, transformers/processor resize, decoding).

**Step 8 -- the other three candidates (in `X`).**
```bash
$X/bin/python -u $D/run_arm2_sweep_jetson.py \
   --candidates internvl2_5_4b paligemma2_3b_mix:prompt paligemma2_3b_mix:detect moondream2:query moondream2:point \
   --stage all --results-dir $R --run-label jetson_run1 2>&1 | tee -a $R/extra_run.log
```
The gate line will say "not applicable here (qwen is not in this invocation)" -- expected. This
invocation is **refused at the start** unless step 7's gate passed for this `--run-label` (or you
pass `--skip-validation-gate` knowingly). If InternVL2.5 fails to load on this transformers (recorded as
`load_failed` with a traceback, the rest continue), the named fallback is
`--candidates internvl3_5_4b_hf` (InternVL3.5, a different generation).

**Step 9 -- combined table + export (either environment; add the same `--qwen-model-dir` if you used it).**
```bash
$Q/bin/python $D/run_arm2_sweep_jetson.py --stage export --results-dir $R --run-label jetson_run1
```
Prints the headline candidate x variant table (with the two no-model `[ref]` yardsticks: constant
centre guess, uniform-random), the coordinate-space diagnostic, and writes
`$R/arm2_colab_sweep_results_jetson_run1.json` (+ per-candidate scorer-format JSON + CSV; the
`colab_sweep` in the filename is the engine's fixed name, not a Colab dependency). Each candidate's
stage record in that file carries the environment that produced it.

**Interrupt/resume**: Ctrl-C or a kill at any point; rerun the same command. Finished calls are skipped
(a fully finished candidate is not even loaded). Failed calls are never stored as done.

## 5. How to read the result (unchanged from the Colab framing)

n = 60 (6 per session, 10 sessions); a hit-rate difference under ~10 points is within noise, per-session
spread matters more than the pooled number, and the `[ref]` rows are the yardstick (the constant-centre
guess already scores 25.2 mm mean / 9 hits locally). One run, one seed of prompt wording: not a ranking
(`model-iteration-constraints`). Latency/memory numbers are now the Orin's; record the `nvpmodel` mode
(it is captured in each candidate's `env.jetson.nvpmodel` when readable without sudo).

## 6. What was verified where

**Verified on the dev machine (real code, mock models):** `deployment/test_run_arm2_sweep_jetson_mock.py`
-- environment probe + static candidate rules; preflight dropping only incompatible candidates; the
reference-locked session set + parity assertion; a full mock run of all six specs; the two-invocation
split sharing one checkpoint (union identical to a single run); a **real hard process kill mid-sweep**
followed by resume (checkpoint valid JSON, finished calls not redone, final items identical to the
uninterrupted run); the gate failing on an oracle mock, passing when fed the reference outputs,
`--force-continue` recorded as forced, and a sweep refused (before any model work) without a passing gate
record. Plus `py_compile`. The shared engine's own 31-check test (`test_colab_sweep_mock.py`) is unchanged.

**Not verified (needs the Jetson):** that any real model loads on aarch64 / this transformers line;
that InternVL2.5's remote code runs on transformers 4.56+; whether `timm`/`einops` are the right extras;
that Qwen reproduces the local reference (the gate); real latency/memory; the HF gated flow; on-device
OpenCV/PyAV behaviour; that `cp -a` of the Qwen venv relocates cleanly (expected to, when called via
`$X/bin/python`; if not, use the `--system-site-packages` route above).
