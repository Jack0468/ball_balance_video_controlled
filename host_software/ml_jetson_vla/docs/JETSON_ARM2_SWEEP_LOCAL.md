# Arm 2 minimal-baseline sweep, run directly on the Jetson (2026-09-22, container pivot)

Companion to `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` (read its §8 erratum first: the scorer this
uses is the corrected v2 one). This is the plain-script alternative to the Colab notebook
(`deployment/arm2_colab_sweep.ipynb`, left untouched as a working alternative). Same experiment: 4
candidate model families (6 candidate specs: `qwen2_5_vl_3b`, `internvl2_5_4b`,
`paligemma2_3b_mix:prompt`, `paligemma2_3b_mix:detect`, `moondream2:query`, `moondream2:point`) x 3
prompt variants (`baseline`, `oriented`, `oriented_aruco`), 6 evenly-subsampled colour-command frames
per session x the 10 reference sessions = 60 frames, `max_new_tokens=24`, corrected scoring,
per-candidate fault isolation, validation gate, per-call checkpointing.

Why on-device: Colab's T4/A100 is not the Orin's Ampere GPU; latency/memory numbers from the
deployment target are what this comparison is supposed to measure, and the 10
`session_jetson_track4_*` sessions were collected here (no 666 MB bundle).

**Superseded, this section (2026-09-22, same day): everything below used to describe two bare-metal
venvs (`Q`/`X`).** That plan is dropped, not extended — `Q` (an "existing, smoke-tested Qwen venv")
turned out not to exist on this device, and bare-metal venv installs failed repeatedly this session
(wrong wheel resolution via `--extra-index-url`, a 20-30GB NGC training image that exhausted the
device's 57GB disk). What actually worked, confirmed on this exact device: `docker run --rm -it
dustynv/l4t-pytorch:r36.4.0 python3 -c "import torch; print(torch.__version__,
torch.cuda.is_available())"` → `2.4.0 True` — real CUDA-enabled PyTorch, one command, no bare-metal
install at all. The whole plan below is now container-based. **Nothing in the container plan has run
on the Jetson yet either** — sections 2-4 were verified against mock backends only (section 7); the
Docker builds and the first real model run are the user's.

## 1. The shape

- Code: `deployment/run_arm2_sweep_jetson.py` (a thin stage sequencer, ~600 lines) importing the
  unchanged engine `deployment/colab_sweep.py`. Runs from a **full repo checkout**, bind-mounted into
  the container (not copied in at build time — see section 4).
- Stages (`--stage`): `probe` (what is installed / which candidate can run here), `frames`,
  `hfauth`, `smoke`, `calibrate`, `validate`, `sweep`, `export`, `all` (default: frames → hf auth →
  smoke → calibrate → validate → sweep → export). All of these are plain Python over local files —
  see section 5 for why the container changes nothing about how they run.
- `--candidates KEY [KEY ...]` splits the run across environments; separate invocations (now:
  separate `docker run`s, one per image) share one checkpoint (`--results-dir` + `--run-label`), one
  at a time.
- Candidate keys: `qwen2_5_vl_3b`, `internvl2_5_4b`, `paligemma2_3b_mix:prompt`,
  `paligemma2_3b_mix:detect`, `moondream2:query`, `moondream2:point`; optional named-only fallback
  `internvl3_5_4b_hf` (InternVL3.5-4B via native transformers; a **different model generation**, report
  it as such).
- `--use-mock` = fake models, no GPU/downloads, for exercising the plumbing. This is what section 7's
  test suite runs — it needs no Docker, no Jetson, nothing beyond a plain Python interpreter.

## 2. Container decision

### The base image

**`dustynv/l4t-pytorch:r36.4.0`** — confirmed pulled and running real CUDA torch on this exact
Jetson AGX Orin p3730 (JetPack 6.2.3, L4T r36.4) this session, 6.3GB. Do not swap to a different or
larger base image without a stated reason: disk space here is a real, already-hit constraint (57GB
total; a 20-30GB NGC training image exhausted it once already this session — check `docker images`
and `df -h` and remove that abandoned pull before building anything below, see section 4 step 0).

### What the base image already has vs. what had to be added

**Confirmed present**: `torch` 2.4.0 with a working CUDA build (`torch.cuda.is_available() ==
True`) — this is the hard-won part (a JetPack-6.2.3-matched, Ampere-capable PyTorch wheel; see
`JETSON_ENV_SETUP.md` for how expensive this was to get right on bare metal) and neither Dockerfile
touches it.

**Not confirmed, checked explicitly in both Dockerfiles rather than assumed**:
- `numpy` version — the base image ships *some* numpy compatible with its own torch build, but which
  one was never independently checked before this plan. Both Dockerfiles print it (`base numpy
  X.Y.Z`) as their first build step and then treat it as fixed (see the constraints mechanism below)
  rather than guessing a version and pinning against that guess.
- `cv2`/OpenCV with ArUco support — **not expected to be present**: `l4t-pytorch` is a PyTorch-focused
  image in the `dustynv`/jetson-containers family, where OpenCV is normally a separate stacked image
  (`l4t-cv`/similar), not bundled into the PyTorch one. Both Dockerfiles check `import cv2, cv2.aruco`
  first and only `pip install opencv-contrib-python-headless` if that fails. The earlier bare-metal
  plan's rule ("apt's `python3-opencv` needs `numpy<2`, so pin numpy first") was protecting a
  **shared host environment** — Track 1's own bare-metal venv, which this same numpy pin must not
  break. That shared-environment risk does not exist inside an isolated container (nothing else runs
  there), so it is **not reapplied verbatim**: the actual constraint that still matters — don't let
  pip silently replace the base image's numpy/torch/torchvision while adding opencv — is enforced
  directly (see below), not worked around by guessing numpy has to be `<2` here too. If it turns out
  the base image's numpy actually is `<2`, `opencv-contrib-python-headless` will simply resolve to a
  1.x-compatible wheel and this distinction is moot in practice, but it's now verified, not assumed.

### The safety mechanism (same discipline as the old venv plan, retargeted)

Each Dockerfile's first `RUN` layer is `pip freeze --exclude-editable > /opt/base_constraints.txt`
— a snapshot of every package version the base image actually ships, captured **before** anything is
installed. Every subsequent `pip install` in that Dockerfile is run as `pip install -c
/opt/base_constraints.txt --only-binary=:all: ...`: `-c` makes pip **refuse, loudly**, to change any
package already in that snapshot (numpy, torch, torchvision, and anything else the base image
shipped) instead of silently upgrading it — this is exactly the failure class that broke Track 1 once
already (the `lerobot`/numpy incident, `JETSON_ENV_SETUP.md`) and exactly the same fix, just applied
to a Docker layer instead of a venv. `--only-binary=:all:` forbids an aarch64 source build from
half-installing. Each Dockerfile's last `RUN` layer re-`pip freeze`s and diffs/asserts the base
packages are unchanged, and additionally hard-asserts the transformers-version-dependent import each
image exists for (see below) — so a build against the wrong transformers range, or a base-image
regression, fails **at `docker build` time**, not silently at sweep time on the device.

### The transformers-version split (the real decision here)

**Two Dockerfiles, both `FROM dustynv/l4t-pytorch:r36.4.0`, split by transformers major version —
not one image with an in-container pip swap.** Reasons, for real:
- The two transformers ranges are genuinely mutually exclusive for at least one candidate each
  (Moondream2 needs `<5`, confirmed broken on `>=5`; Qwen2.5-VL and PaliGemma2 need `>=5`, confirmed
  their model classes are simply absent under `4.57.6` — see below). An in-container pip swap between
  runs would repeat the exact "mutate the only known-good environment" risk the old venv plan's `Q`
  was specifically designed to avoid, except now for a container instead of a venv — no benefit, same
  risk, and it forfeits the loud-failure-at-build-time property above (a live swap only fails, if at
  all, deep into a run).
  A separate image per candidate (six images, one per spec) was also considered and rejected: each
  would still need the same CUDA torch base layer (shared on disk regardless, since Docker dedupes
  identical layers — see the size estimate below) for no additional isolation benefit, since the two
  transformers ranges are the only real mutual-exclusion boundary; splitting further just multiplies
  Dockerfiles to build without buying anything.
- Two images sharing one base layer is **cheap on this device's tight disk budget**: Docker stores a
  base image layer once regardless of how many derived images reference it (content-addressed by
  layer hash), so building both `Dockerfile.arm2-transformers5` and `Dockerfile.arm2-transformers4`
  costs the 6.3GB base layer **once**, not twice.

**Which candidates run where — corrected 2026-09-22 from a real on-device probe this session, not
from the pre-container plan's assumption:**

| Image | Dockerfile | transformers | Candidates | Why |
|---|---|---|---|---|
| Primary | `Dockerfile.arm2-transformers5` | `==5.17.0` | `qwen2_5_vl_3b` (+ validation gate), `paligemma2_3b_mix:prompt`, `paligemma2_3b_mix:detect` | A real probe on this Jetson under `transformers==4.57.6` (the newest 4.x — exactly what the other image's `>=4.56,<5` range resolves to) showed `Qwen2_5_VLForConditionalGeneration` **and** `PaliGemmaForConditionalGeneration` both **missing** from the installed transformers. Not "unproven" — genuinely absent. `5.17.0` is the version the historical local CPU reference run used, so it's also what the validation gate needs to reproduce against. |
| Secondary | `Dockerfile.arm2-transformers4` | `>=4.56,<5` | `internvl2_5_4b`, `moondream2:query`, `moondream2:point` | Moondream2's remote code (pinned revision `2025-06-21`) is reported broken on `transformers>=5` (`all_tied_weights_keys`, `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` §8.4) — confirmed, must run `<5`. InternVL2.5-4B works under **either** major (confirmed this session) — kept here so it isn't duplicated into both images. |

This is the **opposite split** from what `requirements-jetson-arm2.txt` / `requirements-jetson-arm2-
fallback-transformers5.txt` said before today: those files previously treated `<5` as the environment
for "all four candidates" and `5.17.0` as an occasional fallback tested only if a validation-gate
failure looked version-related. That framing is now known to be wrong (Qwen and PaliGemma2 simply
cannot import under `4.57.6`, so they were never going to run from the `<5` image regardless of what
the gate said) — both requirements files' headers have been corrected in place with this session's
date and the real evidence; their **package lists were already correct** and are unchanged, only the
"which env is this" framing was stale.

**No driver code change was needed for this.** `run_arm2_sweep_jetson.py`'s `candidate_issues()`
already checks `transformers_has_qwen2_5_vl` / `transformers_has_paligemma` / the Moondream2 `>=5`
rule **at runtime**, per invocation, and BLOCKED-skips (before any multi-GB download) whichever
candidates don't belong in the container it's running in — it was already correct; only this doc's
and the requirements files' *description* of which environment is "the main one" was stale. Each
Dockerfile's final build layer also hard-asserts its own two required imports (see the Dockerfiles),
so a build against the wrong transformers range fails immediately rather than only being caught later
by the driver's own preflight check.

### Image size estimate

- `Dockerfile.arm2-transformers5`: 6.3GB (shared base) + ~1.0-1.4GB (opencv, transformers 5.17.0 +
  tokenizers/huggingface_hub/safetensors, accelerate, qwen-vl-utils, sentencepiece, av, pandas,
  pillow) ≈ **7.3-7.7GB**, of which 6.3GB is shared with the other image.
- `Dockerfile.arm2-transformers4`: 6.3GB (shared base) + ~1.2-1.6GB (the above plus `timm`+`einops`
  for InternVL2.5/Moondream2, transformers 4.57.x instead of 5.17.0) ≈ **7.5-7.9GB**, same shared
  6.3GB base.
- **Both together, accounting for the shared base layer: roughly 9.5-10GB of unique disk**, not
  ~15GB. Model checkpoints (Qwen2.5-VL-3B ~6-7GB, InternVL2.5-4B ~8GB, PaliGemma2-3b-mix ~6GB,
  Moondream2 ~2GB bf16/fp32) are **not** baked into either image — they land in the bind-mounted
  Hugging Face cache at `docker run` time (section 4) and are a separate, larger disk cost (~20-25GB
  if all four checkpoints are pulled) that should be budgeted against the host's `~/.cache/huggingface`
  mount, not against either image.
- On a 57GB device that has already hit "no space left," budget roughly: ~10GB (both images) + ~25GB
  (all checkpoints, worst case) + whatever the OS/JetPack/existing repo/bronze data already use. Check
  `df -h` before and after each build/download step (section 4 step 0 and inline reminders below), and
  clear the abandoned NGC image first if it's still cached.

## 3. Hugging Face auth for PaliGemma2 (gated), inside a container

**Decision: one-time interactive login inside the `transformers5` container (cached token on a
bind-mounted host directory), with `HF_TOKEN` as an override — same mechanism as the old venv plan,
just run inside `docker run -it` instead of a venv shell.**
1. In any browser, open `https://huggingface.co/google/paligemma2-3b-mix-448` logged in, accept the
   Gemma license; create a **read** token at `https://huggingface.co/settings/tokens`.
2. On the Jetson, once, with the HF cache directory already bind-mounted (section 4's `-v
   $HOME/.cache/huggingface:/root/.cache/huggingface`, or adjust `/root` if the image's default user
   isn't root — check with `docker run --rm dustynv/l4t-pytorch:r36.4.0 whoami` / `echo $HOME` once
   first): run the container interactively and `python3 -c "from huggingface_hub import login;
   login()"`, paste the token (input hidden). It is stored at
   `~/.cache/huggingface/token` **inside the container's `$HOME`**, which — because that path is
   bind-mounted from the host — actually lands on the host and survives every future `--rm` container
   exit, including a different container run later.
3. No code change was needed: `colab_sweep.get_hf_token()` already falls back to the `HF_TOKEN` env
   var outside Colab, and `check_hf_access()` passes `token=None` to `hf_hub_download`, which resolves
   to the cached login. `export HF_TOKEN=...` (passed into the container with `-e HF_TOKEN`) also
   works and takes precedence. The driver's `hfauth`/sweep stages print `GATED ACCESS OK`/
   `UNAVAILABLE` and skip (not crash) the two PaliGemma2 specs if access is missing.

Do not put the token in the repo, a script, an image layer, or a shell history file you commit. It
must only ever live in the bind-mounted `~/.cache/huggingface` on the host, never `COPY`'d into either
Dockerfile.

## 4. Ordered commands

Variables (adjust the paths that are specific to this device):
```bash
export REPO=$HOME/VRI_2026                        # the git checkout on the Jetson
export R=$HOME/arm2_jetson_sweep_results           # checkpoints/results, deliberately OUTSIDE the git tree
export D=$REPO/host_software/ml_jetson_vla/deployment
export HFCACHE=$HOME/.cache/huggingface            # HF token + downloaded checkpoints, host-persisted
```

**Step 0 — disk/code sanity (changes nothing but what you tell it to).**
```bash
df -h $HOME                                          # confirm real headroom before building anything
docker images                                        # look for the abandoned 20-30GB NGC pull from earlier
                                                       # this session; `docker rmi <id>` it if still present
docker system df                                     # Docker's own view of what's using space
cd $REPO && git pull && git log -1 --oneline          # code (same as the old plan's step 0)
ls -d $REPO/host_software/data/01_bronze/session_jetson_track4_* | wc -l   # >= 10
sudo nvpmodel -q                                      # note it now; run on the HOST, not in the container —
                                                       # nvpmodel/jetson_clocks are host L4T tools, not
                                                       # present inside dustynv/l4t-pytorch
```

**Step 1 — build both images (from `$D`; each build is independent, either order).**
```bash
cd $D
docker build -f Dockerfile.arm2-transformers5 -t arm2-t5:r36.4.0 .
docker build -f Dockerfile.arm2-transformers4 -t arm2-t4:r36.4.0 .
df -h $HOME     # confirm the ~9.5-10GB combined estimate held; both builds print their own
                # "transformers X.Y.Z" / class-assertion lines near the end — read them
```
If the docker daemon's default runtime is not already `nvidia` (it likely already is, since the
plain `docker run --rm -it dustynv/l4t-pytorch:r36.4.0 python3 -c "..."` command that confirmed CUDA
this session used no explicit `--runtime` flag), pass `--runtime nvidia` explicitly on every `docker
run` below — harmless if it's already the default, required if it isn't. Check with `docker info |
grep -i runtime` or `cat /etc/docker/daemon.json` if unsure.

**Step 2 — mock plumbing check (either image; no GPU needed for this stage, but run it through a
real container to prove the mounts work end-to-end — do not skip: it exercises this device's OpenCV
ArUco homography and PyAV decode on the real sessions, same as the old plan's step 6).**
```bash
docker run --rm --runtime nvidia \
  -v "$REPO":/workspace/VRI_2026 \
  -v "$REPO/host_software/data/01_bronze":/workspace/VRI_2026/host_software/data/01_bronze:ro \
  -v "$R":/workspace/arm2_results \
  -v "$HFCACHE":/root/.cache/huggingface \
  -w /workspace/VRI_2026/host_software \
  arm2-t5:r36.4.0 \
  python3 -u ml_jetson_vla/deployment/run_arm2_sweep_jetson.py \
    --use-mock --stage all --results-dir /workspace/arm2_results --run-label mock 2>&1 | tail -40
```
Expect `reference-locked, 10 sessions`, `60 frames prepared`, `'identical': True`, all six mock
candidates `complete`, oracle errors of a few mm with `HIT` on colour frames. Anything else: stop.

**Step 3 — HF login (once, inside the `transformers5` image; section 3).**
```bash
docker run --rm -it --runtime nvidia \
  -v "$HFCACHE":/root/.cache/huggingface \
  arm2-t5:r36.4.0 \
  python3 -c "from huggingface_hub import login; login()"
```

**Step 4 — checkpoint downloads, on-device (resumable; do them before the sweep so a network problem
is not mistaken for a model problem). Run each candidate's checkpoint pull from the image that will
actually use it, so the download happens under the right container's disk/library versions.**
```bash
docker run --rm --runtime nvidia -v "$HFCACHE":/root/.cache/huggingface arm2-t5:r36.4.0 python3 - <<'EOF'
from huggingface_hub import snapshot_download
print("paligemma2-3b-mix-448 ->", snapshot_download("google/paligemma2-3b-mix-448"))   # gated: needs step 3
EOF

docker run --rm --runtime nvidia -v "$HFCACHE":/root/.cache/huggingface arm2-t4:r36.4.0 python3 - <<'EOF'
from huggingface_hub import snapshot_download
print("InternVL2_5-4B ->", snapshot_download("OpenGVLab/InternVL2_5-4B"))
print("moondream2 ->", snapshot_download("vikhyatk/moondream2", revision="9a7d4024050840e001defacec2b00727e89149e6"))   # pinned (tag 2025-06-21)
EOF

ls "$REPO/host_software/ml_jetson_vla/models/qwen2_5_vl_3b_instruct" >/dev/null 2>&1 && \
  echo "Qwen checkpoint present at the default path (bind-mounted automatically via \$REPO)" || \
  echo "Qwen not at the default path: pass --qwen-model-dir <path> to EVERY invocation (part of the checkpoint identity), 'dvc pull' inside the container, or let it download Qwen/Qwen2.5-VL-3B-Instruct at the pinned commit into the mounted HF cache"
```

**Step 5 — Qwen: smoke, then the validation gate, then the Qwen + PaliGemma2 sweep (the
`transformers5` image; in `tmux` on the HOST so an SSH drop cannot kill the container — the checkpoint
would survive a kill anyway, since it's on the bind-mounted `$R`).**
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks   # optional, on the HOST (not in the container); MAXN + max
                                            # clocks for representative latency. Record the mode.
tmux new -s arm2
docker run --rm --runtime nvidia \
  -v "$REPO":/workspace/VRI_2026 \
  -v "$REPO/host_software/data/01_bronze":/workspace/VRI_2026/host_software/data/01_bronze:ro \
  -v "$R":/workspace/arm2_results \
  -v "$HFCACHE":/root/.cache/huggingface \
  -w /workspace/VRI_2026/host_software \
  arm2-t5:r36.4.0 \
  python3 -u ml_jetson_vla/deployment/run_arm2_sweep_jetson.py \
    --candidates qwen2_5_vl_3b paligemma2_3b_mix:prompt paligemma2_3b_mix:detect --stage all \
    --results-dir /workspace/arm2_results --run-label jetson_run1 2>&1 | tee -a "$R/t5_run.log"
```
`--stage all` = 2 smoke frames → coordinate-space calibration → **the gate** (60 frames x
baseline/oriented for Qwen only, compared with the local CPU reference: parse rate >= 0.95, mean
legacy error within 25 mm, mean corrected error within 12 mm, median pixel deviation <= 25 px, same
model-input size) → the remaining sweep calls → export. The gate prints, per variant, hits/mean on
this machine next to the reference's own (baseline 40/60 @ 27.9 mm, oriented 21/60 @ 43.7 mm). **If
it prints `PIPELINE VALIDATION: FAILED` the script stops before the sweep. Do not `--force-continue`;
send `$R/t5_run.log` and the checkpoint.** Not expected to be bit-exact (different kernels than the
CPU bf16 run); see the failure message for the suspect order (dtype, transformers/processor resize,
decoding).

**Step 6 — the other two candidates (the `transformers4` image). Refused at the start unless step
5's gate passed for this `--run-label` (checked via the shared `$R` mount) — or pass
`--skip-validation-gate` knowingly.**
```bash
docker run --rm --runtime nvidia \
  -v "$REPO":/workspace/VRI_2026 \
  -v "$REPO/host_software/data/01_bronze":/workspace/VRI_2026/host_software/data/01_bronze:ro \
  -v "$R":/workspace/arm2_results \
  -v "$HFCACHE":/root/.cache/huggingface \
  -w /workspace/VRI_2026/host_software \
  arm2-t4:r36.4.0 \
  python3 -u ml_jetson_vla/deployment/run_arm2_sweep_jetson.py \
    --candidates internvl2_5_4b moondream2:query moondream2:point --stage all \
    --results-dir /workspace/arm2_results --run-label jetson_run1 2>&1 | tee -a "$R/t4_run.log"
```
The gate line will say "not applicable here (qwen is not in this invocation)" — expected. If
InternVL2.5 fails to load on this transformers (recorded as `load_failed` with a traceback, the rest
continue), the named fallback is `--candidates internvl3_5_4b_hf` (InternVL3.5, a different
generation) added to this same invocation.

**Step 7 — combined table + export (either image; `export` never imports torch/transformers — see
section 5 — so either container can run it against the shared `$R` checkpoint; add the same
`--qwen-model-dir` if you used one).**
```bash
docker run --rm --runtime nvidia \
  -v "$REPO":/workspace/VRI_2026 -v "$R":/workspace/arm2_results \
  -w /workspace/VRI_2026/host_software \
  arm2-t5:r36.4.0 \
  python3 ml_jetson_vla/deployment/run_arm2_sweep_jetson.py \
    --stage export --results-dir /workspace/arm2_results --run-label jetson_run1
```
Prints the headline candidate x variant table (with the two no-model `[ref]` yardsticks: constant
centre guess, uniform-random), the coordinate-space diagnostic, and writes
`$R/arm2_colab_sweep_results_jetson_run1.json` (+ per-candidate scorer-format JSON + CSV) on the
**host**, since `$R` is bind-mounted. Each candidate's stage record in that file carries the
environment (image) that produced it.

**Interrupt/resume**: `Ctrl-C`, `docker stop`, or a host reboot at any point; rerun the same `docker
run` command. Finished calls are skipped (a fully finished candidate is not even loaded); the
container being `--rm` loses nothing because the checkpoint lives on the bind-mounted `$R`, not
inside the container's writable layer. Failed calls are never stored as done.

## 5. What still works unchanged inside a container, and what doesn't

**Unchanged, confirmed by re-reading both files against the container plan (no code changes made):**
- All eight stages (`probe`, `frames`, `hfauth`, `smoke`, `calibrate`, `validate`, `sweep`, `export`,
  and `all`) are plain local-filesystem Python — no venv-specific paths, no `sys.prefix` assumptions
  that break under a container's own `sys.prefix` (the probe's `in_venv`/`pyvenv.cfg` check just
  reports `False`/absent inside a container, which is correct and informational, not a failure mode).
- `probe_environment()`'s `/proc/device-tree/model` and `/etc/nv_tegra_release` reads work inside a
  plain container with no special mount: `/proc` is kernel-generated per-namespace, not a host bind
  mount, so Jetson model identification works the same as bare metal.
- `--stage export` (`stage_export()` → `_load_engine()` → `colab_sweep`) never calls a backend's
  `load()`, so it never imports `torch`/`transformers` at all — confirmed by reading
  `core/vlm_backends.py`: every `import torch` / `from transformers import ...` is inside a `load()`
  method, not at module level. This is why step 7 above can run from either image interchangeably.
- The coordinate-space calibration stage (`coord_space_probe.py`) imports only `numpy` and the policy
  module at its own top level — same story, no container-specific concern.

**Different under a container, noted so it isn't mistaken for a bug:**
- `nvpmodel -q`/`-m 0` and `jetson_clocks` are host L4T system tools, **not present inside
  `dustynv/l4t-pytorch`**. `probe_environment()` already handles this gracefully
  (`shutil.which("nvpmodel")` returns `None` inside the container, so `info["jetson"]["nvpmodel"]`
  is simply absent from a container-run probe, no crash) — but it means the nvpmodel mode a
  container-run sweep records is whatever the **host** was set to at container start (section 4 step 0
  and step 5 both run `nvpmodel`/`jetson_clocks` on the host for this reason), not something a
  container invocation can read or set itself.
- The `--results-dir` default (`DEFAULT_RESULTS_DIR`, a path relative to the script's own location
  inside the repo) would land inside the container's ephemeral filesystem if `--results-dir` is
  omitted and the repo isn't the only thing bind-mounted — **always pass `--results-dir
  /workspace/arm2_results`** (or wherever `$R` is mounted) explicitly, as every command above does;
  don't rely on the default.
- HF token caching (`~/.cache/huggingface/token`) only persists across separate `docker run`
  invocations because that directory is bind-mounted (section 4); without the `-v
  $HFCACHE:/root/.cache/huggingface` mount, every `--rm` container would need to re-authenticate.

## 6. How to read the result (unchanged from the Colab framing)

n = 60 (6 per session, 10 sessions); a hit-rate difference under ~10 points is within noise, per-session
spread matters more than the pooled number, and the `[ref]` rows are the yardstick (the constant-centre
guess already scores 25.2 mm mean / 9 hits locally). One run, one seed of prompt wording: not a ranking
(`model-iteration-constraints`). Latency/memory numbers are now the Orin's; record the `nvpmodel` mode
(it is captured in each candidate's `env.jetson.nvpmodel` when readable — from the host, per section 5).

## 7. What was verified where

**Verified on the dev machine (real code, mock models, no Docker/Jetson):**
`deployment/test_run_arm2_sweep_jetson_mock.py` — environment probe + static candidate rules;
preflight dropping only incompatible candidates; the reference-locked session set + parity assertion;
a full mock run of all six specs; the two-invocation split sharing one checkpoint (union identical to
a single run — the same mechanism the two-container split above relies on); a **real hard process
kill mid-sweep** followed by resume (checkpoint valid JSON, finished calls not redone, final items
identical to the uninterrupted run); the gate failing on an oracle mock, passing when fed the
reference outputs, `--force-continue` recorded as forced, and a sweep refused (before any model work)
without a passing gate record. Plus `py_compile` on every file this plan touches (including the two
Dockerfiles' referenced requirements files) and a re-run of the mock test suite, both after this
session's edits, to confirm nothing in the driver/probe logic broke. The shared engine's own 31-check
test (`test_colab_sweep_mock.py`) is unchanged.

**Not verified (needs the Jetson, and now needs Docker specifically)**: that either image actually
builds (the two Dockerfiles' own build-time asserts are a best-effort substitute, not a real build);
that any real model loads on aarch64 under either transformers range once built; that InternVL2.5's
remote code runs on transformers 4.56+ or 5.17.0; whether `timm`/`einops` are the only extras needed;
that Qwen reproduces the local reference under `transformers==5.17.0` on this GPU (the gate); real
latency/memory; the HF gated flow through a bind-mounted cache; on-device OpenCV/PyAV behaviour
inside a container; whether the base image's actual numpy version resolves cleanly against
`opencv-contrib-python-headless` under `--only-binary=:all:` (expected to, per the wheel-availability
audit in `requirements-jetson-arm2-common.txt`, but not run); real combined disk usage against the
image-size estimate in section 2; whether `--runtime nvidia` is actually needed or already the
daemon default on this device.
