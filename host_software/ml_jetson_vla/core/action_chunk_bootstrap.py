"""Video-bootstrapped prototype/validation harness for the decoupled, high-frequency
control subsystem (Track 4 TODO item 7). Read
`docs/ACTION_CHUNK_CONTROL_SUBSYSTEM_BOOTSTRAP.md` first -- this module is that doc's
code half, not a standalone design.

WHAT THIS IS: a cheap, closed-form/lightweight validation of the ACTION-CHUNKING
mechanism itself -- "given a window of W recent frames of already-vision-processed
state, predict a chunk of N future frames well enough to execute open-loop for that
whole chunk before replanning" -- run against the real Track 4 telemetry (10 sessions,
`session_jetson_track4_*`, see `data_processing/session_manifest.json`). This is NOT
another candidate-backbone comparison (that is `docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md`
Metric B's job, item 4/5) -- see the design doc S0 for the full scope argument. In one
line: Metric B asks "whose frozen representation best predicts ONE future point" (backbone
selection); this asks "does the chunk-and-execute-open-loop mechanism itself hold up as N
grows toward the 1-2s range the task specifies" (control-subsystem validation), which
Metric B's single-point target cannot answer at all.

WHAT THIS REUSES, PER THIS PROJECT'S "DON'T DESIGN FROM SCRATCH" CONSTRAINT:
  - The action-chunking mechanism itself (predict N future actions in one forward pass,
    execute open-loop) -- the same mechanism ACT/pi0/SmolVLA already ship, and the one
    `LARGE_VLA_RESEARCH_SPIKE.md`'s 2026-09-15 revision already selected over a
    from-scratch custom inner tier.
  - `qwen_multihead_policy.py`'s Head B (`StateFusionHead`) + Head C (`ActionChunkHead`)
    shapes VERBATIM for the neural variant (see `ChunkSurrogatePolicy` below) -- these are
    item 3's already-designed layer shapes, not reinvented here. The only new module is a
    `WindowEncoder` that stands in for Qwen's own pooled hidden state (see class docstring
    for why a stand-in is necessary and honestly scoped rather than claimed equivalent).
  - Metric B's own methodology (closed-form `sklearn.linear_model.Ridge`, session-level
    leave-one-session-out, per-`model-iteration-constraints`-skill discipline) for the
    primary (ridge) variant, extended from a single future point to a full chunk.

WHAT THIS DOES NOT DO: run the real Qwen2.5-VL-3B or SmolVLA backbone (not practical to
run at home without a GPU box, and not needed to validate the chunk-horizon premise --
see design doc S2), fine-tune anything for real, touch firmware, or run on hardware. The
`ChunkSurrogatePolicy` neural variant is an honest lightweight surrogate for item 3's
real multi-head design, not a claim that it equals Qwen's actual representation quality.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from qwen_multihead_policy import ActionChunkHead, MultiHeadConfig, StateFusionHead  # noqa: E402

_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_BRONZE_DIR = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
DEFAULT_REPORTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "reports")

SESSION_GLOB = "session_jetson_track4_*"
REQUIRED_COLUMNS = ["touch_x", "touch_y", "target_x", "target_y", "theta_a", "theta_b", "theta_c"]

FEATURE_SETS: Dict[str, List[str]] = {
    "state_only": ["touch_x", "touch_y"],
    "state_plus_target": ["touch_x", "touch_y", "target_x", "target_y"],
}
TARGET_SETS: Dict[str, List[str]] = {
    "touch_xy": ["touch_x", "touch_y"],
    "theta": ["theta_a", "theta_b", "theta_c"],
}


# ---------------------------------------------------------------------------
# Data loading -- reads telemetry.csv directly, no video decode needed. This is a
# deliberate choice, not a shortcut that dodges the task's "given video" framing:
# touch_x/y IS the vision pipeline's own already-extracted signal for these Track 4
# sessions (rows_match_video: true in the manifest, single-writer-thread design, zero
# NaN ground-truth rows -- confirmed by re-checking all 10 sessions this session), so a
# window of recent touch_x/y/theta values is exactly "the last W frames of video,
# already processed" rather than a stand-in for it. See the design doc S3 for the
# optional raw-pixel variant this deliberately does not build.
# ---------------------------------------------------------------------------


def load_sessions(bronze_dir: str = DEFAULT_BRONZE_DIR, pattern: str = SESSION_GLOB) -> Dict[str, pd.DataFrame]:
    session_dirs = sorted(glob.glob(os.path.join(bronze_dir, pattern)))
    sessions: Dict[str, pd.DataFrame] = {}
    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            continue  # skip the sibling .dvc pointer files matched by the same glob prefix
        csv_path = os.path.join(session_dir, "telemetry.csv")
        if not os.path.exists(csv_path):
            print(f"[action_chunk_bootstrap] WARNING: no telemetry.csv in {session_dir}, skipping")
            continue
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"[action_chunk_bootstrap] WARNING: {session_dir} missing columns {missing}, skipping")
            continue
        df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
        if len(df) == 0:
            continue
        sessions[os.path.basename(session_dir)] = df
    return sessions


# ---------------------------------------------------------------------------
# Windowing: [t-W+1 .. t] history -> [t+1 .. t+N] target chunk. Per
# MULTI_HEAD_ARCHITECTURE_SPEC.md S2.1's flagged off-by-one hazard, the chunk target
# NEVER includes frame t itself -- t is the last frame the "control subsystem" has
# actually observed when it commits to a chunk.
# ---------------------------------------------------------------------------


def build_windows(
    df: pd.DataFrame, window: int, chunk: int, feature_cols: List[str], target_cols: List[str], stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    values_feat = df[feature_cols].to_numpy(dtype=np.float32)
    values_targ = df[target_cols].to_numpy(dtype=np.float32)
    n_rows = len(df)
    xs, ys = [], []
    # t is the index of the last observed frame; valid t range: [window-1, n_rows-1-chunk]
    for t in range(window - 1, n_rows - chunk, stride):
        hist = values_feat[t - window + 1 : t + 1]  # [window, n_feat]
        fut = values_targ[t + 1 : t + 1 + chunk]  # [chunk, n_targ]
        xs.append(hist.reshape(-1))
        ys.append(fut.reshape(-1))
    if not xs:
        return np.zeros((0, window * len(feature_cols)), dtype=np.float32), np.zeros(
            (0, chunk * len(target_cols)), dtype=np.float32
        )
    return np.stack(xs), np.stack(ys)


def per_step_error(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, chunk: int, target_dim: int) -> np.ndarray:
    """Mean Euclidean error per chunk step k=0..chunk-1, averaged over samples.
    Returns shape [chunk]."""
    y_true = y_true_flat.reshape(-1, chunk, target_dim)
    y_pred = y_pred_flat.reshape(-1, chunk, target_dim)
    dist = np.linalg.norm(y_true - y_pred, axis=-1)  # [n_samples, chunk]
    return dist.mean(axis=0)


# ---------------------------------------------------------------------------
# Ridge variant (primary) -- mirrors BOOTSTRAP_MODEL_COMPARISON_PLAN.md Metric B's own
# closed-form-fit methodology (S2.2: "sklearn.linear_model.Ridge... not gradient-descent
# fine-tuning"), extended from a single-point target to a full chunk.
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    held_out_session: str
    n_train: int
    n_test: int
    per_step_error: List[float]  # length == chunk


@dataclass
class SweepResult:
    window: int
    chunk: int
    feature_set: str
    target_set: str
    folds: List[FoldResult] = field(default_factory=list)

    def mean_curve(self) -> List[float]:
        if not self.folds:
            return []
        arr = np.stack([f.per_step_error for f in self.folds])  # [n_folds, chunk]
        return arr.mean(axis=0).tolist()

    def std_curve(self) -> List[float]:
        if not self.folds:
            return []
        arr = np.stack([f.per_step_error for f in self.folds])
        return arr.std(axis=0).tolist()


def run_ridge_loso(
    sessions: Dict[str, pd.DataFrame],
    window: int,
    chunk: int,
    feature_set: str,
    target_set: str,
    alpha: float = 1.0,
    stride: int = 1,
) -> SweepResult:
    feature_cols = FEATURE_SETS[feature_set]
    target_cols = TARGET_SETS[target_set]
    target_dim = len(target_cols)

    per_session_xy = {
        name: build_windows(df, window, chunk, feature_cols, target_cols, stride=stride)
        for name, df in sessions.items()
    }
    result = SweepResult(window=window, chunk=chunk, feature_set=feature_set, target_set=target_set)

    for held_out in sessions:
        x_test, y_test = per_session_xy[held_out]
        if len(x_test) == 0:
            continue
        x_train = np.concatenate([per_session_xy[n][0] for n in sessions if n != held_out], axis=0)
        y_train = np.concatenate([per_session_xy[n][1] for n in sessions if n != held_out], axis=0)
        if len(x_train) == 0:
            continue

        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        curve = per_step_error(y_test, y_pred, chunk, target_dim)
        result.folds.append(
            FoldResult(
                held_out_session=held_out,
                n_train=len(x_train),
                n_test=len(x_test),
                per_step_error=curve.tolist(),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Neural surrogate (secondary) -- reuses qwen_multihead_policy.py's StateFusionHead
# (Head B) and ActionChunkHead (Head C) VERBATIM. The only new component is
# WindowEncoder, a single Linear standing in for Qwen's own pooled action-query hidden
# state -- honestly scoped as a cheap stand-in, not a claim that a flattened window of
# 2-4 scalars over W frames carries the same information as a real vision-language
# backbone's representation. This variant exists to answer a narrower question than the
# ridge variant: "do item 3's actual Head B/C shapes (nonlinear, tanh-bounded,
# theta-space) behave sanely end-to-end against real data and a real chunk target,
# before any Colab/GPU time is spent wiring them to the real Qwen backbone." Only run at
# one representative (window, chunk) setting by default -- NOT swept the way the ridge
# variant is -- to keep this runnable on a CPU at home in reasonable time; see the design
# doc for why the full sweep is left to the ridge variant.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class WindowEncoder(nn.Module):
    """Stand-in for Qwen's pooled action-query hidden state (see module docstring).
    Flattens the [window, n_feat] history and projects to qwen_hidden_size -- nothing
    more. Not claimed to be architecturally equivalent to a real VLM forward pass."""

    def __init__(self, window_input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(window_input_dim, hidden_size)

    def forward(self, window_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(window_feat)


class ChunkSurrogatePolicy(nn.Module):
    """Wires WindowEncoder -> [StateFusionHead, ActionChunkHead] the same way
    QwenMultiHeadPolicy wires [qwen pooled hidden, StateFusionHead] -> ActionChunkHead
    (qwen_multihead_policy.py). `state` here is the single-latest-frame state vector
    (matches Head B's real Phase-1/Phase-2 contract exactly); `window_feat` is the
    flattened W-frame history (the new, task-specific input this harness adds)."""

    def __init__(self, window_input_dim: int, config: MultiHeadConfig) -> None:
        super().__init__()
        self.window_encoder = WindowEncoder(window_input_dim, config.qwen_hidden_size)
        self.state_head = StateFusionHead(config.state_dim, config.qwen_hidden_size)
        self.action_head = ActionChunkHead(config)

    def forward(self, window_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pooled = self.window_encoder(window_feat)
        state_embed = self.state_head(state)
        fused = torch.cat([pooled, state_embed], dim=-1)
        return self.action_head(fused)  # [B, chunk, 3] degrees, tanh-bounded


def run_neural_loso(
    sessions: Dict[str, pd.DataFrame],
    window: int,
    chunk: int,
    feature_set: str = "state_plus_target",
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 0,
    train_cap: int = 4000,
    batch_size: int = 512,
) -> SweepResult:
    """theta-space only -- ActionChunkHead's tanh*angle_limit_deg bounding is designed
    for RLControl.cpp's degree-space action convention (qwen_multihead_policy.py S3.3),
    not mm-space ball position, so this variant always targets theta_a/b/c.

    COMPUTE-SCOPING NOTE (found the hard way, real run this session): `ActionChunkHead`
    reuses item 3's real shape verbatim, which includes a Linear(4096,512) sized for
    Qwen's actual 2048-dim hidden state -- that layer is NOT cheap when run full-batch
    against ~36K training rows for 30 epochs x 10 folds (measured: a real, unbounded run
    was still going after 10 minutes wall-clock / ~1.5 CPU-hours, confirmed via
    `Get-CimInstance Win32_Process` against the actual PID, not a guess -- ballparked at
    ~150 TFLOPs total for just one (window, chunk) setting). `train_cap` (random
    subsample per fold) and a mini-batch loop below are a deliberate, explicit
    compute-scoping choice to keep this variant runnable on a CPU at home in well under a
    minute -- NOT a claim that 4,000 rows / 10 epochs is enough to actually fit this head
    well; the ridge variant (full data, closed-form) remains the harness's primary,
    non-subsampled result for exactly that reason."""
    feature_cols = FEATURE_SETS[feature_set]
    target_cols = TARGET_SETS["theta"]
    target_dim = len(target_cols)

    per_session_xy = {
        name: build_windows(df, window, chunk, feature_cols, target_cols) for name, df in sessions.items()
    }
    # Single-latest-frame state vector = last row of the window's touch_x/touch_y
    # (Head B's real, already-designed 2-dim Phase-1 contract).
    per_session_state = {
        name: df[["touch_x", "touch_y"]].to_numpy(dtype=np.float32)[window - 1 : len(df) - chunk]
        for name, df in sessions.items()
    }

    config = MultiHeadConfig(state_dim=2, chunk_size=chunk)
    result = SweepResult(window=window, chunk=chunk, feature_set=feature_set, target_set="theta")

    torch.manual_seed(seed)
    for held_out in sessions:
        x_test, y_test = per_session_xy[held_out]
        s_test = per_session_state[held_out]
        if len(x_test) == 0:
            continue
        x_train = np.concatenate([per_session_xy[n][0] for n in sessions if n != held_out], axis=0)
        y_train = np.concatenate([per_session_xy[n][1] for n in sessions if n != held_out], axis=0)
        s_train = np.concatenate([per_session_state[n] for n in sessions if n != held_out], axis=0)
        if len(x_train) == 0:
            continue

        # Compute-scoping: subsample to train_cap rows (see docstring) -- ActionChunkHead's
        # real Linear(4096,512) shape makes full-batch training over ~36K rows too slow for
        # a "prototype at home on a CPU" harness (measured, see docstring).
        rng = np.random.default_rng(seed)
        if len(x_train) > train_cap:
            idx = rng.choice(len(x_train), size=train_cap, replace=False)
            x_train, y_train, s_train = x_train[idx], y_train[idx], s_train[idx]

        policy = ChunkSurrogatePolicy(window_input_dim=x_train.shape[1], config=config)
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        xt = torch.from_numpy(x_train)
        st = torch.from_numpy(s_train)
        yt = torch.from_numpy(y_train).view(-1, chunk, target_dim)
        n_train = len(x_train)

        for _ in range(epochs):
            perm = rng.permutation(n_train)
            for start in range(0, n_train, batch_size):
                batch_idx = perm[start : start + batch_size]
                opt.zero_grad()
                pred = policy(xt[batch_idx], st[batch_idx])
                loss = torch.mean((pred - yt[batch_idx]) ** 2)
                loss.backward()
                opt.step()

        policy.eval()
        with torch.no_grad():
            pred_test = policy(torch.from_numpy(x_test), torch.from_numpy(s_test)).numpy()
        curve = per_step_error(y_test, pred_test.reshape(len(x_test), -1), chunk, target_dim)
        result.folds.append(
            FoldResult(
                held_out_session=held_out,
                n_train=len(x_train),
                n_test=len(x_test),
                per_step_error=curve.tolist(),
            )
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-bootstrapped validation of the action-chunking control subsystem (item 7)."
    )
    parser.add_argument("--bronze-dir", type=str, default=DEFAULT_BRONZE_DIR)
    parser.add_argument(
        "--window-sweep",
        type=int,
        nargs="+",
        default=[1, 5, 15, 30],
        help="History window lengths, in frames @30fps. W=1 matches Head B's real "
        "single-frame state contract; larger W tests whether the task's own 'given 10s "
        "of video' multi-frame framing helps over that.",
    )
    parser.add_argument(
        "--chunk-sweep",
        type=int,
        nargs="+",
        default=[5, 10, 15, 30, 60],
        help="Chunk horizons, in frames @30fps. N=10 (~333ms) is item 3's own starting "
        "point; N=30/60 (~1-2s) is this task's own framing.",
    )
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="state_plus_target")
    parser.add_argument("--target-set", choices=list(TARGET_SETS), default="touch_xy")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=2, help="Window stride, reduces adjacent-window overlap cost.")
    parser.add_argument(
        "--run-neural", action="store_true", help="Also run the neural (item-3-head-reuse) surrogate at one setting."
    )
    parser.add_argument("--neural-window", type=int, default=15)
    parser.add_argument("--neural-chunk", type=int, default=30)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    sessions = load_sessions(args.bronze_dir)
    if not sessions:
        raise SystemExit(f"No usable '{SESSION_GLOB}' sessions found under {args.bronze_dir}")
    print(f"[action_chunk_bootstrap] Loaded {len(sessions)} sessions: {sorted(sessions)}")

    ridge_results = []
    for window in args.window_sweep:
        for chunk in args.chunk_sweep:
            sr = run_ridge_loso(
                sessions,
                window=window,
                chunk=chunk,
                feature_set=args.feature_set,
                target_set=args.target_set,
                alpha=args.ridge_alpha,
                stride=args.stride,
            )
            curve = sr.mean_curve()
            if curve:
                print(
                    f"[ridge] W={window:3d} N={chunk:3d}  step1_err={curve[0]:.2f}  "
                    f"stepN_err={curve[-1]:.2f}  mean_err={np.mean(curve):.2f}  "
                    f"(mm, {args.target_set}, {len(sr.folds)} folds)"
                )
            ridge_results.append(sr)

    neural_results = []
    if args.run_neural:
        print(f"\n[neural] Running ChunkSurrogatePolicy at W={args.neural_window} N={args.neural_chunk} (theta target)...")
        nr = run_neural_loso(sessions, window=args.neural_window, chunk=args.neural_chunk)
        curve = nr.mean_curve()
        if curve:
            print(
                f"[neural] W={args.neural_window} N={args.neural_chunk}  step1_err={curve[0]:.3f}  "
                f"stepN_err={curve[-1]:.3f}  mean_err={np.mean(curve):.3f}  (deg, theta, {len(nr.folds)} folds)"
            )
        neural_results.append(nr)

    out_path = args.out
    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.makedirs(DEFAULT_REPORTS_DIR, exist_ok=True)
        out_path = os.path.join(DEFAULT_REPORTS_DIR, f"action_chunk_bootstrap_{stamp}.json")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sessions_used": sorted(sessions),
        "n_sessions": len(sessions),
        "args": vars(args),
        "ridge_sweep": [
            {
                "window": r.window,
                "chunk": r.chunk,
                "feature_set": r.feature_set,
                "target_set": r.target_set,
                "n_folds": len(r.folds),
                "mean_curve_mm_or_deg": r.mean_curve(),
                "std_curve_mm_or_deg": r.std_curve(),
                "folds": [f.__dict__ for f in r.folds],
            }
            for r in ridge_results
        ],
        "neural_surrogate": [
            {
                "window": r.window,
                "chunk": r.chunk,
                "feature_set": r.feature_set,
                "target_set": r.target_set,
                "n_folds": len(r.folds),
                "mean_curve_deg": r.mean_curve(),
                "std_curve_deg": r.std_curve(),
                "folds": [f.__dict__ for f in r.folds],
            }
            for r in neural_results
        ],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[action_chunk_bootstrap] Wrote results to {out_path}")


if __name__ == "__main__":
    main()
