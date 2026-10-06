"""Per-chunk inference latency for the Arm 2 action heads (SmolVLA, ACT) on the Jetson AGX Orin.

Measures one question: does one `predict_action_chunk` call fit inside the execution time of the chunk it produces
at 30 Hz (p90 latency < chunk_len / 30 s)? Latency only. ACT weights are RANDOM (no general pretrained ACT checkpoint
exists), so ACT numbers say nothing about accuracy. Refuses to run without CUDA and writes `status: failed` JSON
(exit 1) on any error, never a number it did not measure. Runs inside deployment/Dockerfile.arm2-lerobot.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

CONTROL_HZ: float = 30.0
SMOLVLA_REPO_ID: str = "lerobot/smolvla_base"
ACT_DEFAULT_CHUNK_LEN: int = 100  # chunk length used in the Orin Nano ACT latency paper
ACT_IMAGE_HW: tuple[int, int] = (480, 640)  # native camera/LeRobot frame; ACT config has no resize step
STATE_DIM: int = 2  # [touch_x_mm, touch_y_mm]
TARGET_DIM: int = 2  # [target_x_mm, target_y_mm], the custom head added by bench_hybrid_qwen_act.py
ACTION_DIM: int = 3  # [theta_a, theta_b, theta_c] degrees
PLATFORM_W_MM: float = 187.5  # CLAUDE.md platform dimensions, source of truth
PLATFORM_H_MM: float = 142.0
IMAGE_KEY: str = "observation.image"
STATE_KEY: str = "observation.state"
STATE_NAMES: tuple[str, ...] = ("touch_x_mm", "touch_y_mm")
TARGET_NAMES: tuple[str, ...] = ("target_x_mm", "target_y_mm")
LANG_TOKENS_KEY: str = "observation.language.tokens"  # same strings as lerobot.utils.constants
LANG_MASK_KEY: str = "observation.language.attention_mask"
DUMMY_INSTRUCTION: str = "go red"
DTYPE_NAMES: dict[str, str] = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def chunk_duration_s(chunk_len: int, control_hz: float = CONTROL_HZ) -> float:
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
    return chunk_len / control_hz


def summarize_latencies_ms(samples_ms: Sequence[float]) -> dict[str, float]:
    if len(samples_ms) == 0:
        raise ValueError("no latency samples to summarize")
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": float(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def inference_fits_chunk(p90_ms: float, chunk_len: int, control_hz: float = CONTROL_HZ) -> bool:
    return p90_ms < chunk_duration_s(chunk_len, control_hz) * 1000.0


def implied_replan_hz(mean_ms: float) -> float:
    if mean_ms <= 0.0:
        raise ValueError(f"mean latency must be > 0, got {mean_ms}")
    # Sustained replan rate if inference runs back to back, using the mean (p90 is what fits_chunk uses).
    return 1000.0 / mean_ms


def manifest_mm_to_telemetry_mm(
    x_manifest_mm: float, y_manifest_mm: float,
    platform_w_mm: float = PLATFORM_W_MM, platform_h_mm: float = PLATFORM_H_MM,
) -> tuple[float, float]:
    # Inverse of score_minimal_baseline_offline.touch_frame_to_manifest_mm (x mirrored, y offset by H/2).
    return platform_w_mm / 2.0 - x_manifest_mm, y_manifest_mm - platform_h_mm / 2.0


def pixel_to_telemetry_mm(
    homography_mm_to_px: np.ndarray, px: float, py: float, coord_space: str,
    model_input_hw: Optional[tuple[int, int]], raw_hw: tuple[int, int],
) -> tuple[float, float]:
    # Same chain as score_minimal_baseline_offline.score_prediction: model space -> raw px -> manifest mm -> telemetry.
    from ml_jetson_vla.core.minimal_vlm_policy import to_raw_px
    from ml_jetson_vla.deployment.score_minimal_baseline_offline import raw_px_to_mm

    rx, ry = to_raw_px(px, py, coord_space, model_input_hw, raw_hw)
    mx, my = raw_px_to_mm(homography_mm_to_px, rx, ry)
    return manifest_mm_to_telemetry_mm(mx, my)


def build_act_feature_spec(with_target: bool, image_hw: tuple[int, int]) -> dict[str, Any]:
    # Plain data, no lerobot import. with_target appends the slow-loop target to the state vector: the only
    # non-pretrained addition to ACT's input, because pretrained ACT has no target input.
    h, w = image_hw
    state_names = list(STATE_NAMES) + (list(TARGET_NAMES) if with_target else [])
    return {
        "input": {
            IMAGE_KEY: {"type": "VISUAL", "shape": (3, h, w)},
            STATE_KEY: {"type": "STATE", "shape": (len(state_names),), "names": state_names},
        },
        "output": {"action": {"type": "ACTION", "shape": (ACTION_DIM,)}},
    }


def refuse_overwrite(out_path: str, force: bool) -> None:
    if os.path.exists(out_path) and not force:
        raise FileExistsError(f"{out_path} exists; pass --force to overwrite")


def write_json_atomic(out_path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, out_path)


def build_act_policy(chunk_len: int, with_target: bool) -> tuple[Any, Any]:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    spec = build_act_feature_spec(with_target, ACT_IMAGE_HW)
    input_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"])) for k, v in spec["input"].items()
    }
    output_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"])) for k, v in spec["output"].items()
    }
    # pretrained_backbone_weights=None: the lerobot default downloads ImageNet ResNet18 weights, which would make
    # this "random init" claim false.
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_len,
        n_action_steps=chunk_len,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(cfg), cfg


def load_smolvla(chunk_len_override: Optional[int]) -> tuple[Any, Any, str]:
    from huggingface_hub import model_info
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    sha = model_info(SMOLVLA_REPO_ID).sha
    if not sha:
        raise RuntimeError(f"could not resolve a commit hash for {SMOLVLA_REPO_ID}")
    cfg = PreTrainedConfig.from_pretrained(SMOLVLA_REPO_ID, revision=sha)
    if chunk_len_override is not None:
        cfg.chunk_size = chunk_len_override
        cfg.n_action_steps = min(cfg.n_action_steps, chunk_len_override)
    policy = SmolVLAPolicy.from_pretrained(SMOLVLA_REPO_ID, config=cfg, revision=sha)
    return policy, cfg, sha


def build_act_batch(cfg: Any, dtype: "torch.dtype", device: str) -> dict[str, Any]:
    import torch

    batch: dict[str, Any] = {}
    for key, feat in cfg.image_features.items():
        batch[key] = torch.rand((1, *feat.shape), device=device, dtype=dtype)
    state_feat = cfg.robot_state_feature
    batch[STATE_KEY] = torch.zeros((1, *state_feat.shape), device=device, dtype=dtype)
    return batch


def build_smolvla_batch(cfg: Any, dtype: "torch.dtype", device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    batch: dict[str, Any] = {}
    for key, feat in cfg.image_features.items():
        batch[key] = torch.rand((1, *feat.shape), device=device, dtype=dtype)
    if cfg.robot_state_feature is not None:
        batch[STATE_KEY] = torch.zeros((1, *cfg.robot_state_feature.shape), device=device, dtype=dtype)
    # Real tokenizer of the checkpoint's VLM, so token count and padding match the policy config.
    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    tok = tokenizer(
        [DUMMY_INSTRUCTION], padding=cfg.pad_language_to, max_length=cfg.tokenizer_max_length,
        truncation=True, return_tensors="pt",
    )
    batch[LANG_TOKENS_KEY] = tok["input_ids"].to(device)
    batch[LANG_MASK_KEY] = tok["attention_mask"].to(device=device, dtype=torch.bool)
    return batch


def time_chunk_calls(run_once: Callable[[], Any], warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        run_once()
    samples: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def run_policy_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import importlib.metadata as md

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- refusing to record non-device latency")
    dtype = getattr(torch, DTYPE_NAMES[args.precision])
    device = "cuda"
    chunk_source = "default" if args.chunk_len is None else "override"

    extra: dict[str, Any]
    if args.policy == "smolvla":
        policy, cfg, sha = load_smolvla(args.chunk_len)
        chunk_len = int(cfg.chunk_size)
        action_dim = int(cfg.action_feature.shape[0])
        policy = policy.to(device=device, dtype=dtype).eval()
        batch = build_smolvla_batch(cfg, dtype, device)
        # SmolVLA's sample_noise hardcodes float32; pass noise in the model dtype explicitly.
        noise = torch.normal(0.0, 1.0, size=(1, chunk_len, int(cfg.max_action_dim)), device=device, dtype=dtype)

        def run_once() -> Any:
            return policy.predict_action_chunk(batch, noise=noise)

        extra = {"hf_repo_id": SMOLVLA_REPO_ID, "hf_commit": sha, "vlm_model_name": cfg.vlm_model_name,
                 "random_init": False, "num_steps": int(cfg.num_steps), "pad_language_to": cfg.pad_language_to}
    else:
        chunk_len = int(args.chunk_len) if args.chunk_len is not None else ACT_DEFAULT_CHUNK_LEN
        action_dim = ACTION_DIM
        policy, cfg = build_act_policy(chunk_len, with_target=False)
        policy = policy.to(device=device, dtype=dtype).eval()
        batch = build_act_batch(cfg, dtype, device)

        def run_once() -> Any:
            return policy.predict_action_chunk(batch)

        extra = {"random_init": True, "accuracy_meaningful": False, "normalizer_applied": False,
                 "act_input_image_hw": list(ACT_IMAGE_HW), "act_state_dim": STATE_DIM,
                 "act_use_vae": bool(cfg.use_vae), "pretrained_backbone_weights": None}

    out = run_once()
    out_shape = [int(d) for d in out.shape]
    if out_shape != [1, chunk_len, action_dim]:
        raise RuntimeError(f"unexpected output shape {out_shape}, expected [1, {chunk_len}, {action_dim}]")
    output_all_finite = bool(torch.isfinite(out).all().item())

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = time_chunk_calls(run_once, args.warmup, args.iters)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    summary = summarize_latencies_ms(samples)
    return {
        "policy": args.policy,
        "precision": args.precision,
        "dtype": DTYPE_NAMES[args.precision],
        "chunk_len": chunk_len,
        "chunk_len_source": chunk_source,
        "control_hz": CONTROL_HZ,
        "chunk_duration_s": chunk_duration_s(chunk_len),
        "device_name": torch.cuda.get_device_name(0),
        "param_count": int(sum(p.numel() for p in policy.parameters())),
        "latency_ms_per_chunk": summary,
        "implied_replan_hz": implied_replan_hz(summary["mean_ms"]),
        "inference_fits_chunk": inference_fits_chunk(summary["p90_ms"], chunk_len),
        "peak_gpu_mem_gb": peak_gb,
        "output_shape": out_shape,
        "output_all_finite": output_all_finite,
        "warmup": args.warmup,
        "iters": args.iters,
        "lerobot_version": md.version("lerobot"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        **extra,
    }


def base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bench": "bench_action_models",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "policy": args.policy,
        "precision": args.precision,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-chunk latency for SmolVLA / ACT on the Jetson.")
    parser.add_argument("--policy", required=True, choices=["smolvla", "act"])
    parser.add_argument("--precision", default="bf16", choices=sorted(DTYPE_NAMES))
    parser.add_argument("--chunk-len", type=int, default=None,
                        help="override chunk length; default = policy default (SmolVLA config, ACT 100)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing --out file")
    args = parser.parse_args(argv)
    if args.chunk_len is not None and args.chunk_len < 1:
        parser.error("--chunk-len must be >= 1")
    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be >= 1 and --warmup >= 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_path = os.path.abspath(args.out)
    try:
        refuse_overwrite(out_path, args.force)
    except FileExistsError as exc:
        print(f"[bench_action_models] {exc}")
        return 2

    result = base_result(args)
    try:
        result.update(run_policy_benchmark(args))
        result["status"] = "ok"
        code = 0
    except Exception as exc:  # recorded in the JSON, never a fabricated number
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        code = 1
    write_json_atomic(out_path, result)
    print(f"[bench_action_models] status={result['status']} wrote {out_path}")
    if result["status"] == "ok":
        lat = result["latency_ms_per_chunk"]
        print(f"  {args.policy} {result['precision']} chunk={result['chunk_len']}: p50={lat['p50_ms']:.1f}ms "
              f"p90={lat['p90_ms']:.1f}ms  chunk_duration={result['chunk_duration_s'] * 1000:.0f}ms  "
              f"fits={result['inference_fits_chunk']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
