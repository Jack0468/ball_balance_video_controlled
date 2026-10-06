"""LoRA-load smoke test for Arm 2 (Qwen2.5-VL-3B-Instruct) on the Jetson AGX Orin -- Risk #7 in
`docs/MULTI_HEAD_ARCHITECTURE_SPEC.md`: "test plain-PyTorch LoRA load on the Jetson today as the baseline
path". Runs INSIDE the `arm2-t5` container (Dockerfile.arm2-transformers5), not on the Windows dev box.

NOT A TRAINING SCRIPT. It does one forward pass (no_grad, eval mode) and one save/reload. Nothing here calls
`backward()` or an optimizer. Two deliberate choices, both stated so they are not mistaken for training:
  - LoRA's `lora_B` starts at zero, so an untouched adapter is a no-op and a reload check would be trivially
    true. `lora_B` is filled with N(0, 1e-3) before the save so the adapter has a real, non-zero effect and
    the reload comparison means something. This is a random perturbation for testing, not a fitted value.
  - Paths are scratch only: `--out-dir` defaults to `/workspace/arm2_results/lora_smoke/<timestamp>/`. The
    script refuses any out-dir that touches `jetson_run1` (the sweep's checkpoint namespace) and never takes
    a run label.

Steps (each prints wall-clock time and peak GPU memory):
  1. load the base model via the sweep's own loader (`qwen_vl_smoke_test.load_qwen_vl_model`, dtype from
     `vlm_backends.resolve_torch_dtype("auto")`, the same path `QwenVLBackend.load()` takes);
  2. attach a LoRA adapter (peft, r=8, targets q/k/v/o_proj in the LANGUAGE model only, vision tower excluded);
  3. one forward pass on one real Track 4 frame, reusing the sweep's frame helper
     (`colab_sweep.prepare_frames`); prints logits shape and a finite-loss check. Also prints how far the
     adapter moves the logits from the base model (proves the adapter is live);
  4. save the adapter, free the model, reload the base model fresh, attach the saved adapter with
     `PeftModel.from_pretrained`, re-run the same forward, print max |logit diff| vs. the pre-save logits.

Usage (inside the container, from /workspace/VRI_2026):
    python3 ml_jetson_vla/deployment/lora_load_smoke_test.py
    python3 ml_jetson_vla/deployment/lora_load_smoke_test.py --session session_jetson_track4_20260915_151627
Exit code 0 only if the loss is finite and the reloaded logits match within --logit-atol.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR: str = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

from ml_jetson_vla.core.vlm_backends import resolve_torch_dtype  # noqa: E402
from ml_jetson_vla.deployment.qwen_vl_smoke_test import load_qwen_vl_model  # noqa: E402
from ml_jetson_vla.deployment import colab_sweep as cs  # noqa: E402

DEFAULT_BRONZE_DIR: str = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
SESSION_SPLIT_PATH: str = os.path.join(_ML_JETSON_VLA_DIR, "data_processing", "session_split.json")
DEFAULT_OUT_ROOT: str = "/workspace/arm2_results/lora_smoke"
QWEN_REPO_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_REVISION: str = "66285546d2b821cf421d4f5eb2576359d3770cd3"  # the Arm 2 sweep's pinned revision
# Same pixel budget QwenVLBackend uses (core/vlm_backends.py), so this forward matches the sweep's input size.
MIN_PIXELS: int = 64 * 28 * 28
MAX_PIXELS: int = 256 * 28 * 28
# Revision the local checkpoint was downloaded at (vlm_backends.QwenVLBackend docstring). Printed for the
# record (risk #5); the local-path load does not take a revision argument.
EXPECTED_REVISION: str = "66285546d2b821cf421d4f5eb2576359d3770cd3"
PROMPT: str = "Which colour target should the ball be moved to?"
LORA_R: int = 8
LORA_ALPHA: int = 16
# Language-model attention projections only: the negative lookahead keeps the vision tower's modules out
# (its attention is named differently, but the guard makes the intent explicit and survives naming changes).
LORA_TARGET_REGEX: str = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj)$"


class StepTimer:
    """Wall-clock + peak GPU memory per step. Peak is reset at each step start so numbers don't bleed."""

    def __init__(self) -> None:
        self.cuda = torch.cuda.is_available()
        self.records: List[Dict[str, object]] = []

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        yield
        if self.cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb: Optional[float] = torch.cuda.max_memory_allocated() / 1e9 if self.cuda else None
        self.records.append({"step": name, "wall_s": round(dt, 3), "peak_gpu_gb": peak_gb})
        mem = f"{peak_gb:.2f} GB" if peak_gb is not None else "n/a (CPU)"
        print(f"    [{name}] wall={dt:.2f}s peak_gpu={mem}", flush=True)


def pick_default_session() -> str:
    """First Track 4 session in the TRAIN list of session_split.json (the smoke test does not train,
    so this is only about which real frame gets pushed through; train-side is the conventional choice)."""
    with open(SESSION_SPLIT_PATH, "r") as f:
        split = json.load(f)
    for name in split["train"]:
        if name.startswith("session_jetson_track4_"):
            return str(name)
    raise RuntimeError(f"No Track 4 session in the train list of {SESSION_SPLIT_PATH}")


def build_inputs(processor: object, device: str, dtype: Optional[torch.dtype],
                 image_rgb: np.ndarray, answer: str) -> Tuple[Dict[str, torch.Tensor], int]:
    """Teacher-forced input for a loss: user turn (image + prompt) then assistant answer. Labels are the
    answer tokens only; returns (inputs, number_of_prompt_tokens) so the caller can mask the prompt."""
    # PIL, not a raw ndarray: qwen_vl_utils' documented image contract is path/URL/base64/PIL.
    pil_image = Image.fromarray(image_rgb)
    user_msg = [{"role": "user", "content": [{"type": "image", "image": pil_image},
                                             {"type": "text", "text": PROMPT}]}]
    full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    prompt_text = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(user_msg)

    prompt_inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                              padding=True, return_tensors="pt")
    n_prompt = int(prompt_inputs["input_ids"].shape[1])
    full_inputs = processor(text=[full_text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt")

    out: Dict[str, torch.Tensor] = {}
    for k, v in full_inputs.items():
        t = v.to(device)
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        out[k] = t
    labels = out["input_ids"].clone()
    labels[:, :n_prompt] = -100
    out["labels"] = labels
    return out, n_prompt


def forward_logits_and_loss(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits.detach().float().cpu(), out.loss.detach().float().cpu()


def attach_lora(base: torch.nn.Module) -> torch.nn.Module:
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
                     target_modules=LORA_TARGET_REGEX, task_type="CAUSAL_LM")
    peft_model = get_peft_model(base, cfg)
    return peft_model


def lm_num_layers(model: torch.nn.Module) -> int:
    cfg = model.config
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return int(text_cfg.num_hidden_layers)


def nonzero_lora_b(peft_model: torch.nn.Module, std: float, seed: int) -> int:
    """Fill lora_B with a small random value so the adapter is non-trivial (see module docstring)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name and p.requires_grad:
                noise = torch.randn(p.shape, generator=g, dtype=torch.float32) * std
                p.copy_(noise.to(device=p.device, dtype=p.dtype))
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA load/save/reload smoke test for Qwen2.5-VL-3B (no training).")
    ap.add_argument("--model-dir", default=QWEN_REPO_ID,
                    help="Local Qwen2.5-VL-3B-Instruct checkpoint (default: the sweep's own default).")
    ap.add_argument("--revision", default=QWEN_REVISION,
                    help="HF commit to pin (the Arm 2 sweep's own revision). Ignored for local paths.")
    ap.add_argument("--bronze-dir", default=DEFAULT_BRONZE_DIR)
    ap.add_argument("--session", default=None,
                    help="Track 4 session dir name. Default: first Track 4 session in session_split.json's train list.")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help="Scratch root. Must not be under a jetson_run1 path.")
    ap.add_argument("--init-std", type=float, default=1e-3, help="lora_B N(0, std) fill for a non-trivial adapter.")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--logit-atol", type=float, default=0.0,
                    help="Max allowed |logit diff| between pre-save and reloaded adapter. Default 0 = identical.")
    ap.add_argument("--answer", default="go_red", help="Teacher-forced answer text for the loss.")
    args = ap.parse_args()

    if "jetson_run1" in os.path.normpath(args.out_root):
        print("REFUSED: --out-root must not point into jetson_run1 checkpoints.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype, dtype_name = resolve_torch_dtype("auto", device)
    session = args.session or pick_default_session()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, stamp)
    adapter_dir = os.path.join(out_dir, "adapter")

    print(f"device={device} dtype={dtype_name} model_dir={args.model_dir} expected_revision={EXPECTED_REVISION}")
    print(f"session={session} out_dir={out_dir}")
    timer = StepTimer()
    results: Dict[str, object] = {}

    # Frame: one real Track 4 frame via the sweep's own helper (BGR -> RGB, as policy.act() does).
    frames, skipped = cs.prepare_frames(args.bronze_dir, frames_per_session=2, sessions=[session], verbose=True)
    if not frames:
        print(f"No usable frame from {session} (skipped: {skipped})")
        return 1
    image_rgb = np.ascontiguousarray(frames[0].frame_bgr[..., ::-1])
    print(f"frame: session={frames[0].session} frame_index={frames[0].frame_index} shape={image_rgb.shape}")

    # Step 1: base model.
    with timer.step("1_load_base"):
        base, processor, _dev = load_qwen_vl_model(args.model_dir, device=device, min_pixels=MIN_PIXELS,
                                                   max_pixels=MAX_PIXELS, dtype=torch_dtype,
                                                   revision=args.revision)
        base.eval()
    inputs, n_prompt = build_inputs(processor, device, torch_dtype, image_rgb, args.answer)

    # Step 2: attach LoRA (language-model attention only).
    with timer.step("2_attach_lora"):
        peft_model = attach_lora(base)
        n_lora = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A.default"))
        n_layers = lm_num_layers(base)
        expected = 4 * n_layers
        if n_lora != expected:
            raise RuntimeError(f"LoRA attached to {n_lora} modules, expected {expected} (4 projections x {n_layers} LM layers)")
        n_b = nonzero_lora_b(peft_model, args.init_std, args.init_seed)
        peft_model.eval()
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"    lora modules={n_lora} (expected {expected}), lora_B filled={n_b}, trainable params={trainable:,}")
        if any("visual" in n for n, _ in peft_model.named_modules() if "lora_A" in n):
            raise RuntimeError("LoRA leaked into the vision tower")

    # Step 3: one forward pass, finite-loss check, and adapter-effect check.
    with timer.step("3_forward"):
        logits_lora, loss_lora = forward_logits_and_loss(peft_model, inputs)
        with peft_model.disable_adapter():
            logits_base, _ = forward_logits_and_loss(peft_model, inputs)
        finite = bool(torch.isfinite(loss_lora).item()) and bool(torch.isfinite(logits_lora).all().item())
        base_vs_lora = float((logits_lora - logits_base).abs().max().item())
        print(f"    logits shape={tuple(logits_lora.shape)} (batch, seq, vocab)")
        print(f"    loss={float(loss_lora):.6f} finite={finite} n_prompt_tokens_masked={n_prompt}")
        print(f"    max |logits(adapter) - logits(base)| = {base_vs_lora:.6g} (0 would mean the adapter is inert)")
        results.update({"logits_shape": list(logits_lora.shape), "loss": float(loss_lora), "loss_finite": finite,
                        "max_abs_adapter_vs_base": base_vs_lora})
    if not finite:
        print("FAIL: loss or logits not finite")
        return 1

    # Step 4: save, free, reload on a fresh base, compare logits.
    os.makedirs(adapter_dir, exist_ok=True)
    with timer.step("4_save_adapter"):
        peft_model.save_pretrained(adapter_dir)
        print(f"    saved to {adapter_dir}: {sorted(os.listdir(adapter_dir))}")

    del peft_model, base
    if device == "cuda":
        torch.cuda.empty_cache()

    with timer.step("5_reload_fresh_base_and_compare"):
        fresh, _proc2, _dev2 = load_qwen_vl_model(args.model_dir, device=device, min_pixels=MIN_PIXELS,
                                                  max_pixels=MAX_PIXELS, dtype=torch_dtype)
        fresh.eval()
        reloaded = PeftModel.from_pretrained(fresh, adapter_dir)
        reloaded.eval()
        logits_reload, _ = forward_logits_and_loss(reloaded, inputs)
        max_diff = float((logits_reload - logits_lora).abs().max().item())
        print(f"    max |logits(reloaded) - logits(pre-save)| = {max_diff:.6g} (atol={args.logit_atol})")
        results["reload_max_abs_logit_diff"] = max_diff
        reload_ok = max_diff <= args.logit_atol

    print("\n=== timing summary ===")
    for r in timer.records:
        mem = f"{r['peak_gpu_gb']:.2f} GB" if r["peak_gpu_gb"] is not None else "n/a"
        print(f"  {r['step']:<34} wall={r['wall_s']:>8.2f}s  peak_gpu={mem}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke_summary.json"), "w") as f:
        json.dump({"session": session, "frame_index": int(frames[0].frame_index), "device": device,
                   "dtype": dtype_name, "expected_revision": EXPECTED_REVISION, "steps": timer.records,
                   **results, "reload_ok": reload_ok}, f, indent=2)

    print(f"\nRESULT: loss_finite={finite} reload_ok={reload_ok}")
    return 0 if (finite and reload_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
