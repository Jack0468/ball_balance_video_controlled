"""Per-candidate VLM backend adapters for the Arm 2 minimal-baseline multi-candidate
extension (`docs/ARM2_MINIMAL_BASELINE_SCOPE.md`, `docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`).

Each backend wraps exactly ONE candidate's real loading + generation call. This module is
what makes swapping the underlying model a constructor argument
(`MinimalVLMPolicy(backend=QwenVLBackend(...))`) rather than new code per model -- see
`core/minimal_vlm_policy.py`.

**What "verified against the real installed API" actually means per backend below** (per
this task's explicit "don't guess call shapes" instruction) -- stated plainly since the four
backends are NOT verified to the same depth, and pretending otherwise would misrepresent the
comparison:

- `QwenVLBackend`: FULLY verified AND executed (local CPU, transformers 5.17.0, bf16). Reuses
  `deployment/qwen_vl_smoke_test.py`'s `load_qwen_vl_model()`/`generate_qwen_vl()`.
- `InternVLBackend`, `PaliGemma2Backend`, `Moondream2Backend`: NEVER executed against real
  weights as of this writing (the Colab sweep, `deployment/arm2_colab_sweep.ipynb`, is what
  first executes them). **2026-09-19 static review of each candidate's real Hugging Face repo
  found real problems in the original (2026-09-18) versions of these classes, fixed below:**
    * InternVL: `OpenGVLab/InternVL2_5-4B` is a `trust_remote_code` repo with its own
      `model.chat(tokenizer, pixel_values, question, generation_config)` API and its own
      dynamic-tiling image preprocessing. There is NO official HF-native (`-hf`/`-HF`)
      conversion of the 4B checkpoint (official native conversions found: InternVL2_5-2B-MPO-hf,
      InternVL2_5-8B-MPO-hf, InternVL3-*-hf, InternVL3_5-*-HF). The 2026-09-18 version called
      `InternVLForConditionalGeneration.from_pretrained("OpenGVLab/InternVL2_5-4B")`, which
      would have failed on load. `InternVLBackend` now follows the model card's own API
      (preprocessing helpers reproduced from the card); the native-HF path is kept as
      `InternVLNativeBackend` (default repo `OpenGVLab/InternVL3_5-4B-HF`, a DIFFERENT model
      generation -- it is an optional fallback, not a substitute for the requested candidate).
    * PaliGemma2: `pixel_values` were left float32 while the weights are bf16/fp16 (dtype-mismatch
      error on the vision tower); `<loc...>` tokens might be dropped by `skip_special_tokens=True`;
      and the mix checkpoint is documented as task-prefix-prompted (`detect <thing>`), not
      free-chat -- so it now has two modes (see class docstring).
    * Moondream2: its remote code at the README-recommended revision fails on transformers>=5
      (missing `post_init()`, the `all_tied_weights_keys` AttributeError reported on the model's
      HF discussions) -- the Colab notebook therefore installs transformers<5. Revision pinned.
  Still NOT executed; the fixes are from reading the real repos/READMEs, not from running them.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import sys
import time
from typing import Optional, Protocol, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)


# --- dtype / memory helpers (shared) -------------------------------------------------------

def resolve_torch_dtype(name: Optional[str] = "auto", device: Optional[str] = None) -> Tuple[object, str]:
    """Returns `(torch_dtype_or_None, canonical_name)`.

    `"auto"`: CUDA with compute capability >= 8.0 (A100, L4, Ampere/Ada/Hopper, Jetson AGX Orin
    = 8.7) -> bfloat16; CUDA below 8.0 (T4 = 7.5, no native bf16) -> float16; CPU -> bfloat16
    (matches the 2026-09-18 local CPU run). Deliberately uses compute capability, NOT
    `torch.cuda.is_bf16_supported()`, which returns True on a T4 via slow emulation.
    `"bf16"`/`"fp16"`/`"fp32"` force one. `"default"` returns `(None, "default(fp32)")`, meaning
    "do not pass torch_dtype at all" (used for Moondream2, whose README-documented load call
    passes none)."""
    import torch

    key = (name or "auto").lower()
    if key in ("default", "native"):
        return None, "default(fp32)"
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16, "bf16"
    if key in ("fp16", "float16", "half"):
        return torch.float16, "fp16"
    if key in ("fp32", "float32"):
        return torch.float32, "fp32"
    if key != "auto":
        raise ValueError(f"unknown dtype {name!r}; use auto/bf16/fp16/fp32/default")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda") and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        return (torch.bfloat16, "bf16") if major >= 8 else (torch.float16, "fp16")
    return torch.bfloat16, "bf16"


def free_gpu_memory() -> None:
    """Best-effort release between candidates: collect garbage, then empty the CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # never let cleanup mask the real result/error
        pass


def _reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def _peak_gb(device: str) -> Optional[float]:
    if device.startswith("cuda"):
        import torch

        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return None


@dataclasses.dataclass
class BackendOutput:
    """Everything `MinimalVLMPolicy`'s parser needs, plus the raw timing/memory numbers the
    run/test plan's "real latency/memory numbers, no fabricated numbers" requirement asks
    for. `native_points` is populated ONLY by a backend whose real API already returns
    structured points (currently just `Moondream2Backend`) -- when set, the parser in
    `minimal_vlm_policy.py` uses it directly and skips free-text JSON parsing entirely,
    since parsing would be reinventing what the backend already gives for free. `dtype` is the
    weight dtype the backend actually loaded in (recorded so a run's numbers are attributable)."""

    raw_text: str
    latency_s: float
    load_time_s: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    backend_name: str = ""
    native_points: Optional[list] = None  # [(x_px, y_px), ...] or None
    dtype: Optional[str] = None
    # Which pixel space the parsed absolute coordinates live in (2026-09-19, see the scorer's
    # docstring): "raw_image" = the frame handed to generate() (the default, and what the prompt
    # asks for); "model_input" = the resized image the model's vision tower actually sees, whose
    # size is `model_input_hw` (h, w). Qwen2.5-VL is "model_input" -- verified empirically, the
    # local 60-frame run scores 40/60 hits under it vs 1/60 as raw pixels.
    # 2026-09-22: "norm1000" / "norm1" (0-1000 / 0-1 normalized over the raw frame) and an optional
    # "_yx" suffix (row-first answers) are also understood by `minimal_vlm_policy.to_raw_px`. The
    # Jetson driver's `--stage calibrate` (`deployment/coord_space_probe.py`) tells you which one a
    # candidate really answers in; set this field (in that backend's `generate()`) accordingly.
    coord_space: str = "raw_image"
    model_input_hw: Optional[tuple] = None


class VLMBackend(Protocol):
    """Structural interface every candidate backend satisfies. `target_label` is a short
    noun phrase (e.g. "green marker", "the ball") -- optional for prompt+generate backends
    (folded into the free-text prompt instead), required for backends whose real API takes
    a label directly (`Moondream2Backend` in "point" mode)."""

    name: str

    def load(self) -> None:
        """Loads weights/processor. Idempotent -- safe to call more than once."""
        ...

    def unload(self) -> None:
        """Drops model references so `free_gpu_memory()` can reclaim them."""
        ...

    def generate(
        self, image: np.ndarray, prompt: str, target_label: Optional[str] = None
    ) -> BackendOutput:
        """`image`: RGB uint8 ndarray, HxWx3 (matches `Policy.act()`'s image convention in
        `policy_interface.py`, NOT `run_jetson_standalone.py`'s raw BGR camera frame --
        `MinimalVLMPolicy.act()` converts before calling this)."""
        ...


class QwenVLBackend:
    """Real, executed backend (local CPU, 2026-09-18). Reuses `deployment/qwen_vl_smoke_test.py`'s
    model-loading path verbatim (import, not copy-paste) per this project's
    `feedback_reuse_existing_export_tooling` convention.

    2026-09-19: `dtype` ("auto" default -- see `resolve_torch_dtype`) and `revision` added for
    the Colab sweep; `model_dir` may be a HF repo id. Pin `revision` to
    66285546d2b821cf421d4f5eb2576359d3770cd3 to get the exact weights the local checkpoint is."""

    name = "qwen2_5_vl_3b_instruct"
    DEFAULT_REPO_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 64,
        min_pixels: int = 64 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        dtype: Optional[str] = "auto",
        revision: Optional[str] = None,
    ) -> None:
        # Smaller default min/max_pixels than qwen_vl_smoke_test.py's CLI default
        # (1280*28*28) -- deliberate, see the 2026-09-18 note in the multi-candidate doc.
        self.model_dir = model_dir
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.dtype = dtype
        self.revision = revision
        self.dtype_name: Optional[str] = None
        self._model = None
        self._processor = None
        self._resolved_device: Optional[str] = None
        self.load_time_s: Optional[float] = None

    def config_summary(self) -> dict:
        return {"model": self.model_dir or "<local default>", "revision": self.revision,
                "dtype": self.dtype_name or self.dtype, "max_new_tokens": self.max_new_tokens,
                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels}

    def load(self) -> None:
        if self._model is not None:
            return
        from ml_jetson_vla.deployment.qwen_vl_smoke_test import (
            DEFAULT_MODEL_DIR,
            load_qwen_vl_model,
        )

        model_dir = self.model_dir or DEFAULT_MODEL_DIR
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype, self.dtype_name = resolve_torch_dtype(self.dtype, device)
        t0 = time.time()
        self._model, self._processor, self._resolved_device = load_qwen_vl_model(
            model_dir, device=device, min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            dtype=torch_dtype, revision=self.revision,
        )
        self.load_time_s = time.time() - t0

    def unload(self) -> None:
        self._model = None
        self._processor = None

    def generate(
        self, image: np.ndarray, prompt: str, target_label: Optional[str] = None
    ) -> BackendOutput:
        self.load()
        from ml_jetson_vla.deployment.qwen_vl_smoke_test import generate_qwen_vl

        result = generate_qwen_vl(
            self._model, self._processor, self._resolved_device, image, prompt,
            max_new_tokens=self.max_new_tokens,
        )
        return BackendOutput(
            raw_text=result["output_text"],
            latency_s=result["latency_s"],
            load_time_s=self.load_time_s,
            peak_memory_gb=result["peak_vram_gb"],
            backend_name=self.name,
            dtype=self.dtype_name,
            coord_space="model_input",
            model_input_hw=result.get("model_input_hw"),
        )


def qwen_model_input_hw(
    raw_h: int, raw_w: int, min_pixels: int = 64 * 28 * 28, max_pixels: int = 256 * 28 * 28,
    message_max_pixels: int = 1280 * 28 * 28,
) -> tuple:
    """Predicts the (h, w) `QwenVLBackend` feeds the vision tower for a raw frame: qwen_vl_utils
    first smart-resizes to `message_max_pixels` (the hardcoded 1280*28*28 in `generate_qwen_vl`),
    then the processor smart-resizes again to `min_pixels`/`max_pixels`. Used ONLY to rescore old
    result files that did not record the size (the live path measures it from the real processor
    output); tests assert the two agree. 640x480 -> (364, 504) with this backend's defaults."""
    from qwen_vl_utils.vision_process import smart_resize

    h1, w1 = smart_resize(raw_h, raw_w, factor=28, min_pixels=4 * 28 * 28, max_pixels=message_max_pixels)
    return smart_resize(h1, w1, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)


# --- InternVL2.5 (remote-code path, the only official path for the 4B checkpoint) ----------
# Preprocessing helpers reproduced from the model card's own "Inference with Transformers"
# section (https://huggingface.co/OpenGVLab/InternVL2_5-4B, fetched 2026-09-19), adapted only to
# accept an in-memory PIL image instead of a file path. The card's own dynamic tiling is what the
# checkpoint was trained with, so it is kept rather than replaced with a plain resize.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def internvl_pixel_values(pil_image, input_size: int = 448, max_num: int = 6):
    """Model-card `load_image()` for an in-memory PIL image -> float32 tensor (N_tiles,3,H,W)."""
    import torch

    transform = _internvl_build_transform(input_size)
    tiles = _internvl_dynamic_preprocess(
        pil_image.convert("RGB"), image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


class InternVLBackend:
    """InternVL2.5-4B via its own `trust_remote_code` + `model.chat()` API (see module
    docstring for why the 2026-09-18 native-class version could not have worked). NOT executed
    against real weights yet. Known risk: the checkpoint's remote code was written against
    transformers ~4.37-4.4x; whether it runs unmodified on the transformers the Colab notebook
    installs is exactly what the smoke stage is there to find out (fault-isolated -- a failure
    here only skips this candidate). `max_tiles` bounds the dynamic-tiling tile count (card's own
    example uses 12; 6 default keeps a 640x480 frame's cost moderate; recorded in the results)."""

    name = "internvl2_5_4b"
    DEFAULT_REPO_ID = "OpenGVLab/InternVL2_5-4B"

    def __init__(
        self,
        model_dir_or_repo: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        revision: Optional[str] = None,
        max_new_tokens: int = 64,
        max_tiles: int = 6,
    ) -> None:
        self.model_dir_or_repo = model_dir_or_repo or self.DEFAULT_REPO_ID
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.max_tiles = max_tiles
        self.dtype_name: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._torch_dtype = None
        self.load_time_s: Optional[float] = None

    def config_summary(self) -> dict:
        return {"model": self.model_dir_or_repo, "revision": self.revision,
                "dtype": self.dtype_name or self.dtype, "max_new_tokens": self.max_new_tokens,
                "max_tiles": self.max_tiles, "trust_remote_code": True}

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_dtype, self.dtype_name = resolve_torch_dtype(self.dtype, device)
        hub = {"revision": self.revision} if self.revision else {}
        t0 = time.time()
        # use_flash_attn=False: the card's example passes True, which needs the flash-attn
        # package (not on a stock Colab, no T4 support); the remote config falls back to eager.
        self._model = AutoModel.from_pretrained(
            self.model_dir_or_repo, torch_dtype=self._torch_dtype, low_cpu_mem_usage=True,
            use_flash_attn=False, trust_remote_code=True, **hub,
        ).eval().to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir_or_repo, trust_remote_code=True, use_fast=False, **hub)
        self._device = device
        self.load_time_s = time.time() - t0

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def generate(
        self, image: np.ndarray, prompt: str, target_label: Optional[str] = None
    ) -> BackendOutput:
        self.load()
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        pixel_values = internvl_pixel_values(pil_image, max_num=self.max_tiles)
        pixel_values = pixel_values.to(self._torch_dtype).to(self._device)
        question = "<image>\n" + prompt
        gen_cfg = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        _reset_peak(self._device)
        t0 = time.time()
        with torch.no_grad():
            response = self._model.chat(self._tokenizer, pixel_values, question, gen_cfg)
        latency_s = time.time() - t0
        return BackendOutput(
            raw_text=str(response), latency_s=latency_s, load_time_s=self.load_time_s,
            peak_memory_gb=_peak_gb(self._device), backend_name=self.name, dtype=self.dtype_name,
        )


class InternVLNativeBackend:
    """OPTIONAL fallback, not the requested candidate: an official HF-native InternVL checkpoint
    (default `OpenGVLab/InternVL3_5-4B-HF` -- a later generation than InternVL2.5, ~4B) loaded
    through transformers' own `AutoModelForImageTextToText`, in case `InternVLBackend`'s remote
    code does not run on the installed transformers. Only meaningful if reported as what it is
    (InternVL3.5-4B), never as "InternVL2.5-4B". NOT executed against real weights."""

    name = "internvl3_5_4b_hf"
    DEFAULT_REPO_ID = "OpenGVLab/InternVL3_5-4B-HF"

    def __init__(
        self,
        model_dir_or_repo: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        revision: Optional[str] = None,
        max_new_tokens: int = 64,
    ) -> None:
        self.model_dir_or_repo = model_dir_or_repo or self.DEFAULT_REPO_ID
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.dtype_name: Optional[str] = None
        self._model = None
        self._processor = None
        self._device = None
        self._torch_dtype = None
        self.load_time_s: Optional[float] = None

    def config_summary(self) -> dict:
        return {"model": self.model_dir_or_repo, "revision": self.revision,
                "dtype": self.dtype_name or self.dtype, "max_new_tokens": self.max_new_tokens}

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_dtype, self.dtype_name = resolve_torch_dtype(self.dtype, device)
        hub = {"revision": self.revision} if self.revision else {}
        t0 = time.time()
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir_or_repo, torch_dtype=self._torch_dtype, device_map=device,
            low_cpu_mem_usage=True, **hub,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_dir_or_repo, **hub)
        self._device = device
        self.load_time_s = time.time() - t0

    def unload(self) -> None:
        self._model = None
        self._processor = None

    def generate(
        self, image: np.ndarray, prompt: str, target_label: Optional[str] = None
    ) -> BackendOutput:
        self.load()
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text=text, images=pil_image, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)
        _reset_peak(self._device)
        t0 = time.time()
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        latency_s = time.time() - t0
        trimmed = out_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self._processor.decode(trimmed, skip_special_tokens=True)
        return BackendOutput(
            raw_text=raw_text, latency_s=latency_s, load_time_s=self.load_time_s,
            peak_memory_gb=_peak_gb(self._device), backend_name=self.name, dtype=self.dtype_name,
        )


class PaliGemma2Backend:
    """PaliGemma2-3B (mix checkpoint, `google/paligemma2-3b-mix-448`, a GATED Hugging Face repo
    -- the Gemma license must be accepted with the account whose token is used). Native
    grounding output is NOT JSON: `<locY1><locX1><locY2><locX2>` tokens on a 1024 grid (Y before
    X), parsed by `minimal_vlm_policy.py`'s dedicated branch.

    Two modes, because the mix checkpoint is documented as task-prefix-prompted (`detect
    <thing>`), not free-chat, and a prompt-variant comparison is only meaningful if the prompt
    text actually reaches the model:
      - `mode="prompt"` (default, the original 2026-09-18 behavior): sends the full prompt text
        (whichever variant is under test) as-is. This is the mode the A/B/C comparison uses.
      - `mode="detect"`: sends the native `detect {target_label}` task prefix and IGNORES the
        prompt text (so all prompt variants are identical by construction -- run it once).
    2026-09-19 fixes: `pixel_values` cast to the model dtype (was left float32); decoded with
    `skip_special_tokens=False` then `<eos>`/`<bos>`/`<pad>` stripped, since whether `<locNNNN>`
    tokens are flagged special is not something this code should depend on. NOT executed."""

    name = "paligemma2_3b_mix"
    DEFAULT_REPO_ID = "google/paligemma2-3b-mix-448"

    def __init__(
        self,
        model_dir_or_repo: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        revision: Optional[str] = None,
        max_new_tokens: int = 64,
        mode: str = "prompt",
    ) -> None:
        if mode not in ("prompt", "detect"):
            raise ValueError("mode must be 'prompt' or 'detect'")
        self.model_dir_or_repo = model_dir_or_repo or self.DEFAULT_REPO_ID
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.mode = mode
        self.dtype_name: Optional[str] = None
        self._model = None
        self._processor = None
        self._device = None
        self._torch_dtype = None
        self.load_time_s: Optional[float] = None

    def config_summary(self) -> dict:
        return {"model": self.model_dir_or_repo, "revision": self.revision, "mode": self.mode,
                "dtype": self.dtype_name or self.dtype, "max_new_tokens": self.max_new_tokens}

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_dtype, self.dtype_name = resolve_torch_dtype(self.dtype, device)
        hub = {"revision": self.revision} if self.revision else {}
        t0 = time.time()
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_dir_or_repo, torch_dtype=self._torch_dtype, device_map=device,
            low_cpu_mem_usage=True, **hub,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_dir_or_repo, **hub)
        self._device = device
        self.load_time_s = time.time() - t0

    def unload(self) -> None:
        self._model = None
        self._processor = None

    def generate(
        self, image: np.ndarray, prompt: str, target_label: Optional[str] = None
    ) -> BackendOutput:
        self.load()
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        text = f"detect {target_label}" if (self.mode == "detect" and target_label) else prompt
        inputs = self._processor(text=text, images=pil_image, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)
        _reset_peak(self._device)
        t0 = time.time()
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        latency_s = time.time() - t0
        trimmed = out_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self._processor.decode(trimmed, skip_special_tokens=False)
        for tok in ("<eos>", "<bos>", "<pad>"):
            raw_text = raw_text.replace(tok, "")
        return BackendOutput(
            raw_text=raw_text.strip(), latency_s=latency_s, load_time_s=self.load_time_s,
            peak_memory_gb=_peak_gb(self._device), backend_name=f"{self.name}:{self.mode}",
            dtype=self.dtype_name,
        )


# Moondream2 remote code is pinned: its README recommends `revision="2025-06-21"` for production
# use (fetched 2026-09-19); that tag resolves to this commit (Hugging Face refs API, same date).
# Pinning the immutable commit rather than the mutable tag name.
MOONDREAM2_PINNED_TAG = "2025-06-21"
MOONDREAM2_PINNED_REVISION = "9a7d4024050840e001defacec2b00727e89149e6"


class Moondream2Backend:
    """1.9B params -- smallest candidate. Ships a native structured pointing API (`.point()`).
    Real API surface confirmed 2026-09-19 from the model's own remote code at the pinned revision
    (`moondream.py`: `point(image: PIL.Image | EncodedImage, object: str)` returns
    `{"points": [{"x": <float>, "y": <float>}, ...]}`, coordinates normalized to [0,1]) and the
    README (`query(image, question)["answer"]`, `revision="2025-06-21"` recommendation,
    `device_map={"": "cuda"}`). NOT executed against the weights.

    Two modes (same reasoning as `PaliGemma2Backend`):
      - `mode="point"` (default): the native `.point(image, target_label)` call; the prompt text is
        not used at all, so every prompt variant is identical by construction -- run it once.
      - `mode="query"`: free-text `.query(image, prompt)["answer"]` with the actual prompt
        variant, parsed by the shared parser -- the only Moondream2 mode where the A/B/C prompt
        comparison means anything.
    Dtype: the README's documented load call passes NO dtype (=> fp32 in transformers 4.x), so
    the default here is `dtype="default"` (do not pass one) rather than a guess at fp16/bf16
    behaviour of the remote code. Requires transformers<5 (see module docstring)."""

    name = "moondream2"
    DEFAULT_REPO_ID = "vikhyatk/moondream2"

    def __init__(
        self,
        model_dir_or_repo: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "default",
        revision: Optional[str] = MOONDREAM2_PINNED_REVISION,
        mode: str = "point",
    ) -> None:
        if mode not in ("point", "query"):
            raise ValueError("mode must be 'point' or 'query'")
        self.model_dir_or_repo = model_dir_or_repo or self.DEFAULT_REPO_ID
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.mode = mode
        self.dtype_name: Optional[str] = None
        self._model = None
        self._device = None
        self.load_time_s: Optional[float] = None

    def config_summary(self) -> dict:
        return {"model": self.model_dir_or_repo, "revision": self.revision, "mode": self.mode,
                "dtype": self.dtype_name or self.dtype, "trust_remote_code": True,
                "pinned_tag": MOONDREAM2_PINNED_TAG}

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype, self.dtype_name = resolve_torch_dtype(self.dtype, device)
        kwargs = {"trust_remote_code": True, "device_map": {"": device}}
        if self.revision and not os.path.exists(self.model_dir_or_repo):
            kwargs["revision"] = self.revision
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        t0 = time.time()
        self._model = AutoModelForCausalLM.from_pretrained(self.model_dir_or_repo, **kwargs)
        self._device = device
        self.load_time_s = time.time() - t0

    def unload(self) -> None:
        self._model = None

    def generate(
        self, image: np.ndarray, prompt: str, target_label: Optional[str] = None
    ) -> BackendOutput:
        """`mode="point"` with a `target_label`: real `.point()` API, bypassing free-text
        prompting. Otherwise (query mode, or no label -- the directional/hold fallback path):
        real free-text `.query()`."""
        self.load()
        from PIL import Image

        pil_image = Image.fromarray(image)
        _reset_peak(self._device)
        t0 = time.time()
        if self.mode == "point" and target_label:
            result = self._model.point(pil_image, target_label)
            latency_s = time.time() - t0
            points = result.get("points", [])
            # `.point()` returns points normalized to [0,1] (top-left origin) -- denormalize
            # so native_points is in the same raw-pixel space as every other backend.
            img_h, img_w = image.shape[0], image.shape[1]
            native_points = (
                [(p["x"] * img_w, p["y"] * img_h) for p in points] if points else None
            )
            return BackendOutput(
                raw_text=repr(result), latency_s=latency_s, load_time_s=self.load_time_s,
                peak_memory_gb=_peak_gb(self._device), backend_name=f"{self.name}:point",
                native_points=native_points, dtype=self.dtype_name,
            )
        result = self._model.query(pil_image, prompt)
        latency_s = time.time() - t0
        return BackendOutput(
            raw_text=result.get("answer", ""), latency_s=latency_s, load_time_s=self.load_time_s,
            peak_memory_gb=_peak_gb(self._device), backend_name=f"{self.name}:query",
            dtype=self.dtype_name,
        )


# Registry -- lets a config/CLI flag pick a backend by name rather than importing a class
# directly, matching this task's "swapping the underlying model is a config/constructor
# change" requirement.
BACKEND_REGISTRY = {
    QwenVLBackend.name: QwenVLBackend,
    InternVLBackend.name: InternVLBackend,
    InternVLNativeBackend.name: InternVLNativeBackend,
    PaliGemma2Backend.name: PaliGemma2Backend,
    Moondream2Backend.name: Moondream2Backend,
}
