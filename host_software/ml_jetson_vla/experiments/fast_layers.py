"""Fast-layer options behind one interface: FastLayer.plan(target_mm, state_mm, frame) -> ChunkPlan.

ACT_FAST (name "act_fast"): ACT with the target as two extra state elements, observation.state is
4-dim [touch_x, touch_y, target_x, target_y]. RANDOM weights unless a trained checkpoint directory
is given. Action output is not un-normalised here (no dataset stats are applied): see the plan doc.

SMOLVLA_DIRECT (name "smolvla_direct"): pretrained lerobot/smolvla_base at a pinned commit. Receives
the instruction text plus frame and state, does its own grounding. Its 6-value output is reduced to
the first 3 values, a placeholder; the real reduction is an open decision. Not fine-tuned.

Torch and lerobot are imported inside the functions that need them, so this module imports on CPU
without either. Model construction reuses the loaders in deployment/bench_action_models.py.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol

import numpy as np

from ml_jetson_vla.deployment.bench_action_models import (
    ACT_IMAGE_HW,
    IMAGE_KEY,
    LANG_MASK_KEY,
    LANG_TOKENS_KEY,
    SMOLVLA_REPO_ID,
    STATE_KEY,
    build_act_policy,
)
from ml_jetson_vla.experiments.schedule import ChunkPlan
from ml_jetson_vla.experiments.serial_protocol import N_MOTORS, validate_chunk_angles

SMOLVLA_PINNED_COMMIT: str = "d9f33c94a60fb382c90dea2164c96845bd955e28"
ACT_DEFAULT_CHUNK_LEN: int = 10  # 0.33 s at 30 Hz; a control choice (ACT's cost does not depend on it)
ACT_WEIGHTS_RANDOM: str = "RANDOM_INIT: no trained ACT checkpoint; actions are not meaningful"
SMOLVLA_WEIGHTS: str = (
    f"PRETRAINED lerobot/smolvla_base@{SMOLVLA_PINNED_COMMIT}, NOT fine-tuned; "
    "output 6->3 reduction is a placeholder"
)


class FastLayer(Protocol):
    name: str
    weights_status: str

    def plan(
        self,
        target_mm: tuple[float, float],
        state_mm: tuple[float, float],
        frame: np.ndarray,
        instruction: Optional[str] = None,
    ) -> ChunkPlan:
        """`frame`: BGR, as logged or as captured. `instruction` is used by SMOLVLA_DIRECT only."""
        ...


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def bgr_frame_to_chw_tensor(
    frame: np.ndarray, hw: Optional[tuple[int, int]], device: str, dtype: Any
) -> Any:
    import torch

    img = frame
    if hw is not None and tuple(img.shape[:2]) != tuple(hw):
        import cv2

        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(img[..., ::-1]).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _make_plan(option: str, weights: str, actions: np.ndarray, inference_ms: float) -> ChunkPlan:
    # Safety is applied here, before the plan leaves this module. Nothing is clamped.
    return ChunkPlan(
        option=option,
        weights_status=weights,
        actions_deg=actions,
        safety=validate_chunk_angles(actions),
        inference_ms=inference_ms,
    )


class ActFastLayer:
    name: str = "act_fast"

    def __init__(
        self, chunk_len: int = ACT_DEFAULT_CHUNK_LEN, device: str = "cpu", checkpoint_dir: Optional[str] = None
    ) -> None:
        import torch

        if chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        self._chunk_len: int = chunk_len
        self._device: str = device
        self._dtype: Any = torch.float32  # bf16 failed in the Orin benchmark; fp32 until autocast is verified
        if checkpoint_dir is None:
            policy, _cfg = build_act_policy(chunk_len, with_target=True)
            self.weights_status: str = ACT_WEIGHTS_RANDOM
        else:
            from lerobot.policies.act.modeling_act import ACTPolicy

            policy = ACTPolicy.from_pretrained(checkpoint_dir)
            self.weights_status = f"TRAINED checkpoint {checkpoint_dir}; normaliser stats not verified"
        self._policy: Any = policy.to(device=device, dtype=self._dtype).eval()

    def plan(
        self,
        target_mm: tuple[float, float],
        state_mm: tuple[float, float],
        frame: np.ndarray,
        instruction: Optional[str] = None,
    ) -> ChunkPlan:
        import torch

        image = bgr_frame_to_chw_tensor(frame, ACT_IMAGE_HW, self._device, self._dtype)
        state = torch.tensor(
            [[float(state_mm[0]), float(state_mm[1]), float(target_mm[0]), float(target_mm[1])]],
            device=self._device,
            dtype=self._dtype,
        )
        batch = {IMAGE_KEY: image, STATE_KEY: state}
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = self._policy.predict_action_chunk(batch)
            _sync(self._device)
            inference_ms = (time.perf_counter() - t0) * 1000.0
        actions = _chunk_to_array(out, self._chunk_len)
        return _make_plan(self.name, self.weights_status, actions, inference_ms)


class SmolVLAFastLayer:
    name: str = "smolvla_direct"

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        # Pinned revision. bench_action_models.load_smolvla() resolves the hub's current HEAD instead,
        # so it is not reused for weights here.
        cfg = PreTrainedConfig.from_pretrained(SMOLVLA_REPO_ID, revision=SMOLVLA_PINNED_COMMIT)
        policy = SmolVLAPolicy.from_pretrained(SMOLVLA_REPO_ID, config=cfg, revision=SMOLVLA_PINNED_COMMIT)
        self._device: str = device
        self._dtype: Any = torch.float32
        self._policy: Any = policy.to(device=device, dtype=self._dtype).eval()
        self._cfg: Any = cfg
        self._chunk_len: int = int(cfg.chunk_size)
        self._state_dim: int = int(cfg.robot_state_feature.shape[0]) if cfg.robot_state_feature is not None else 0
        self._tokenizer: Any = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
        self.weights_status: str = (
            SMOLVLA_WEIGHTS + f"; state zero-padded from 2 to {self._state_dim} dims"
        )

    def plan(
        self,
        target_mm: tuple[float, float],
        state_mm: tuple[float, float],
        frame: np.ndarray,
        instruction: Optional[str] = None,
    ) -> ChunkPlan:
        import torch

        if instruction is None:
            raise ValueError("SMOLVLA_DIRECT needs the instruction text")
        batch: dict[str, Any] = {}
        for key, feat in self._cfg.image_features.items():
            # Native frame size: SmolVLA resizes with padding internally. The same frame is used for every camera key.
            batch[key] = bgr_frame_to_chw_tensor(frame, None, self._device, self._dtype)
        if self._state_dim > 0:
            state = np.zeros((1, self._state_dim), dtype=np.float32)
            state[0, 0] = float(state_mm[0])
            state[0, 1] = float(state_mm[1])
            batch[STATE_KEY] = torch.from_numpy(state).to(device=self._device, dtype=self._dtype)
        tok = self._tokenizer(
            [instruction],
            padding=self._cfg.pad_language_to,
            max_length=self._cfg.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch[LANG_TOKENS_KEY] = tok["input_ids"].to(self._device)
        batch[LANG_MASK_KEY] = tok["attention_mask"].to(device=self._device, dtype=torch.bool)
        noise = torch.normal(
            0.0,
            1.0,
            size=(1, self._chunk_len, int(self._cfg.max_action_dim)),
            device=self._device,
            dtype=self._dtype,
        )
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = self._policy.predict_action_chunk(batch, noise=noise)
            _sync(self._device)
            inference_ms = (time.perf_counter() - t0) * 1000.0
        if out.shape[-1] < N_MOTORS:
            raise RuntimeError(f"SmolVLA output has {out.shape[-1]} dims, need at least {N_MOTORS}")
        actions = _chunk_to_array(out[..., :N_MOTORS], self._chunk_len)  # placeholder reduction
        return _make_plan(self.name, self.weights_status, actions, inference_ms)


class StubFastLayer:
    """No model. Returns a constant chunk. For exercising the schedule and safety path only. Its
    inference_ms times a numpy fill, which says nothing about any model's latency."""

    name: str = "stub"

    def __init__(self, value_deg: float = 0.0, chunk_len: int = ACT_DEFAULT_CHUNK_LEN) -> None:
        self._value: float = float(value_deg)
        self._chunk_len: int = chunk_len
        self.weights_status: str = f"STUB (no model), constant {self._value} deg"

    def plan(
        self,
        target_mm: tuple[float, float],
        state_mm: tuple[float, float],
        frame: np.ndarray,
        instruction: Optional[str] = None,
    ) -> ChunkPlan:
        t0 = time.perf_counter()
        actions = np.full((self._chunk_len, N_MOTORS), self._value, dtype=np.float64)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return _make_plan(self.name, self.weights_status, actions, inference_ms)


def _chunk_to_array(out: Any, expected_steps: int) -> np.ndarray:
    arr = out.detach().float().cpu().numpy()
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] != expected_steps or arr.shape[2] != N_MOTORS:
        raise RuntimeError(f"unexpected chunk shape {arr.shape}, expected (1, {expected_steps}, {N_MOTORS})")
    return arr[0].astype(np.float64)
