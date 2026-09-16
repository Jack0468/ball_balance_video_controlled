"""Architecture skeleton for the multi-head Qwen2.5-VL-3B design in
`docs/MULTI_HEAD_ARCHITECTURE_SPEC.md` (read that doc first -- this module exists to make
its layer shapes/attachment points checkable against real code, not as a training-ready
implementation).

NOT TRAINING CODE. No loss function, no optimizer, no data loader, no PPO/BC loop -- those
are items 4/5 on the Track-4 TODO list, not this one. `forward()` below wires shapes only;
it will run structurally (given a real Qwen2_5_VLForConditionalGeneration instance and
correctly-shaped dummy tensors) but nothing here has been trained or hardware-validated,
matching this module's sibling `control_net.py`'s own "NOT YET HARDWARE-VALIDATED" framing.

Three heads, per the spec doc:
  Head A -- Qwen's own native VLM path (vision tower + 36-layer decoder + lm_head). Not
            reimplemented here -- this module wraps an existing
            `Qwen2_5_VLForConditionalGeneration` instance rather than rebuilding it.
  Head B -- state fusion, Phase 1: Linear(2, 2048) + LayerNorm. `state_dim=2` matches
            convert_to_lerobot.py's ACTUAL committed `observation.state` schema
            ([touch_x_mm, touch_y_mm]) today -- NOT the plan's originally-recommended 9-dim
            RLControl.cpp contract, which convert_to_lerobot.py does not currently emit (see
            spec doc S2.1). `state_dim` is a constructor arg specifically so Phase 2 (9-dim)
            is a config change, not a rewrite, once/if that extension is built.
  Head C -- action-chunk regression head. Reuses RT1LiteVLA.action_head's exact
            Linear(512,128)->ReLU->Linear(128,action_dim) pattern
            (ml_multimodal/core/vla_architecture.py) with a new Linear(4096,512) fusion
            down-projection in front of it, and action_dim generalized to N*3 for chunking.
            Output is bounded via tanh * ANGLE_LIMIT_DEG (spec doc S3.3).

Backbone hidden_size (2048) and vision out_hidden_size (2048) are read from this repo's own
`models/qwen2_5_vl_3b_instruct/config.json`, not hardcoded from memory of Qwen's public
spec -- if a different checkpoint is ever swapped in, `qwen_hidden_size` must be re-checked
against its config.json, not assumed.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn


# From models/qwen2_5_vl_3b_instruct/config.json -- see spec doc S1 for the full table and
# the derived ~3.74B total-parameter computation this was cross-checked against.
QWEN_HIDDEN_SIZE = 2048

# RLControl.cpp's MAX_MOTOR_STEP=98 steps, converted via motor_geometry.py's confirmed
# 0.1125 deg/step linear map (98 * 0.1125 = 11.025). Arm 1's proven OPERATING envelope, not
# necessarily the platform's mechanical limit -- see spec doc S3.3 before treating this as a
# hard safety bound rather than a cited starting point.
DEFAULT_ANGLE_LIMIT_DEG = 11.025


@dataclasses.dataclass
class MultiHeadConfig:
    """Constructor knobs called out explicitly in the spec doc as open/unconfirmed rather
    than fixed, so changing any of them is a config edit, not an architecture rewrite."""

    state_dim: int = 2  # Phase 1 (matches convert_to_lerobot.py today). Phase 2 = 9.
    chunk_size: int = 10  # N in [N,3] output. Spec doc S5.4: starting point, NOT tuned.
    angle_limit_deg: float = DEFAULT_ANGLE_LIMIT_DEG
    qwen_hidden_size: int = QWEN_HIDDEN_SIZE
    fusion_hidden_dim: int = 512  # matches RT1LiteVLA.action_head's first Linear's output
    action_head_hidden_dim: int = 128  # matches RT1LiteVLA.action_head's second Linear's input


class StateFusionHead(nn.Module):
    """Head B. Mirrors RT1LiteVLA's `state_embed: nn.Linear(state_dim, 512)` pattern
    (vla_architecture.py), but projects into Qwen's 2048-dim hidden space instead of
    RT1LiteVLA's own from-scratch 512-dim Transformer -- so a LayerNorm is added here (not
    present in RT1LiteVLA's version) since this embedding is fused directly against a
    pretrained, layer-normed backbone's own hidden states. See spec doc S2.2."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(state_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: [B, state_dim] -> [B, hidden_size]
        return self.norm(self.proj(state))


class ActionChunkHead(nn.Module):
    """Head C. First Linear is new (down-projects the Qwen-scale fused representation);
    the two layers after it are RT1LiteVLA.action_head's exact shape, generalized from
    action_dim=3 to chunk_size*3. See spec doc S3.2 for the param-count table."""

    def __init__(self, config: MultiHeadConfig) -> None:
        super().__init__()
        fused_dim = config.qwen_hidden_size * 2  # pooled Qwen hidden + Head B state embed
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.fusion_hidden_dim),
            nn.ReLU(),
        )
        # Verbatim shape reuse of RT1LiteVLA.action_head (vla_architecture.py lines 62-65),
        # with the final Linear's output width generalized to chunk_size * 3.
        self.action_head = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.action_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.action_head_hidden_dim, config.chunk_size * 3),
        )
        self.chunk_size = config.chunk_size
        self.angle_limit_deg = config.angle_limit_deg

    def forward(self, fused_input: torch.Tensor) -> torch.Tensor:
        # fused_input: [B, 2 * qwen_hidden_size]
        x = self.fusion(fused_input)  # [B, fusion_hidden_dim]
        raw = self.action_head(x)  # [B, chunk_size * 3]
        raw = raw.view(-1, self.chunk_size, 3)  # [B, N, 3]
        # Bounding activation -- resolves the plan's flagged gap that RT1LiteVLA.action_head
        # has no bounding at all. See spec doc S3.3 for where angle_limit_deg comes from.
        return torch.tanh(raw) * self.angle_limit_deg  # [B, N, 3], degrees


class QwenMultiHeadPolicy(nn.Module):
    """Wires Heads A/B/C together per spec doc S3.1/S4. Wraps an existing, externally-loaded
    `Qwen2_5_VLForConditionalGeneration` instance (Head A) rather than reimplementing it --
    that model is loaded exactly as qwen_vl_smoke_test.py already does (real, working code),
    not rebuilt here.

    NOTE ON THE ACTION-QUERY TOKEN: this skeleton represents the learned query embedding and
    documents where its final hidden state would be extracted from Qwen's output, but does
    NOT implement the actual sequence-splicing (appending the query embedding into Qwen's
    `inputs_embeds` before the forward pass, then indexing the right output position back
    out) -- that requires wiring against the real `Qwen2_5_VLForConditionalGeneration`
    forward signature (`inputs_embeds` vs `input_ids`, `image_grid_thw`, etc.), which is
    implementation work belonging to items 4/5, not this design task. `forward()` below
    takes `pooled_query_hidden` as a precomputed argument for exactly that reason -- it
    documents the shape contract at this boundary without asserting the splicing is done.
    """

    def __init__(self, qwen_backbone: "nn.Module", config: Optional[MultiHeadConfig] = None) -> None:
        super().__init__()
        self.config = config or MultiHeadConfig()
        self.qwen_backbone = qwen_backbone  # Head A -- native, unmodified Qwen2.5-VL-3B

        # Head B
        self.state_head = StateFusionHead(self.config.state_dim, self.config.qwen_hidden_size)

        # Learned action-query embedding, appended to Qwen's input sequence upstream of this
        # module (see class docstring) -- shape [qwen_hidden_size], not [1, qwen_hidden_size]
        # or [B, qwen_hidden_size]; broadcast/expand to batch happens at the splicing site,
        # not here.
        self.action_query_embedding = nn.Parameter(torch.zeros(self.config.qwen_hidden_size))

        # Head C
        self.action_head = ActionChunkHead(self.config)

    def forward(
        self,
        pooled_query_hidden: torch.Tensor,  # [B, qwen_hidden_size] -- Qwen's final hidden
        # state at the action-query token's sequence position, AFTER a real forward pass
        # through self.qwen_backbone with the query embedding spliced in (see class
        # docstring -- that splicing is not implemented in this skeleton).
        state: torch.Tensor,  # [B, state_dim]
    ) -> torch.Tensor:
        state_embed = self.state_head(state)  # [B, qwen_hidden_size]
        fused = torch.cat([pooled_query_hidden, state_embed], dim=-1)  # [B, 2*qwen_hidden_size]
        return self.action_head(fused)  # [B, chunk_size, 3], degrees, bounded

    def num_new_trainable_params(self) -> int:
        """Sums only Heads B/C + the query embedding -- excludes self.qwen_backbone, whether
        or not the backbone is frozen/LoRA-adapted, since that's a separate accounting
        question (spec doc S1/S5.7) from "how big are the genuinely new heads" (spec doc
        S3.2's ~2.18M figure, computed by hand there at chunk_size=10; this method lets that
        number be checked against the real module rather than trusted from the doc alone)."""
        heads = [self.state_head, self.action_head]
        total = sum(p.numel() for m in heads for p in m.parameters())
        total += self.action_query_embedding.numel()
        return total


def self_test() -> None:
    """Structural-only check (shapes/param count), mirroring control_net.py's own
    self_test() convention -- does NOT load a real Qwen backbone or validate against
    hardware/training. Run directly: `python qwen_multihead_policy.py`."""
    config = MultiHeadConfig()
    policy = QwenMultiHeadPolicy(qwen_backbone=nn.Identity(), config=config)

    batch_size = 4
    pooled_query_hidden = torch.zeros(batch_size, config.qwen_hidden_size)
    state = torch.zeros(batch_size, config.state_dim)

    out = policy(pooled_query_hidden, state)
    assert out.shape == (batch_size, config.chunk_size, 3), out.shape
    assert torch.all(out.abs() <= config.angle_limit_deg + 1e-5)

    n_new = policy.num_new_trainable_params()
    print(f"[qwen_multihead_policy self_test] output shape OK: {tuple(out.shape)}")
    print(f"[qwen_multihead_policy self_test] new trainable params (Heads B+C, excl. backbone): {n_new:,}")


if __name__ == "__main__":
    self_test()
