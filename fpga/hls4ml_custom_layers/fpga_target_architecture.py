"""FPGA-target variant of SharedVisionBackbone: AdaptiveAvgPool2d((1,1)) swapped
for AvgPool2d(16).

hls4ml's PyTorch converter has no AdaptiveAvgPool2d handler (verified against
the installed hls4ml==1.3.0 package's get_supported_pytorch_layers() list,
2026-08-12 -- see docs/plans/ml_system_parameter_budget.md Section 5.8). Given
the encoder's fixed 16x16 spatial output (locked 128x128 input, Section 5.4),
AdaptiveAvgPool2d((1,1)) is mathematically identical to plain AvgPool2d(16),
which IS supported. This is an exact substitution, not an approximation --
verified below by direct numerical comparison against the real production
SharedVisionBackbone, not asserted from the math alone.

Production's mask_head/heatmap_head now use Upsample(nearest)+Conv2d instead
of ConvTranspose2d -- DECIDED 2026-08-13 after a 30-run reweighting sweep
found it dominates on mask quality with matched ball-tracking stability once
mask/heatmap loss weights are reduced to ~0.1/0.02 (docs/plans/
ml_system_parameter_budget.md Section 5.8/5.9). Shipped into production as of
the shared_vision_backbone_v2 checkpoint (2026-08-14) -- this file previously
carried a stand-in reference architecture while that was pending; it's been
removed now that the real thing exists, per the TODO that used to be here.

This file lives under fpga/, not host_software/ml_vision/, on purpose: per
agent_fpga.md's boundary rules, this agent has read-only access to
host_software/ml_vision/ and writes stay inside fpga/. It never modifies the
production model or its checkpoint -- it reads production weights via
load_state_dict() (AdaptiveAvgPool2d/AvgPool2d are both parameter-free, so
there's nothing pooling-specific to remap) and produces an FPGA-target
variant as a separate, fpga/-owned artifact.

Run the equivalence check:

    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe -m fpga.hls4ml_custom_layers.fpga_target_architecture
    # or, against the real trained checkpoint instead of random init:
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe -m fpga.hls4ml_custom_layers.fpga_target_architecture \
        --checkpoint host_software/ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone  # noqa: E402

INPUT_SIZE = (128, 128)


def _decoder_head() -> nn.Sequential:
    """Upsample(nearest, scale=2) + Conv2d(3x3, pad=1) + BatchNorm2d + GELU,
    repeated 3x (64->32->16->8 channels), then the final Conv2d(8,1,1) --
    byte-for-byte matches production's mask_head/heatmap_head structure
    (train_cnn_2d_tracker_marker.py). Shared between mask_head and
    heatmap_head here so there's exactly one place this can drift out of
    sync with production."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.GELU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(16, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.GELU(),
        nn.Conv2d(8, 1, 1),
    )


class FPGATargetSharedVisionBackbone(nn.Module):
    """Production's architecture with ball_head's AdaptiveAvgPool2d((1,1))
    swapped for AvgPool2d(16) -- see module docstring. That's the ONLY delta
    from production; encoder, ball_head's conv/BN/GELU, mask_head, and
    heatmap_head are otherwise identical."""

    def __init__(self, input_size: Tuple[int, int] = INPUT_SIZE) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.ball_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(16),  # was AdaptiveAvgPool2d((1, 1)) -- exact given the fixed 16x16 encoder output
            nn.Flatten(),
            nn.Linear(32, 2),
        )

        self.mask_head = _decoder_head()
        self.heatmap_head = _decoder_head()

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        ball_xy = self.ball_head(features)

        mask_logits = self.mask_head(features)
        heatmap_logits = self.heatmap_head(features)
        mask_logits = F.interpolate(mask_logits, size=self.input_size, mode="bilinear", align_corners=False)
        heatmap_logits = F.interpolate(heatmap_logits, size=self.input_size, mode="bilinear", align_corners=False)
        return ball_xy, mask_logits[:, :1], heatmap_logits[:, :1]

    @classmethod
    def from_pretrained(cls, source: SharedVisionBackbone) -> "FPGATargetSharedVisionBackbone":
        """Copy every weight from a trained/untrained SharedVisionBackbone.
        AdaptiveAvgPool2d and AvgPool2d are both parameter-free (no weights,
        no buffers), so they contribute zero state_dict keys either way --
        the two models' state_dicts are keyed identically. strict=True (the
        default) means this raises loudly if that assumption is ever wrong,
        rather than silently masking a real architecture mismatch."""
        model = cls(input_size=source.input_size)
        model.load_state_dict(source.state_dict())
        model.eval()
        return model


def _check_equivalence(checkpoint: Optional[Path] = None) -> None:
    torch.manual_seed(0)
    source = SharedVisionBackbone(input_size=INPUT_SIZE)
    if checkpoint is not None:
        source.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded real checkpoint: {checkpoint}")
    else:
        print("No --checkpoint given -- using random init (structural check only).")
    source.eval()

    target = FPGATargetSharedVisionBackbone.from_pretrained(source)

    probe = torch.rand(3, 3, *INPUT_SIZE)
    with torch.no_grad():
        src_ball, src_mask, src_heatmap = source(probe)
        tgt_ball, tgt_mask, tgt_heatmap = target(probe)

    ball_diff = (src_ball - tgt_ball).abs().max().item()
    mask_diff = (src_mask - tgt_mask).abs().max().item()
    heatmap_diff = (src_heatmap - tgt_heatmap).abs().max().item()
    print(
        f"AvgPool2d(16) vs. AdaptiveAvgPool2d((1,1)) equivalence check -- "
        f"max abs diff: ball={ball_diff:.2e} mask={mask_diff:.2e} heatmap={heatmap_diff:.2e}"
    )
    tolerance = 1e-5
    if max(ball_diff, mask_diff, heatmap_diff) > tolerance:
        raise AssertionError(
            f"Pooling substitution equivalence check FAILED (tolerance={tolerance}) -- "
            "AvgPool2d(16) is NOT producing the same result as AdaptiveAvgPool2d((1,1)) here. "
            "Do not use this FPGA-target variant until this is understood."
        )
    print("PASSED -- AvgPool2d(16) is an exact substitution, confirmed numerically, not just by the math.")

    n_params = sum(p.numel() for p in target.parameters())
    print(f"\nFPGATargetSharedVisionBackbone total params: {n_params:,}")
    expected = 91_140
    if n_params != expected:
        raise AssertionError(
            f"Param count {n_params:,} != {expected:,} expected -- architecture has drifted from production. "
            "Re-sync this file with train_cnn_2d_tracker_marker.py's SharedVisionBackbone."
        )
    print(f"MATCHES production's param count ({expected:,}) exactly.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="Real trained checkpoint (optional)")
    args = parser.parse_args()
    _check_equivalence(args.checkpoint)
