"""FPGA-target variant of SharedVisionBackbone: AdaptiveAvgPool2d((1,1)) swapped
for AvgPool2d(16), and ConvTranspose2d decoder heads swapped for Upsample+Conv2d.

hls4ml's PyTorch converter has no AdaptiveAvgPool2d or ConvTranspose2d handler
(verified against the installed hls4ml==1.3.0 package's
get_supported_pytorch_layers() list, 2026-08-12 -- see docs/plans/
ml_system_parameter_budget.md Section 5.8). Two fixes:

1. AdaptiveAvgPool2d((1,1)) -> AvgPool2d(16): given the encoder's fixed 16x16
   spatial output (locked 128x128 input, Section 5.4), these are mathematically
   identical -- an exact substitution, not an approximation. Verified below by
   direct numerical comparison, not asserted from the math alone.

2. ConvTranspose2d -> Upsample(nearest, scale=2) + Conv2d(3x3): DECIDED
   2026-08-13 after a 30-run reweighting sweep (Section 5.8/5.9) found
   Upsample+Conv2d dominates on mask quality with matched ball-tracking
   stability, once mask/heatmap loss weights are reduced to ~0.1/0.02. This is
   NOT a parameter-free swap like the pooling fix -- Upsample+Conv2d costs
   91,140 total params vs. 64,260 for ConvTranspose2d (verified by hand below,
   matching the trial's reported total exactly, not just trusted).

*** IMPORTANT DEPENDENCY: as of 2026-08-13, ml_vision has NOT yet shipped the
Upsample+Conv2d decision into production `SharedVisionBackbone`
(host_software/ml_vision/training/train_cnn_2d_tracker_marker.py) -- that file
still defines the OLD ConvTranspose2d architecture. Until it ships, there is no
real checkpoint this file's `from_pretrained` can actually load from wearing
its full Upsample+Conv2d shape. `_UpsampleConvReferenceArchitecture` below is a
STAND-IN matching the *decided* architecture (same pattern
trial_activation_functions.py already uses for TrialSharedVisionBackbone when
production hasn't caught up to a trial's finding yet) -- it exists ONLY so this
file's logic can be verified now instead of shipped untested. ONCE ml_vision
ships this into production:
  1. Delete `_UpsampleConvReferenceArchitecture` below.
  2. Change the import back to
     `from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone`
  3. Point `_check_equivalence()` and `from_pretrained()`'s type hint at that
     real import instead of the stand-in.
No other changes should be needed -- at that point this file's only remaining
delta from production is the free pooling swap (item 1 above), the same
situation as before the decoder decision.

This file lives under fpga/, not host_software/ml_vision/, on purpose: per
agent_fpga.md's boundary rules, this agent has read-only access to
host_software/ml_vision/ and writes stay inside fpga/. It never modifies the
production model or its checkpoint.

Run the equivalence check (no dataset or trained checkpoint required --
untrained weights are sufficient to prove the pooling swap is exact; the
decoder swap is NOT parameter-free, so this only proves the two architectures'
*shapes* and *pooling* line up, not that Upsample+Conv2d matches
ConvTranspose2d numerically -- they don't, by construction, that's the whole
point of the trial):

    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe -m fpga.hls4ml_custom_layers.fpga_target_architecture
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_SIZE = (128, 128)


def _decoder_head() -> nn.Sequential:
    """Upsample(nearest, scale=2) + Conv2d(3x3, pad=1) + BatchNorm2d + GELU,
    repeated 3x (64->32->16->8 channels, matching production's channel
    progression), then the same final Conv2d(8,1,1) as before. Shared between
    mask_head and heatmap_head, and between FPGATargetSharedVisionBackbone and
    the reference stand-in below, so there's exactly one place this can drift
    out of sync between the two."""
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


class _UpsampleConvReferenceArchitecture(nn.Module):
    """STAND-IN for what production SharedVisionBackbone will look like once
    ml_vision ships the Upsample+Conv2d decision -- see module docstring for
    why this exists and when to delete it. Identical to today's production
    class except mask_head/heatmap_head use _decoder_head() instead of
    ConvTranspose2d, and ball_head keeps AdaptiveAvgPool2d((1,1)) (this
    represents production AS ml_vision will ship it -- the AvgPool2d swap is
    FPGATargetSharedVisionBackbone's job below, not production's)."""

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
            nn.AdaptiveAvgPool2d((1, 1)),
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


# TODO once ml_vision ships Upsample+Conv2d into production: delete
# _UpsampleConvReferenceArchitecture above and use this import instead:
# from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone
SharedVisionBackbone = _UpsampleConvReferenceArchitecture


class FPGATargetSharedVisionBackbone(nn.Module):
    """The actual FPGA deployment target: production's architecture (as it
    will exist once ml_vision ships Upsample+Conv2d) with ball_head's
    AdaptiveAvgPool2d((1,1)) swapped for AvgPool2d(16) -- see module
    docstring, item 1. That's now the ONLY delta from production."""

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
    def from_pretrained(cls, source: "SharedVisionBackbone") -> "FPGATargetSharedVisionBackbone":
        """Copy every weight from a trained/untrained SharedVisionBackbone.
        AdaptiveAvgPool2d and AvgPool2d are both parameter-free (no weights,
        no buffers), so they contribute zero state_dict keys either way --
        the two models' state_dicts are keyed identically. strict=True (the
        default) means this raises loudly if that assumption is ever wrong,
        rather than silently masking a real architecture mismatch -- in
        particular, this will raise loudly (not silently misbehave) if
        pointed at a checkpoint still using the OLD ConvTranspose2d
        architecture, since those state_dict keys/shapes genuinely differ."""
        model = cls(input_size=source.input_size)
        model.load_state_dict(source.state_dict())
        model.eval()
        return model


def _check_equivalence() -> None:
    """Verifies the pooling swap only -- see module docstring's note on what
    this test can and can't prove given the decoder change isn't parameter-free."""
    torch.manual_seed(0)
    source = SharedVisionBackbone(input_size=INPUT_SIZE)
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
            f"Param count {n_params:,} != {expected:,} expected from the reweighting trial's reported total. "
            "The reconstructed Upsample+Conv2d architecture doesn't match what the trial actually used -- "
            "do not trust this file until the discrepancy is understood."
        )
    print(f"MATCHES the reweighting trial's reported total ({expected:,}) exactly -- architecture reconstruction confirmed correct.")


if __name__ == '__main__':
    _check_equivalence()
