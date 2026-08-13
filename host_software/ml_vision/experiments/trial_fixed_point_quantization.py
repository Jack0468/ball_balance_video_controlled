"""Simulate ap_fixed<W, I> fixed-point quantization (docs/HLS_DATA_TYPES.md) on a
trained Shared Vision Backbone checkpoint and measure accuracy degradation vs. the
float32 baseline, motivated by the FPGA port (docs/plans/ml_system_parameter_budget.md
Section 5) -- before committing hardware bit-widths, we want to know how much accuracy
a given (weight_bits, weight_int_bits, act_bits, act_int_bits) choice actually costs.

Unlike trial_activation_functions.py / trial_augmentation_strategies.py, this does NOT
retrain a variant per configuration -- quantization is applied post-hoc (post-training
quantization, PTQ) to a single trained checkpoint, since the question is "how much does
quantizing THIS trained model hurt," not "does a differently-trained model do better."

BatchNorm is folded into the preceding conv/deconv before quantization (Section 5.5 of
the budget doc) since that's what the deployed HLS core will actually run -- quantizing
unfused BN parameters separately isn't representative of the real hardware path.

*** IMPORTANT DEPENDENCY, updated 2026-08-13: the decoder heads (mask_head,
heatmap_head) now use Upsample(nearest, scale=2) + Conv2d(3x3), not
ConvTranspose2d -- DECIDED after a 30-run reweighting sweep found it dominates
on mask quality with matched ball-tracking stability once mask/heatmap loss
weights are reduced to ~0.1/0.02 (docs/plans/ml_system_parameter_budget.md
Section 5.8/5.9). As of this update, ml_vision has NOT yet shipped this into
production `SharedVisionBackbone` -- this script currently imports the
Upsample+Conv2d architecture from fpga/hls4ml_custom_layers/
fpga_target_architecture.py's stand-in reference class instead (same reason
that file has one -- see its docstring). It will NOT load a checkpoint still
using the OLD ConvTranspose2d architecture (strict state_dict load will raise
loudly, not silently misbehave). Once ml_vision ships the new architecture,
swap the import below back to
`from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone, temporal_split`.

Run as a module from the repo root:

    # Smoke-test the pipeline today, before the trained checkpoint exists tomorrow
    # (random weights, synthetic data, no disk/dataset dependency):
    python -m host_software.ml_vision.experiments.trial_fixed_point_quantization --dry-run

    # Real run once shared_vision_backbone_best.pt exists:
    python -m host_software.ml_vision.experiments.trial_fixed_point_quantization \
        --checkpoint host_software/ml_vision/models/shared_vision_backbone/shared_vision_backbone_best.pt

TODO once real weights exist: log observed per-layer activation min/max during the
float32 pass so int_bits can be chosen from actual dynamic range (docs/HLS_DATA_TYPES.md:
"you must mathematically prove the maximum possible value"), not guessed -- the configs
below are reasonable starting points, not derived from real data yet.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_software.ml_vision.evaluations.evaluate_shared_vision_backbone import mask_iou_dice
from host_software.ml_vision.training.augmentations import build_eval_transform
from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset
from host_software.ml_vision.training.train_cnn_2d_tracker_marker import temporal_split

# TODO once ml_vision ships Upsample+Conv2d into production (see module docstring):
# swap this back to
# `from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone`
from fpga.hls4ml_custom_layers.fpga_target_architecture import (
    _UpsampleConvReferenceArchitecture as SharedVisionBackbone,
)

DEFAULT_CSV = Path("host_software/data/03_gold/shared_vision/labels.csv")
DEFAULT_IMAGES_DIR = Path("host_software/data/03_gold/shared_vision/images")
DEFAULT_MASKS_DIR = Path("host_software/data/03_gold/shared_vision/masks")
DEFAULT_CHECKPOINT = Path("host_software/ml_vision/models/shared_vision_backbone/shared_vision_backbone_best.pt")
DEFAULT_OUTPUT_DIR = Path("host_software/ml_vision/experiments/results")

INPUT_SIZE = (128, 128)  # (H, W) -- matches production


@dataclass
class QuantConfig:
    """ap_fixed<W, I> for weights and activations tracked separately -- their value
    distributions differ (conv accumulation sums widen activations' dynamic range
    beyond the trained weights'), so a single shared (W, I) isn't realistic."""

    name: str
    weight_bits: Optional[int]  # None => float32 baseline, no quantization
    weight_int_bits: Optional[int]
    act_bits: Optional[int]
    act_int_bits: Optional[int]


QUANT_CONFIGS: List[QuantConfig] = [
    QuantConfig("float32_baseline", None, None, None, None),
    QuantConfig("fixed_16_6", 16, 6, 16, 8),
    QuantConfig("fixed_12_4", 12, 4, 12, 6),
    QuantConfig("fixed_8_2", 8, 2, 8, 4),
]


def fake_quantize(x: torch.Tensor, total_bits: int, int_bits: int) -> torch.Tensor:
    """Simulate ap_fixed<total_bits, int_bits> two's-complement rounding + saturation.
    docs/HLS_DATA_TYPES.md's convention: int_bits includes the sign bit, so the
    representable range is [-2^(int_bits-1), 2^(int_bits-1) - step]."""
    frac_bits = total_bits - int_bits
    step = 2.0 ** (-frac_bits)
    q_min = -(2.0 ** (int_bits - 1))
    q_max = 2.0 ** (int_bits - 1) - step
    return torch.clamp(torch.round(x / step) * step, q_min, q_max)


def _fold_conv_bn(conv: nn.Module, bn: nn.BatchNorm2d, transpose: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse BatchNorm2d into the preceding conv/deconv's weight+bias -- standard
    inference-time BN folding. ConvTranspose2d's weight layout is
    (in_channels, out_channels, kH, kW) -- the OPPOSITE of Conv2d's
    (out_channels, in_channels, kH, kW) -- so the per-output-channel scale broadcasts
    on a different dim depending on which one this is. Getting this backwards silently
    produces a numerically-plausible but wrong result, so it's handled explicitly."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv_bias = conv.bias if conv.bias is not None else torch.zeros(bn.num_features)

    if transpose:
        new_weight = conv.weight * scale.view(1, -1, 1, 1)  # broadcast over dim=1 (out_c)
    else:
        new_weight = conv.weight * scale.view(-1, 1, 1, 1)  # broadcast over dim=0 (out_c)

    new_bias = (conv_bias - bn.running_mean) * scale + bn.bias
    return new_weight.detach().clone(), new_bias.detach().clone()


class FoldedQuantizedBackbone(nn.Module):
    """Same forward graph as SharedVisionBackbone, but with every BatchNorm2d folded
    into its preceding conv/deconv (nothing left to run on-chip the HLS core wouldn't
    also run) and every weight/activation optionally snapped to an ap_fixed<W, I> grid.
    Built by from_pretrained() below; never trained directly."""

    def __init__(self, input_size: Tuple[int, int] = INPUT_SIZE) -> None:
        super().__init__()
        self.input_size = input_size
        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.GELU()  # Section 5.3: GELU is kept, not swapped -- decided by trial_activation_functions.py

        self.ball_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.ball_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ball_fc = nn.Linear(32, 2)

        # Upsample(nearest)+Conv2d(3x3), not ConvTranspose2d -- decided 2026-08-13, see
        # module docstring. Upsample itself has no weights; each stage's Conv2d does.
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.mask_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.mask_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.mask_conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.mask_out = nn.Conv2d(8, 1, 1)

        self.heatmap_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.heatmap_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.heatmap_conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.heatmap_out = nn.Conv2d(8, 1, 1)

        self.quant_cfg: Optional[QuantConfig] = None

    def _q_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_cfg is None or self.quant_cfg.act_bits is None:
            return x
        return fake_quantize(x, self.quant_cfg.act_bits, self.quant_cfg.act_int_bits)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._q_act(self.act(self.enc_conv1(x)))
        x = self.pool(x)
        x = self._q_act(self.act(self.enc_conv2(x)))
        x = self.pool(x)
        x = self._q_act(self.act(self.enc_conv3(x)))
        features = self.pool(x)

        b = self._q_act(self.act(self.ball_conv(features)))
        b = self.ball_pool(b).flatten(1)
        ball_xy = self.ball_fc(b)

        m = self._q_act(self.act(self.mask_conv1(self.up(features))))
        m = self._q_act(self.act(self.mask_conv2(self.up(m))))
        m = self._q_act(self.act(self.mask_conv3(self.up(m))))
        mask_logits = self.mask_out(m)

        h = self._q_act(self.act(self.heatmap_conv1(self.up(features))))
        h = self._q_act(self.act(self.heatmap_conv2(self.up(h))))
        h = self._q_act(self.act(self.heatmap_conv3(self.up(h))))
        heatmap_logits = self.heatmap_out(h)

        # F.interpolate(..., size=input_size) is a documented no-op given kernel=stride=2
        # three times from the 16x16 bottleneck (Section 5.4 of the budget doc) --
        # omitted here on purpose, matching what the FPGA implementation should do.
        return ball_xy, mask_logits[:, :1], heatmap_logits[:, :1]

    @classmethod
    def from_pretrained(cls, source: SharedVisionBackbone, quant_cfg: QuantConfig) -> "FoldedQuantizedBackbone":
        model = cls(input_size=source.input_size)

        # mask_head/heatmap_head Sequential layout (Upsample+Conv2d, decided 2026-08-13):
        # [0]Upsample [1]Conv2d(64,32) [2]BN(32) [3]GELU [4]Upsample [5]Conv2d(32,16)
        # [6]BN(16) [7]GELU [8]Upsample [9]Conv2d(16,8) [10]BN(8) [11]GELU [12]Conv2d(8,1,1)
        # -- Upsample has no state_dict entries (parameter-free), so it doesn't shift
        # indices relative to itself, but it DOES shift every index after it relative
        # to the old ConvTranspose2d layout. All conv/BN pairs are now regular
        # Conv2d+BatchNorm2d (transpose=False), not ConvTranspose2d+BatchNorm2d.
        conv_bn_pairs = [
            (model.enc_conv1, source.encoder[0], source.encoder[1], False),
            (model.enc_conv2, source.encoder[4], source.encoder[5], False),
            (model.enc_conv3, source.encoder[8], source.encoder[9], False),
            (model.ball_conv, source.ball_head[0], source.ball_head[1], False),
            (model.mask_conv1, source.mask_head[1], source.mask_head[2], False),
            (model.mask_conv2, source.mask_head[5], source.mask_head[6], False),
            (model.mask_conv3, source.mask_head[9], source.mask_head[10], False),
            (model.heatmap_conv1, source.heatmap_head[1], source.heatmap_head[2], False),
            (model.heatmap_conv2, source.heatmap_head[5], source.heatmap_head[6], False),
            (model.heatmap_conv3, source.heatmap_head[9], source.heatmap_head[10], False),
        ]
        for dst_conv, src_conv, src_bn, transpose in conv_bn_pairs:
            weight, bias = _fold_conv_bn(src_conv, src_bn, transpose)
            if quant_cfg.weight_bits is not None:
                weight = fake_quantize(weight, quant_cfg.weight_bits, quant_cfg.weight_int_bits)
                bias = fake_quantize(bias, quant_cfg.weight_bits, quant_cfg.weight_int_bits)
            dst_conv.weight.data.copy_(weight)
            dst_conv.bias.data.copy_(bias)

        # No BN follows these -- copy (quantized if requested) as-is.
        no_bn_layers = [
            (model.mask_out, source.mask_head[12]),
            (model.heatmap_out, source.heatmap_head[12]),
            (model.ball_fc, source.ball_head[5]),
        ]
        for dst, src in no_bn_layers:
            weight, bias = src.weight.detach().clone(), src.bias.detach().clone()
            if quant_cfg.weight_bits is not None:
                weight = fake_quantize(weight, quant_cfg.weight_bits, quant_cfg.weight_int_bits)
                bias = fake_quantize(bias, quant_cfg.weight_bits, quant_cfg.weight_int_bits)
            dst.weight.data.copy_(weight)
            dst.bias.data.copy_(bias)

        model.quant_cfg = quant_cfg
        model.eval()
        return model


def evaluate_config(model: FoldedQuantizedBackbone, loader: DataLoader, device: torch.device) -> dict:
    px_scale = torch.tensor([INPUT_SIZE[1], INPUT_SIZE[0]], device=device, dtype=torch.float32)
    heatmap_criterion = nn.MSELoss(reduction="sum")

    px_error_sum, n_samples = 0.0, 0
    iou_sum, dice_sum = 0.0, 0.0
    heatmap_sq_error_sum, n_heatmap_px = 0.0, 0

    model.eval()
    with torch.no_grad():
        for images, ball_xy, masks, heatmap_targets in loader:
            images, ball_xy, masks, heatmap_targets = (
                images.to(device), ball_xy.to(device), masks.to(device), heatmap_targets.to(device)
            )
            pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)

            px_error = (pred_ball_xy - ball_xy) * px_scale
            px_error_sum += px_error.norm(dim=1).sum().item()
            n_samples += images.size(0)

            batch_iou, batch_dice = mask_iou_dice(pred_mask_logits, masks)
            iou_sum += batch_iou
            dice_sum += batch_dice

            heatmap_sq_error_sum += heatmap_criterion(torch.sigmoid(pred_heatmap_logits), heatmap_targets).item()
            n_heatmap_px += heatmap_targets.numel()

    return {
        "ball_px_error": px_error_sum / max(n_samples, 1),
        "mask_iou": iou_sum / max(n_samples, 1),
        "mask_dice": dice_sum / max(n_samples, 1),
        "heatmap_mse": heatmap_sq_error_sum / max(n_heatmap_px, 1),
    }


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    source = SharedVisionBackbone(input_size=INPUT_SIZE).to(device)
    source.load_state_dict(torch.load(args.checkpoint, map_location=device))
    source.eval()

    labels_df = pd.read_csv(args.csv_file)
    _, val_df = temporal_split(labels_df, val_fraction=args.val_fraction)
    dataset = SharedVisionDataset(
        csv_file="", root_dir=str(args.images_dir), mask_dir=str(args.masks_dir),
        input_size=INPUT_SIZE, transform=build_eval_transform(INPUT_SIZE), labels_df=val_df,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} held-out validation frames.")

    results = []
    for cfg in QUANT_CONFIGS:
        print(f"\nConfig: {cfg.name}")
        model = FoldedQuantizedBackbone.from_pretrained(source, cfg).to(device)
        metrics = evaluate_config(model, loader, device)
        print(f"  ball_px_error={metrics['ball_px_error']:.2f}px  iou={metrics['mask_iou']:.3f}  "
              f"heatmap_mse={metrics['heatmap_mse']:.4f}")
        results.append({"config": cfg.name, **metrics})

    return pd.DataFrame(results)


def _check_fold_equivalence(source: SharedVisionBackbone, device: torch.device) -> None:
    """"Runs without crashing" doesn't prove the BN-fold math (Section 5.5) is
    correct -- a bug in the ConvTranspose2d dim=1-vs-Conv2d dim=0 broadcast, or in
    treating F.interpolate as a no-op, would still execute cleanly and just silently
    produce wrong numbers. Directly compare the float32-config folded model's output
    against the original unfused SharedVisionBackbone's forward pass on the same
    input -- they should match to floating-point rounding, not just "be close-ish"."""
    probe = torch.rand(2, 3, *INPUT_SIZE, device=device)
    with torch.no_grad():
        orig_ball, orig_mask, orig_heatmap = source(probe)

        float32_cfg = QuantConfig("float32_baseline", None, None, None, None)
        folded = FoldedQuantizedBackbone.from_pretrained(source, float32_cfg).to(device)
        folded_ball, folded_mask, folded_heatmap = folded(probe)

    ball_diff = (orig_ball - folded_ball).abs().max().item()
    mask_diff = (orig_mask - folded_mask).abs().max().item()
    heatmap_diff = (orig_heatmap - folded_heatmap).abs().max().item()
    print(
        f"Fold equivalence check (orig vs. folded, float32, same input) -- "
        f"max abs diff: ball={ball_diff:.2e} mask={mask_diff:.2e} heatmap={heatmap_diff:.2e}"
    )
    tolerance = 1e-4
    if max(ball_diff, mask_diff, heatmap_diff) > tolerance:
        raise AssertionError(
            f"BN-fold/no-op-interpolate equivalence check FAILED (tolerance={tolerance}) -- "
            "the folded model's forward pass diverges from the original architecture's. "
            "Do not trust quantization results from this script until this is fixed."
        )
    print("Fold equivalence check PASSED -- folded float32 model matches the original architecture.")


def run_dry_run_smoke_test() -> pd.DataFrame:
    """Structural self-test with a freshly-initialized (untrained) model and synthetic
    random data -- no checkpoint, no dataset on disk required. Proves the fold +
    quantize + forward + eval pipeline runs end-to-end without crashing, AND that the
    fold math is actually correct (see _check_fold_equivalence), not just crash-free.
    The metrics table below is NOT a real accuracy result (random weights + random
    data), only a pipeline sanity check."""
    print("=== DRY RUN: synthetic data, untrained weights -- numbers below are NOT meaningful ===")
    device = torch.device("cpu")
    torch.manual_seed(0)

    source = SharedVisionBackbone(input_size=INPUT_SIZE).to(device)
    source.eval()  # untrained BN running stats default to mean=0/var=1, fine for a smoke test

    _check_fold_equivalence(source, device)

    batch_size, n_batches = 4, 3
    images = [torch.rand(batch_size, 3, *INPUT_SIZE) for _ in range(n_batches)]
    ball_xy = [torch.rand(batch_size, 2) for _ in range(n_batches)]
    masks = [torch.randint(0, 2, (batch_size, 1, *INPUT_SIZE)).float() for _ in range(n_batches)]
    heatmaps = [torch.rand(batch_size, 1, *INPUT_SIZE) for _ in range(n_batches)]
    synthetic_batches = list(zip(images, ball_xy, masks, heatmaps))

    results = []
    for cfg in QUANT_CONFIGS:
        print(f"\nConfig: {cfg.name}")
        model = FoldedQuantizedBackbone.from_pretrained(source, cfg).to(device)
        metrics = evaluate_config(model, synthetic_batches, device)
        print(f"  ball_px_error={metrics['ball_px_error']:.2f}px  iou={metrics['mask_iou']:.3f}  "
              f"heatmap_mse={metrics['heatmap_mse']:.4f}")
        results.append({"config": cfg.name, **metrics})

    df = pd.DataFrame(results)
    print("\nDry run completed without errors -- pipeline is structurally sound.")
    return df


def save_report(results_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to {csv_path}")
    print(results_df.to_string(index=False))

    fig, (ax_px, ax_iou) = plt.subplots(1, 2, figsize=(11, 5))
    ax_px.bar(results_df["config"], results_df["ball_px_error"], color="steelblue")
    ax_px.set_ylabel("Val ball position error (px)")
    ax_px.set_title("Ball Position Error by Quantization Config")
    ax_px.tick_params(axis="x", rotation=20)
    ax_px.grid(True, axis="y", alpha=0.3)

    ax_iou.bar(results_df["config"], results_df["mask_iou"], color="darkorange")
    ax_iou.set_ylabel("Val mask IoU")
    ax_iou.set_ylim(0, 1)
    ax_iou.set_title("Marker Mask IoU by Quantization Config")
    ax_iou.tick_params(axis="x", rotation=20)
    ax_iou.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ap_fixed<W,I> Quantization Sensitivity -- Shared Vision Backbone")
    fig.tight_layout()
    plot_path = output_dir / f"{prefix}_{timestamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ap_fixed<W,I> quantization accuracy cost for the Shared Vision Backbone (FPGA port)"
    )
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Must match the --val-fraction used at training time, so the held-out split lines up",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Structural smoke test with synthetic data and untrained weights -- no checkpoint/dataset needed",
    )
    args = parser.parse_args()

    if args.dry_run:
        df = run_dry_run_smoke_test()
        save_report(df, args.output_dir, prefix="dryrun_fixed_point_trial")
        return

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Train it first, or pass --dry-run to smoke-test "
            "the pipeline without a real checkpoint."
        )
    if not args.csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_file}. Run merge_shared_vision_sessions.py first.")

    results_df = run_sweep(args)
    save_report(results_df, args.output_dir, prefix="fixed_point_trial")


if __name__ == "__main__":
    main()
