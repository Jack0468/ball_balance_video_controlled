# hls4ml Custom Layers — FPGA Vision Backbone Port

Working area for getting `SharedVisionBackbone` (`host_software/ml_vision/training/train_cnn_2d_tracker_marker.py`)
through hls4ml's PyTorch → HLS conversion pipeline. See `docs/plans/ml_system_parameter_budget.md`
Sections 5.8-5.9 for the full research trail and reasoning behind everything here — this
README only covers setup and a status summary.

## Environment

hls4ml is **not** part of the project's shared `ball_balance_env` — that environment is
pinned and shared with the ml_vision/ml_audio agents (`AGENTS.md`), and hls4ml pulls its
own dependency tree. It lives in a separate, isolated conda environment instead:

```
conda create -y -n vri_fpga_hls4ml python=3.10
C:/Users/Admin/.conda/envs/vri_fpga_hls4ml/python.exe -m pip install hls4ml torch \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Installed and verified 2026-08-12: `hls4ml==1.3.0`, `torch==2.13.0+cpu`.

The `fpga_target_architecture.py` equivalence check doesn't need hls4ml at all (pure
PyTorch, needs `pandas`/`albumentations` for the production model's imports) — run that
one with `ball_balance_env` instead. `gelu_lut.py` and `test_gelu_lut_conversion.py` need
`vri_fpga_hls4ml`.

## Status (updated 2026-08-13)

| Item | Status | Result |
|---|---|---|
| `gelu_lut.py` — custom hls4ml GELU layer | Conversion verified | `hls4ml.converters.convert_from_pytorch_model` accepts it; confirmed in the model graph log (`Layer name: act, layer type: HGeluLUT`). C-simulation did not run — Windows shell incompatibility in hls4ml's build scripts (`'.' is not recognized...`), not a flaw in the layer. Likely needs WSL or a Linux box for real csim/synthesis, not native Windows — worth confirming before further HLS work. Real Vitis HLS synthesis unverified (no Vitis installed on this machine). |
| `fpga_target_architecture.py` — `AvgPool2d(16)` swap for `AdaptiveAvgPool2d((1,1))` | Verified exact | Numerically confirmed equivalent to the reconstructed production architecture (max abs diff ~7e-9, floating-point noise), not just asserted from the math. |
| `fpga_target_architecture.py` / `../../host_software/ml_vision/experiments/trial_fixed_point_quantization.py` — `Upsample+Conv2d` decoder (replacing `ConvTranspose2d`) | Updated & verified 2026-08-13 | Decoder architecture reconstructed from the reweighting trial's reported param count only (91,140) — no source code was available, so the reconstruction was verified two ways before trusting it: (1) `fpga_target_architecture.py`'s param count matches 91,140 exactly, and (2) `trial_fixed_point_quantization.py`'s BN-fold equivalence check (folded vs. unfused, same input) passed at floating-point noise level (~5e-8), which would have failed loudly on any index-mapping mistake in the new `Upsample`-containing `Sequential` layout. **Both files currently import a stand-in architecture** (`_UpsampleConvReferenceArchitecture` in `fpga_target_architecture.py`), not the real production `SharedVisionBackbone` — see that file's docstring for exactly what to change once `ml_vision` ships this into production. Until then, neither script can load a real trained checkpoint (there isn't one yet with this architecture) — both were verified with untrained/synthetic weights only. |

## Gotchas hit along the way

- `hls_config['InputShape']` must be a **list of shapes** (`[[8]]`), not a bare shape
  (`[8]`) — passing a bare shape fails with a confusing `TypeError: 'int' object is not
  subscriptable` deep in `pytorch_to_hls.py`, not an obviously-input-shape-related error.
- `register_pytorch_layer_handler` lives at `hls4ml.converters.register_pytorch_layer_handler`
  (verified against the installed 1.3.0 package directly, not assumed from docs).

## Not yet done

- Full 4D CNN conversion attempt (this only proves the GELU layer mechanism on a toy
  1D `Linear -> GELU` model, not the real architecture — see `docs/plans/
  ml_system_parameter_budget.md` §5.9 step 5).
- `ConvTranspose2d` handling (custom layer vs. the `Upsample + Conv2d` retrain option —
  §5.9 step 3, blocked on the retrained checkpoint).
- Real Vitis HLS synthesis of anything here.
