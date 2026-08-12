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

## Status (2026-08-12)

| Item | Status | Result |
|---|---|---|
| `gelu_lut.py` — custom hls4ml GELU layer | Conversion verified | `hls4ml.converters.convert_from_pytorch_model` accepts it; confirmed in the model graph log (`Layer name: act, layer type: HGeluLUT`). C-simulation did not run — Windows shell incompatibility in hls4ml's build scripts (`'.' is not recognized...`), not a flaw in the layer. Likely needs WSL or a Linux box for real csim/synthesis, not native Windows — worth confirming before further HLS work. Real Vitis HLS synthesis unverified (no Vitis installed on this machine). |
| `fpga_target_architecture.py` — `AvgPool2d(16)` swap for `AdaptiveAvgPool2d((1,1))` | Verified exact | Numerically confirmed equivalent to the production architecture (max abs diff ~7e-9, floating-point noise), not just asserted from the math. Safe to use as-is. |

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
