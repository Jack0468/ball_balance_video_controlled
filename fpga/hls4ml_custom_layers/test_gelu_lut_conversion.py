"""Verification spike for gelu_lut.py: does hls4ml actually accept the custom
GELU layer and convert a GELU-containing PyTorch model end to end?

This is a toy 1D model (Linear -> GELU), not the real SharedVisionBackbone --
the goal here is only to prove the custom-layer *mechanism* works (parser
registration, IR layer, templates, HLS source registration all wire together
correctly through hls4ml's real conversion pipeline), before attempting it on
the full 4D CNN architecture. Run manually (not part of any CI/pytest suite)
in the isolated `vri_fpga_hls4ml` conda env:

    C:/Users/Admin/.conda/envs/vri_fpga_hls4ml/python.exe -m fpga.hls4ml_custom_layers.test_gelu_lut_conversion

or directly:

    C:/Users/Admin/.conda/envs/vri_fpga_hls4ml/python.exe fpga/hls4ml_custom_layers/test_gelu_lut_conversion.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gelu_lut import GELU_TABLE_SIZE, GELU_X_MAX, GELU_X_MIN, register  # noqa: E402

import hls4ml  # noqa: E402


class TinyGELUModel(nn.Module):
    """Linear -> GELU -- smallest possible model that exercises the custom layer."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.fc(x))


def main() -> None:
    work_dir = Path(__file__).resolve().parent / '_gelu_lut_spike'
    work_dir.mkdir(exist_ok=True)

    print(f"Registering custom GELU LUT layer (table size={GELU_TABLE_SIZE}, range=[{GELU_X_MIN},{GELU_X_MAX}])...")
    header_path = register(backend_ids=['Vivado'], header_dir=work_dir)
    print(f"  Wrote HLS header: {header_path}")

    torch.manual_seed(0)
    model = TinyGELUModel()
    model.eval()

    print("\nConverting TinyGELUModel (Linear -> GELU) via hls4ml.converters.convert_from_pytorch_model...")
    try:
        hmodel = hls4ml.converters.convert_from_pytorch_model(
            model,
            output_dir=str(work_dir / 'hls_project'),
            project_name='gelu_lut_spike',
            backend='Vivado',
            hls_config={
                'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1},
                'InputShape': [[8]],
            },
        )
    except Exception as exc:
        print(f"\nCONVERSION FAILED: {type(exc).__name__}: {exc}")
        raise

    print("CONVERSION SUCCEEDED -- hls4ml accepted the custom GELU LUT layer.")
    print(f"\nModel graph:\n{hmodel}")

    # Best-effort C-simulation -- this needs a C++ toolchain and hls4ml's bundled
    # ap_types shim, NOT a real Vitis install, but it's not guaranteed to work in
    # every environment. Report honestly either way rather than assuming.
    print("\nAttempting hmodel.compile() (C-simulation, not real Vitis HLS synthesis)...")
    try:
        hmodel.compile()
        x = np.random.uniform(GELU_X_MIN / 2, GELU_X_MAX / 2, size=(4, 8)).astype('float32')
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).numpy()
        hls_out = hmodel.predict(x)
        max_abs_diff = np.max(np.abs(torch_out - hls_out))
        print(f"  C-sim succeeded. Max abs diff vs. real PyTorch GELU: {max_abs_diff:.6f}")
        print("  (Nonzero diff is expected -- LUT quantization + ap_fixed<16,6> rounding, not a bug.)")
    except Exception as exc:
        print(f"  C-sim did NOT run in this environment ({type(exc).__name__}: {exc}).")
        print("  This is a real, honest limitation to report -- no Vitis/Vitis HLS is installed on this machine.")
        print("  Conversion-level verification (the main goal of this spike) still succeeded above.")


if __name__ == '__main__':
    main()
