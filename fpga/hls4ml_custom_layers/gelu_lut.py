"""Custom hls4ml layer: GELU via a precomputed lookup table.

GELU(x) = 0.5*x*(1 + erf(x/sqrt(2))) requires erf(), which has no native
fixed-point/HLS implementation -- docs/HLS_DATA_TYPES.md, and
docs/plans/ml_system_parameter_budget.md Section 5.3/5.8. SharedVisionBackbone
keeps GELU (empirically justified over ReLU -- see the budget doc Section 5.3;
ReLU cost ~16% relative ball-position accuracy and showed severe seed-to-seed
instability), so every one of its 10 activation sites needs this.

hls4ml's PyTorch converter has no GELU handler (verified directly against the
installed hls4ml==1.3.0 package's hls4ml.converters.get_supported_pytorch_layers()
list, 2026-08-12 -- not assumed from documentation). This module follows
hls4ml's documented Extension API pattern for adding one: an hls4ml IR layer
class, a PyTorch parser registered via hls4ml.converters.register_pytorch_layer_handler,
config/function templates, and a generated HLS header. The exact API shapes
below (parser signature, Layer/LayerConfigTemplate/FunctionCallTemplate base
classes, registration function names) were verified against hls4ml's own
test/pytest/test_extensions.py (the KReverse/HReverse reference example) and
hls4ml/converters/pytorch/core.py, fetched from the fastmachinelearning/hls4ml
GitHub repo and cross-checked against the actually-installed 1.3.0 package
(not assumed from search-result summaries -- see docs/plans/
ml_system_parameter_budget.md Section 5.8 for why that distinction mattered
here: an earlier pass over-trusted search summaries and got the supported-
layer list wrong).

Verified so far (see test_gelu_lut_conversion.py in this directory):
- This layer registers and converts successfully through hls4ml's PyTorch
  frontend end to end.
NOT yet verified:
- Real Vitis HLS C-synthesis. This machine does not have Vitis installed.
  Per agent_fpga.md's hard rule on Vitis/HLS syntax, treat the generated
  header's pragmas as a reasonable first draft to check against the real
  tool, not a synthesis-verified result.

Setup: this needs the `hls4ml` package, which is NOT part of the project's
shared `ball_balance_env` (that env is pinned/shared with the ml_vision and
ml_audio agents -- see AGENTS.md -- and hls4ml pulls its own dependency tree,
so it lives in a separate, isolated conda env instead: `vri_fpga_hls4ml`
(python=3.10, `pip install hls4ml torch`). See README.md in this directory.
"""

import math
from pathlib import Path
from typing import List

import hls4ml
import hls4ml.backends
import hls4ml.converters
import hls4ml.model.layers

GELU_TABLE_SIZE = 1024
GELU_X_MIN = -8.0
GELU_X_MAX = 8.0
# Beyond +-8, GELU(x) is within float32 rounding of 0 or x -- safe saturation bounds.


def _gelu_reference(x: float) -> float:
    """The real GELU formula (Python float, math.erf) -- used only to bake the
    LUT values below. Never used at HLS runtime."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def generate_gelu_lut_header() -> str:
    """Bake the GELU lookup table into a static C array, mirroring how a real
    hardware LUT/BRAM-resident activation table is initialized -- computed
    once here in Python, not synthesized on-chip."""
    step = (GELU_X_MAX - GELU_X_MIN) / GELU_TABLE_SIZE
    values = [_gelu_reference(GELU_X_MIN + (i + 0.5) * step) for i in range(GELU_TABLE_SIZE)]
    table_lines = ",\n    ".join(f"{v:.8f}f" for v in values)

    return f"""#ifndef NNET_GELU_LUT_H_
#define NNET_GELU_LUT_H_

#include "nnet_common.h"

namespace nnet {{

struct gelu_lut_config {{
    static const unsigned n_in = 10;
    static const unsigned table_size = {GELU_TABLE_SIZE};
}};

// Precomputed GELU(x) = 0.5*x*(1+erf(x/sqrt(2))) over x in [{GELU_X_MIN}, {GELU_X_MAX}],
// {GELU_TABLE_SIZE} bins -- erf() has no native fixed-point/HLS form, so this table is
// computed once in Python (math.erf) and baked in here, NOT computed on-chip.
// NOTE: index computed via division by CONFIG_T::table_size, evaluated once into a
// `static const` per call, not re-divided per loop iteration -- standard C++ semantics,
// but NOT yet verified against real Vitis HLS synthesis on this machine (no Vitis
// install available). Treat as a first draft to check against the real tool.
static const float gelu_table[{GELU_TABLE_SIZE}] = {{
    {table_lines}
}};

template<class data_T, typename CONFIG_T>
void gelu_lut(
    data_T input[CONFIG_T::n_in],
    data_T output[CONFIG_T::n_in]
) {{
    #pragma HLS PIPELINE
    static const data_T x_min = ({GELU_X_MIN});
    static const data_T x_max = ({GELU_X_MAX});
    static const data_T step = (x_max - x_min) / (data_T) CONFIG_T::table_size;

    for (int i = 0; i < CONFIG_T::n_in; i++) {{
        #pragma HLS UNROLL
        data_T x = input[i];
        if (x < x_min) x = x_min;
        if (x > x_max) x = x_max;
        int idx = (int)((x - x_min) / step);
        if (idx < 0) idx = 0;
        if (idx >= (int) CONFIG_T::table_size) idx = (int) CONFIG_T::table_size - 1;
        output[i] = (data_T) gelu_table[idx];
    }}
}}

}}

#endif
"""


class HGeluLUT(hls4ml.model.layers.Layer):
    """hls4ml IR node for a GELU activation implemented via lookup table.
    Elementwise op -- output shape always equals input shape."""

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        self.add_output_variable(shape)


def parse_gelu_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    """PyTorch parser for nn.GELU -- signature verified against hls4ml's real
    @pytorch_handler-decorated parsers (e.g. parse_activation_layer,
    parse_batchnorm_layer in hls4ml/converters/pytorch/core.py). The n_in
    computation (flatten all non-batch dims) mirrors parse_batchnorm_layer's
    approach for the same reason: an elementwise op over an arbitrary-shaped
    CNN feature map, not a flat 1D vector."""
    assert 'GELU' in operation

    layer = {}
    layer['class_name'] = 'HGeluLUT'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
    layer['n_in'] = in_size

    return layer, input_shapes[0][:]


gelu_lut_config_template = (
    "struct config{index} : nnet::gelu_lut_config {{\n"
    "    static const unsigned n_in = {n_in};\n"
    f"    static const unsigned table_size = {GELU_TABLE_SIZE};\n"
    "}};\n"
)
gelu_lut_function_template = 'nnet::gelu_lut<{input_t}, {config}>({input}, {output});'
gelu_lut_include_list = ['nnet_utils/nnet_gelu_lut.h']


class HGeluLUTConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HGeluLUT)
        self.template = gelu_lut_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class HGeluLUTFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HGeluLUT, include_header=gelu_lut_include_list)
        self.template = gelu_lut_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def register(backend_ids: List[str] = ('Vivado', 'Vitis'), header_dir: Path = None) -> Path:
    """Register the GELU LUT layer with hls4ml for the given backend(s). Call
    this once before hls4ml.converters.convert_from_pytorch_model(). Returns
    the path of the written HLS header (caller doesn't need it, but useful
    for inspection/debugging)."""
    hls4ml.converters.register_pytorch_layer_handler('GELU', parse_gelu_layer)
    hls4ml.model.layers.register_layer('HGeluLUT', HGeluLUT)

    header_dir = header_dir or Path.cwd()
    header_path = header_dir / 'nnet_gelu_lut.h'
    header_path.write_text(generate_gelu_lut_header())

    for backend_id in backend_ids:
        backend = hls4ml.backends.get_backend(backend_id)
        backend.register_template(HGeluLUTConfigTemplate)
        backend.register_template(HGeluLUTFunctionTemplate)
        backend.register_source(header_path)

    return header_path
