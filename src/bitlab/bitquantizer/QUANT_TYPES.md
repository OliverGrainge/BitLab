## BitQuantizer Schemes

### Activation Quantizers
- `ai8pc` — Int8 per-channel for 4D conv activations; scales broadcast over H×W.
- `ai8pt` — Int8 per-tensor for 2D activations; single scale for whole batch.
- `ai8ptk` — Int8 per-token (row) for 2D activations; scale per batch element.
- `abf16` / `af16` — bfloat16 / float16 passthrough (scale fixed at 1).
- `none` — Identity; returns tensor unchanged with scale 1.

### Weight Quantizers
- `wpt` — Ternary per-tensor; single global scale.
- `wpc` — Ternary per-output-channel (rows for linear, filters for conv).
- `wpg` / `wpg{N}` — Ternary per-group along input features (default group size 128).
- `wbf16` / `wf16` — bfloat16 / float16 passthrough (scale fixed at 1).
- `none` — Identity passthrough.

### Combined Keys
Quantizer keys follow `{activation}_{weight}` (e.g., `ai8pc_wpt`). The registry dynamically builds matching autograd functions for every activation/weight pair, plus the special `none` entry that disables quantization.

