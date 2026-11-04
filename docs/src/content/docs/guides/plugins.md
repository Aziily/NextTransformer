---
title: Add a method (plugins)
description: Add a new architecture as a plugin, without editing shared files.
---

A new method is a plugin: one directory under `nexttransformer/methods/` that the registry discovers. It needs no
change to shared files, so it cannot change another method. Its cost proof fingerprints only its own package and the
shared core. Please [open a proposal issue](https://github.com/Interactive-Active-AI/NextTransformer/issues/new/choose)
before you start.

## Layout

```text
nexttransformer/methods/<name>/
  method.json            declaration, read without importing code
  __init__.py            hooks (other modules of the method may sit next to it)
configs/<family>/<name>.json                        calibrated configuration, written by calibrate
evidence/costs/<proof-set>/<name>.json, dense.json  cost proof and the SSSL anchor proof, written by calibrate
results/methods/<name>.json                         model card
tests/test_<name>.py                                tests
```

## Declaration

```json
{
 "name": "output_gate",
 "family": "others",
 "title": "Attention with a learned output gate",
 "paper": {"title": "<paper title>", "url": "<paper URL>"},
 "options": {"gate_bias": 0.0},
 "debug_options": {"gate_bias": 0.5},
 "shell": {"layers": 24, "mlp_width": 6144, "window_pattern": "L"},
 "layer_multiple": 1
}
```

| Field | Meaning |
|---|---|
| `name` | Lowercase letters, digits and `_`; equals the directory name; not a built-in method |
| `family` | Display family: `looped`, `sparse`, `linear`, `residual`, `hybrid` or `others` |
| `options` | The method's own parameters and their defaults; read with `config.option(name)` |
| `debug_options` | Values for the small CPU configuration used by tests (`debug_config`) |
| `shell` | Starting `layers`, `mlp_width` and `window_pattern` of the shell (default 24, 6144, `L`) |
| `layer_multiple` | Size of a repeating unit; calibration and validation keep the layer count a multiple of it |

## Hooks

Define one or more in `__init__.py`:

| Hook | Use |
|---|---|
| `mixer(config, layer)` | Returns the attention module of a layer, called as `(x, ve, cos_sin, window, kv_cache)` |
| `ffn(config, layer)` | Returns the feed-forward module of a layer, called as `(x)` |
| `block(block, config, layer)` | Wraps or replaces a block, called as `(x, ve, cos_sin, window, kv_cache)` |
| `model` | A subclass of `nexttransformer.models.lm.ReferenceLM`, for a method that changes the forward flow |
| `frozen` | Names of shell parameters the method leaves unused, such as `("backout_lambda",)` |
| `init(model)` | Initializes the method's own parameters after the shared initialization |
| `stages(config)` | Contiguous training stages in order, `[(name, updates), ...]` summing to 46,875; default one `joint` stage |
| `check(config)` | Raises `ValueError` for invalid option values |

A plugin keeps to the shell's contract:

- Every block keeps `attn.c_q`, `attn.c_k`, `attn.c_v`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` as linear layers;
  the shell initializes them, and the projections into the residual stream start at zero.
- Every trainable parameter receives a gradient at every update; list unused shell parameters in `frozen`.
- A module may set a tensor `last_aux` (an auxiliary loss added in training) and define `set_step(step)`.
- Parameters keep stable names once a run exists: checkpoints store tensors by name.

A plugin cannot change a built-in or another plugin, the cost ledger, the trainer, the optimizer, the data or the
evaluation, and it cannot add a display family.

## Example

```python
"""Causal attention whose output passes a learned sigmoid gate (an illustration of the hooks)."""
import torch
from torch import nn

from nexttransformer.models.backbone import CausalSelfAttention, Linear


class GatedAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.gate = Linear(config.width, config.width, bias=False)
        self.bias = config.option("gate_bias")

    def forward(self, x, ve, cos_sin, window, kv_cache):
        return super().forward(x, ve, cos_sin, window, kv_cache) * torch.sigmoid(self.gate(x) + self.bias)


def mixer(config, layer):
    return GatedAttention(config, layer)


def init(model):
    for block in model.transformer.h:
        nn.init.zeros_(block.attn.gate.weight)
```

The test plugins in `tests/plugins/` cover a mixer, a feed-forward network with an option check, and a whole-model
plugin with two training stages and a frozen shell parameter. A published method should follow its paper's exact
formulation and document any adaptation in the [source notes](../../methods/sources/).

## Cost

Calibrate the method to the shared budget, within 1% of the SSSL anchor:

```bash
python -m nexttransformer.protocol.calibrate --methods <name> --cache .agent-work/cost-cache \
    --output configs --results evidence/costs/<name>-<date>
```

Calibration searches the registered grid (16 to 32 layers in steps of `layer_multiple`, then FFN widths from 3d to 5d
in multiples of 128), writes the configuration and the proof, and the SSSL anchor proof next to it.

- Compute softmax attention with `nanochat.flash_attention.flash_attn.flash_attn_func`; the ledger counts its causal
  or windowed edges exactly.
- Express everything else with operators the ledger counts; the audit names any uncovered operator and refuses.
- A custom kernel needs a pure PyTorch path that the CPU audit runs, and a test showing both compute the same function.

Formal results need the audited PyTorch build (2.9.1+cu128); research mode trains with any build.

## Before the pull request

```bash
python -m pytest tests/test_<name>.py -q && python -m pytest -q -n auto
python -m nexttransformer.protocol.fingerprint        # every published method still matches its proof
python -m nexttransformer.arena --scope origin/main   # the pull request adds only the method's own files
python -m nexttransformer.checks.changes --base origin/main
launch/train.sh --config configs/<family>/<name>.json --site-config site.local.json \
    --output runs/<name>-smoke --micro-batch 8 --smoke-steps 2
```

After these checks, train the calibrated configuration on your fork with `--formal`. Follow the
[result guide](../submit-a-result/) to audit and export the completed run. Then open one pull request with the
plugin, calibrated configuration, cost proof, tests, model card, source notes and formal result. Link the proposal
issue. A maintainer re-audits the method before merging. For an Arena entry, follow the [Arena rules](../../arena/rules/).

To try plugins kept outside the repository, list their roots in `NEXTTRANSFORMER_METHOD_PATHS`. Fingerprints cover
only plugin roots inside the repository, so such methods are for research.
