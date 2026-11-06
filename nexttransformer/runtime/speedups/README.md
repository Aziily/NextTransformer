# Execution speedups for shared components

`--speedups` makes training faster without changing what a method computes. It accelerates the parts
that every architecture shares: the shell around the mixer, the loss and the data path.

```bash
launch/train.sh --config configs/baseline/dense_full.json --output runs/x --micro-batch 16 --speedups all
```

| Name | Replaces | Scope |
|---|---|---|
| `norm` | `F.rms_norm`, with several rows per program | every method |
| `mlp` | `F.relu(h).square()`, one pass each way | every method with the shared FFN |
| `residual` | `resid_lambdas[i]*x+x0_lambdas[i]*x0` | methods with the native residual scalars |
| `readout` | soft cap and cross entropy around ATen's `log_softmax` | every method |
| `embedding` | dense embedding gradients, one sort per batch for all tables | every method |
| `attention` | rotary + QK norm + scale, the value-embedding gate, and the input gradient of the projections | softmax attention layers |
| `data` | row packing on a background thread | every method with its own stream per rank |
| `weights` | `weight.to(bf16)`, once per update instead of once per micro batch | every `Linear` |
| `addnorm` | `x+attn(...)` with the next norm, the residual mix with the first norm, and the residual gradients around them | methods on the shared `Block` |

`exact` selects all of the above. Each one reproduces the eager result bit for bit: rounding points
are preserved, and reductions keep ATen's exact order. Parameters, optimizer state and the data
cursor after a run with `exact` equal those of a run without it, on the deterministic path.

`all` adds two options that compute the same function with other rounding, so such a run is not
comparable bit for bit:

| Name | Replaces |
|---|---|
| `cudnn` | prefers cuDNN SDPA where FlashAttention-3 is unavailable; off with `--deterministic-attention` |
| `merge` | removes intermediate roundings to save memory traffic: projection gradients add inside the GEMMs, the residual scalars' gradients sum inside the kernel, rotary embedding rounds once, weight gradients leave the GEMM in FP32, and the MLP residual is written over its input |

On a run of `dense_full`, the loss of `all` stays within what two runs of the default path differ by,
since SDPA flash is not deterministic either. Use `exact` where a run must match the default path.

## Reproduce

`python -m nexttransformer.checks.speedups` covers the GPU tests, alternating timing, bit identity across the
architecture families, resume and the numerics of `all`; its docstring has the commands. Run it after touching
this package or a function listed in `REVIEWED`.

## Rules

- The shared model files are untouched and the cost audit never imports this package, so published
  proofs and fingerprints stay valid.
- A fast path applies only to CUDA BF16 tensors with the expected layout. Anything else takes the
  original code.
- `install` refuses a torch build or a patched function it was not reviewed against. A new torch
  version needs `tests/test_speedups.py` to pass, since it pins ATen's reduction order. A change to a
  patched function needs the patch reviewed, then its hash updated in `REVIEWED`.
- The kernels are pure elementwise and reduction code without `tl.dot`, and nothing here is tied to one GPU generation.
