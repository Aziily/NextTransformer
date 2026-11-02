"""Time causal attention backends at the training shape on one GPU; research only, no model or data.

Compares PyTorch SDPA (what `nanochat/flash_attention.py` runs without FlashAttention-3) with the
`kernels-community/flash-attn3` hub build for full causal attention and for the SSSL sliding window, and reports
each backend's largest output and dQ difference from an FP32 reference. Run on one idle GPU:

    CUDA_VISIBLE_DEVICES=0 python launch/attention_backends.py --output .agent-work/attention.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def sliding_window(sequence_len: int) -> int:
    """Left window of the short SSSL layers, as `ReferenceLM._window` computes it."""
    return -(-sequence_len // 4 // 128) * 128


def window_mask(torch, sequence_len: int, window: int | None, device):
    """The boolean mask the SDPA fallback builds; None for full causal attention."""
    if window is None or window >= sequence_len - 1:
        return None
    row = torch.arange(sequence_len, device=device)[:, None]
    col = torch.arange(sequence_len, device=device)[None, :]
    return (col <= row) & (row - col <= window)


def _time(torch, fn, iterations: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1e3


def run(batch: int, sequence_len: int, heads: int, head_dim: int, iterations: int, fa3_version: int) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = torch.device("cuda")
    q, k, v = (torch.randn(batch, sequence_len, heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(3))

    def forward_backward(op):
        xs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
        out = op(*xs)
        out.backward(torch.ones_like(out))
        return out.detach(), xs[0].grad

    def largest(a, b):
        return float((a.float() - b.float()).abs().max())

    try:
        from kernels import get_kernel
        fa3 = get_kernel("kernels-community/flash-attn3", version=fa3_version)
    except Exception as exc:  # no network, no build for this torch, or no kernels package
        fa3, fa3_error = None, repr(exc)[:300]
    results = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                   shape=dict(batch=batch, sequence_len=sequence_len, heads=heads, head_dim=head_dim), cases={})
    for name, window in (("full_causal", None), ("sssl_window", sliding_window(sequence_len))):
        mask = window_mask(torch, sequence_len, window, device)

        def sdpa(q, k, v, mask=mask):
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
            return out.transpose(1, 2)

        reference_out, reference_dq = forward_backward(lambda q, k, v: sdpa(q.float(), k.float(), v.float()))
        case = {}
        backends = dict(sdpa=sdpa)
        if fa3 is not None:
            def flash3(q, k, v, window=window):
                out = fa3.flash_attn_func(q, k, v, causal=True, window_size=(-1 if window is None else window, 0))
                return out[0] if isinstance(out, tuple) else out
            backends["fa3_hub"] = flash3
        for label, op in backends.items():
            out, dq = forward_backward(op)
            case[label] = dict(ms=_time(torch, lambda op=op: forward_backward(op), iterations),
                               out_vs_fp32=largest(out, reference_out), dq_vs_fp32=largest(dq, reference_dq))
        if fa3 is None:
            case["fa3_hub"] = dict(error=fa3_error)
        results["cases"][name] = dict(window=window, **case)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--fa3-version", type=int, default=1, help="kernels-community/flash-attn3 major version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    results = run(args.batch, args.sequence_len, args.heads, args.head_dim, args.iterations, args.fa3_version)
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
