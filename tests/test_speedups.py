"""GPU bit-identity of the fused kernels against the eager operations they replace.

Skipped without CUDA. A difference here means a patched run no longer reproduces the reference path.
"""
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="speedups run on CUDA")
if torch.cuda.is_available():
    from nexttransformer.models.backbone import apply_rotary_emb
    from nexttransformer.runtime.speedups import embed, kernels as K


def identical(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    assert torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def grads(function, *inputs, upstream):
    for tensor in inputs: tensor.grad = None
    out = function(*inputs)
    out.backward(upstream)
    result = [out.detach().clone(), *(tensor.grad.clone() for tensor in inputs)]
    for tensor in inputs: tensor.grad = None
    return result


@pytest.mark.parametrize("shape", [(4, 256, 1536), (4, 256, 12, 128), (4, 256, 6144), (4, 256, 768), (4, 256, 256), (4, 256, 8, 64)])
@pytest.mark.parametrize("scale", [1., 30., 1e-3])
def test_rms_norm(shape, scale):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda")*scale).bfloat16().requires_grad_()
    g = torch.randn(shape, device="cuda").bfloat16()
    for a, b in zip(grads(lambda t: F.rms_norm(t, (shape[-1],)), x, upstream=g), grads(K.RMSNorm.apply, x, upstream=g)):
        identical(a, b)


def rotary_table(t, d=128):
    angles = torch.outer(torch.arange(t, device="cuda").float(), 1/(100000**(torch.arange(0, d, 2, device="cuda").float()/d)))
    return angles.cos().bfloat16()[None, :, None, :], angles.sin().bfloat16()[None, :, None, :]


@pytest.mark.parametrize("scale", [1., 5.])
def test_rope_norm_scale(scale):
    torch.manual_seed(0)
    cos, sin = rotary_table(512)
    x = (torch.randn(2, 512, 12, 128, device="cuda")*scale).bfloat16().requires_grad_()
    g = torch.randn_like(x)
    eager = grads(lambda t: F.rms_norm(apply_rotary_emb(t, cos, sin), (128,))*1.2, x, upstream=g)
    for a, b in zip(eager, grads(lambda t: K.RopeNormScale.apply(t, cos, sin, 1.2), x, upstream=g)): identical(a, b)


def test_relu_square():
    torch.manual_seed(0)
    h = torch.randn(4, 256, 6144, device="cuda").bfloat16().requires_grad_()
    g = torch.randn_like(h)
    for a, b in zip(grads(lambda t: F.relu(t).square(), h, upstream=g), grads(K.ReluSquare.apply, h, upstream=g)): identical(a, b)


def test_residual_mix():
    torch.manual_seed(0)
    x, x0 = (torch.randn(4, 256, 1536, device="cuda").bfloat16().requires_grad_() for _ in range(2))
    p, q = torch.tensor([1.1234, .9], device="cuda", requires_grad=True), torch.tensor([.1877, .05], device="cuda", requires_grad=True)
    g = torch.randn_like(x)
    eager = grads(lambda a, b, c, d: a[0]*c+b[0]*d, p, q, x, x0, upstream=g)
    fused = grads(lambda a, b, c, d: K.ResidualMix.apply(b[0], d, a[0], c), p, q, x, x0, upstream=g)
    for a, b in zip(eager, fused): identical(a, b)


def test_value_residual():
    torch.manual_seed(0)
    v, e = (torch.randn(4, 256, 12, 128, device="cuda").bfloat16().requires_grad_() for _ in range(2))
    gate = (3*torch.rand(4, 256, 12, device="cuda")).bfloat16().requires_grad_()
    g = torch.randn_like(v)
    eager = grads(lambda a, b, c: a+b.unsqueeze(-1)*c, v, gate, e, upstream=g)
    for a, b in zip(eager, grads(K.ValueResidual.apply, v, gate, e, upstream=g)): identical(a, b)


def test_soft_cap_exhaustive():
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values.float())]
    capped = torch.empty(values.shape, device="cuda")
    K._cap_fwd[(triton_blocks(values.numel()),)](values, capped, values.numel(), K.INV_CAP, B=1024, num_warps=4)
    identical(15*torch.tanh(values.float()/15), capped)


def triton_blocks(n): return -(-n//1024)


def pair_grads(function, inputs, upstream):
    outs = function(*inputs)
    gs = torch.autograd.grad(outs, inputs, upstream)
    return [o.detach().clone() for o in outs]+list(gs)


@pytest.mark.parametrize("scale", [1., 30., 1e-3])
def test_add_norm(scale):
    torch.manual_seed(0)
    x, d = ((torch.randn(4, 256, 1536, device="cuda")*scale).bfloat16().requires_grad_() for _ in range(2))
    up = [torch.randn_like(x) for _ in range(2)]
    def eager(a, b):
        s = a+b
        return s, F.rms_norm(s, (1536,))
    for p, q in zip(pair_grads(eager, (x, d), up), pair_grads(K.AddNorm.apply, (x, d), up)): identical(p, q)


@pytest.mark.parametrize("same", [False, True])
def test_mix_norm(same):
    torch.manual_seed(0)
    x, x0 = (torch.randn(4, 256, 1536, device="cuda").bfloat16().requires_grad_() for _ in range(2))
    if same: x0 = x
    p, q = torch.tensor([1.1234, .9], device="cuda", requires_grad=True), torch.tensor([.1877, .05], device="cuda", requires_grad=True)
    up = [torch.randn_like(x) for _ in range(2)]
    inputs = (p, q, x) if same else (p, q, x, x0)
    def eager(a, b, c, d=None):
        s = a[0]*c+b[0]*(c if d is None else d)
        return s, F.rms_norm(s, (1536,))
    fused = lambda a, b, c, d=None: K.MixNorm.apply(b[0], c if d is None else d, a[0], c)
    for u, v in zip(pair_grads(eager, inputs, up), pair_grads(fused, inputs, up)): identical(u, v)


@pytest.mark.parametrize("value", [False, True])
def test_attention_layer_gradient(value):
    """The projections add their gradients in the order autograd receives them."""
    from nexttransformer.models.backbone import CausalSelfAttention, GPTConfig
    from nexttransformer.runtime.speedups import patches
    torch.manual_seed(0)
    config = GPTConfig(sequence_len=256, vocab_size=512, n_layer=2, n_head=4, n_kv_head=4, n_embd=512)
    layer = CausalSelfAttention(config, 1 if value else 0).cuda().train()
    assert (layer.ve_gate is not None) == value
    for p in layer.parameters(): torch.nn.init.normal_(p, std=.05)
    x = torch.randn(2, 256, 512, device="cuda").bfloat16().requires_grad_()
    ve = torch.randn(2, 256, 512, device="cuda").bfloat16() if value else None
    cos, sin = rotary_table(256)
    up = torch.randn_like(x)
    def run(forward):
        layer.zero_grad(); x.grad = None
        forward(layer, x, ve, (cos, sin), (-1, 0), None).backward(up)
        return [x.grad.clone()]+[p.grad.clone() for p in layer.parameters()]
    previous = torch.backends.cuda.flash_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    try:
        for a, b in zip(run(CausalSelfAttention.forward), run(patches.attention_forward)): identical(a, b)
    finally:
        torch.backends.cuda.enable_flash_sdp(previous)


def test_mix_norm_shared_x0():
    """Three blocks over one x0, the first with x0 as its input too, as the model uses them."""
    torch.manual_seed(0)
    root = torch.randn(2, 128, 1536, device="cuda").bfloat16().requires_grad_()
    a = torch.tensor([1.1234, .9, 1.3], device="cuda", requires_grad=True)
    b = torch.tensor([.1877, .05, .3], device="cuda", requires_grad=True)
    up = torch.randn_like(root)
    def run(mix):
        x = x0 = root*1
        for i in range(3):
            s, y = mix(i, x, x0)
            x = s+y.square()
        return torch.autograd.grad(x, (root, a, b), up)
    def eager(i, x, x0):
        s = a[i]*x+b[i]*x0
        return s, F.rms_norm(s, (1536,))
    def fused(i, x, x0):
        if not hasattr(x0, "grad_sum"): x0.grad_sum = K.GradSum()
        return K.MixNorm.apply(b[i], x0, a[i], x, x0.grad_sum)
    for p, q in zip(run(eager), run(fused)): identical(p, q)


def test_value_gate_sum_wide_range():
    """ATen's order only shows on values of very different size."""
    torch.manual_seed(0)
    shape = (8, 256, 12, 128)
    scale = torch.exp2(torch.randint(-8, 9, shape, device="cuda").float())
    v, e = ((torch.randn(shape, device="cuda")*scale).bfloat16().requires_grad_() for _ in range(2))
    gate = torch.rand(shape[:3], device="cuda").bfloat16().requires_grad_()
    g = (torch.randn(shape, device="cuda")*scale).bfloat16()
    eager = grads(lambda a, b, c: a+b.unsqueeze(-1)*c, v, gate, e, upstream=g)
    for a, b in zip(eager, grads(K.ValueResidual.apply, v, gate, e, upstream=g)): identical(a, b)


def test_merged_block_is_close():
    """`merge` rounds differently: same gradients up to BF16 noise, and the in-place residual stays legal."""
    from nexttransformer.models.backbone import Block, GPTConfig
    from nexttransformer.runtime.speedups import patches
    torch.manual_seed(0)
    config = GPTConfig(sequence_len=256, vocab_size=512, n_layer=2, n_head=4, n_kv_head=4, n_embd=512)
    block = Block(config, 1).cuda().train()
    for p in block.parameters(): torch.nn.init.normal_(p, std=.05)
    x = torch.randn(2, 256, 512, device="cuda").bfloat16().requires_grad_()
    ve = torch.randn(2, 256, 512, device="cuda").bfloat16()
    cos, sin = rotary_table(256)
    up = torch.randn_like(x)
    def run(forward):
        block.zero_grad(); x.grad = None
        (forward(block, x*1, ve, (cos, sin), (-1, 0), None)).backward(up)
        return [x.grad.float()]+[p.grad.float().clone() for p in block.parameters()]
    reference = run(Block.forward)
    linear = patches.backbone.Linear.forward
    try:
        patches.ORIGINAL["linear"] = linear
        patches.backbone.Linear.forward = patches.linear_forward
        patches.MERGE[0] = True; K.RopeNormScale.exact = False; K.AddNorm.keep_output = True
        patches.UPDATE[0] += 1
        merged = run(patches.block_forward)
    finally:
        patches.backbone.Linear.forward = linear
        patches.ORIGINAL.pop("linear")
        patches.MERGE[0] = False; K.RopeNormScale.exact = True; K.AddNorm.keep_output = False
    for a, b in zip(reference, merged):
        assert (a-b).norm()/a.norm().clamp_min(1e-12) < 3e-2


def test_weight_cast_reused():
    torch.manual_seed(0)
    w = torch.randn(256, 512, device="cuda", requires_grad=True)
    xs = [torch.randn(64, 512, device="cuda").bfloat16() for _ in range(3)]
    def run(weight):
        w.grad = None
        for x in xs:  # a shared weight used twice in one backward, then accumulated
            (F.linear(F.linear(x, weight()), weight().T)).float().pow(2).sum().backward()
        return w.grad.clone()
    cast = w.detach().bfloat16()
    identical(run(lambda: w.to(torch.bfloat16)), run(lambda: K.WeightCast.apply(w, cast)))


def test_soft_cap_monotone():
    """_ce_fwd takes the capped maximum as the cap of the maximum."""
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    v = bits.view(torch.bfloat16).float()
    v = v[v.isfinite()].sort().values
    c = 15*torch.tanh(v/15)
    assert (c[1:] >= c[:-1]).all()


@pytest.mark.parametrize("mean", [False, True])
def test_capped_cross_entropy(mean):
    torch.manual_seed(0)
    n, v = 1024, 32768
    logits = (torch.randn(n, v, device="cuda")*12).bfloat16().requires_grad_()
    targets = torch.randint(0, v, (n,), device="cuda"); targets[5] = -1
    reduction = "mean" if mean else "none"
    upstream = torch.rand((), device="cuda") if mean else torch.rand(n, device="cuda")/n
    eager = grads(lambda t: F.cross_entropy(15*torch.tanh(t.float()/15), targets, ignore_index=-1, reduction=reduction), logits, upstream=upstream)
    for a, b in zip(eager, grads(lambda t: K.CappedCrossEntropy.apply(t, targets, mean), logits, upstream=upstream)): identical(a, b)


OWN_LOSS = """
from dataclasses import replace
import torch, torch.nn.functional as F
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import ReferenceLM, build_model
from nexttransformer.runtime import speedups

class OwnLoss(ReferenceLM):
    @staticmethod
    def _token_loss(logits, targets):
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction="none").reshape_as(targets)

config = replace(debug_config("dense_full", width=256, sequence_len=256), heads=2, vocab_size=32768)
models = []
for cls in (ReferenceLM, OwnLoss):
    model = build_model(config, "cuda", seed=0)
    with torch.no_grad(): model.lm_head.weight.mul_(400)  # logits well inside the soft cap's range
    model.__class__ = cls
    models.append(model.train())
torch.manual_seed(0)
idx = torch.randint(0, config.vocab_size, (8, config.sequence_len), device="cuda")
targets = torch.roll(idx, -1, 1)
before = [float(m(idx, targets).detach()) for m in models]
speedups.install("readout")
after = [float(m(idx, targets).detach()) for m in models]
assert before == after and speedups.COUNTS == {"readout": 1}, (before, after, speedups.COUNTS)
"""


def test_readout_keeps_a_model_class_with_its_own_token_loss():
    """`install` patches the process, so this runs in its own interpreter."""
    out = subprocess.run([sys.executable, "-c", OWN_LOSS], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]


@pytest.mark.parametrize("kind", ["zipf", "uniform", "one"])
def test_embedding_gradient(kind):
    torch.manual_seed(0)
    v, d, n = 32768, 1536, 8192
    probabilities = 1/torch.arange(1, v+1, device="cuda", dtype=torch.float64)
    indices = {"zipf": lambda: torch.multinomial(probabilities, n, replacement=True), "uniform": lambda: torch.randint(0, v, (n,), device="cuda"),
               "one": lambda: torch.full((n,), 7, device="cuda")}[kind]().view(4, -1)
    g = torch.randn(*indices.shape, d, device="cuda").bfloat16()
    identical(torch.ops.aten.embedding_dense_backward(g, indices, v, -1, False), embed.dense_gradient(g, embed.Plan(indices), v))
