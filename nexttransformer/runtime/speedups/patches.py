"""The patched methods. Each keeps its original as the fallback for everything outside the fast path."""
import sys

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn
from nexttransformer.models import backbone, lm, sparse
from nexttransformer.models.config import HYBRID, LINEAR, MLA, SPARSE
from nexttransformer.models.registry import PLUGIN_METHODS
from nexttransformer.runtime.speedups import _count
from nexttransformer.runtime.speedups.embed import Lookup
from nexttransformer.runtime.speedups.kernels import (AddNorm, CappedCrossEntropy, GradSum, MixNorm, Projections, ReluSquare, ResidualLinear,
                                                      ResidualMix, RMSNorm, RopeNormScale, ValueResidual, WeightCast, WideLinear)

MAX_ROW = 8192


def bf16(x):
    return x.is_cuda and x.dtype == torch.bfloat16


def fast_norm(x):
    n = x.shape[-1]
    if bf16(x) and n % 4 == 0 and n <= MAX_ROW and x.is_contiguous() and x.data_ptr() % 16 == 0:
        _count("norm")
        return RMSNorm.apply(x)
    return ORIGINAL["norm"](x)


def wide_row(x):
    """Rows the fused add and norm kernels take; narrower ones are not worth another specialization."""
    n = x.shape[-1]
    return bf16(x) and n % 4 == 0 and 128 < n <= MAX_ROW and x.is_contiguous() and x.data_ptr() % 16 == 0


def add_norm(x, d):
    """x+d, norm(x+d)"""
    if wide_row(x) and bf16(d) and d.shape == x.shape and d.is_contiguous() and d.data_ptr() % 16 == 0:
        _count("addnorm")
        return AddNorm.apply(x, d)
    s = x+d
    return s, backbone.norm(s)


def block_forward(self, x, ve, cos_sin, window_size, kv_cache, normed=None):
    if kv_cache is not None: return ORIGINAL["block"](self, x, ve, cos_sin, window_size, kv_cache)
    s, y = add_norm(x, self.attn(backbone.norm(x) if normed is None else normed, ve, cos_sin, window_size, kv_cache))
    if MERGE[0] and type(self.mlp) in (backbone.MLP, lm.FlexibleMLP) and wide_row(y) and torch.is_grad_enabled():
        h = relu_square(self.mlp.c_fc(y))
        return ResidualLinear.apply(s, h, self.mlp.c_proj.weight, update_cast(self.mlp.c_proj, h))
    return s+self.mlp(y)


UPDATE = [0]  # counts optimizer updates; casts of an earlier update are stale


def set_step(self, step):
    UPDATE[0] += 1
    return ORIGINAL["set_step"](self, step)


def update_cast(linear, x):
    """The BF16 copy of an FP32 weight, made once per update."""
    w = linear.weight
    key = UPDATE[0], w._version
    cached = linear.__dict__.get("_update_cast")
    if cached is None or cached[0] != key:
        cached = linear.__dict__["_update_cast"] = key, w.detach().to(x.dtype)
    _count("weights")
    return cached[1]


def cast_weight(linear, x):
    """linear.weight.to(x.dtype), cast once per update."""
    if "linear" not in ORIGINAL or linear.weight.dtype != torch.float32: return linear.weight.to(x.dtype)
    return WeightCast.apply(linear.weight, update_cast(linear, x))


def wide_weight(w):
    """Shapes the FP32-output GEMM takes; the gates' few rows and columns are not worth it."""
    return w.dtype == torch.float32 and w.shape[0] % 64 == 0 and w.shape[1] % 64 == 0


def linear_forward(self, x):
    if not (self.training and torch.is_grad_enabled() and self.weight.requires_grad and bf16(x)):
        return ORIGINAL["linear"](self, x)
    if MERGE[0] and wide_weight(self.weight): return WideLinear.apply(x, self.weight, update_cast(self, x))
    return F.linear(x, cast_weight(self, x))


MERGE = [False]


def projections(self, x, channels, sole_reader=True):
    """q, k, v and the value gate's logits (None without `channels`), all from x.

    `sole_reader`: nothing else in the layer reads x. Another reader's gradient would arrive between
    these paths in eager, so they cannot be added ahead of it.
    """
    gate = self.ve_gate if channels else None
    linears = (self.c_q, self.c_k, self.c_v)
    if not (sole_reader and self.training and torch.is_grad_enabled() and bf16(x) and x.is_contiguous()):
        return (*(f(x) for f in linears), gate(x[..., :channels]) if channels else None)
    _count("projections")
    if MERGE[0] and all(wide_weight(f.weight) for f in linears):
        casts = (*(update_cast(f, x) for f in linears), update_cast(gate, x) if channels else None)
        out = Projections.apply(x, *(f.weight for f in linears), gate.weight if channels else None, channels, casts)
    else:
        out = Projections.apply(x, *(cast_weight(f, x) for f in linears), cast_weight(gate, x) if channels else None, channels, None)
    return out if channels else (*out, None)


def rope_norm_scale(z, cos, sin):
    """norm(apply_rotary_emb(z, cos, sin))*1.2"""
    if bf16(z) and z.ndim == 4 and z.shape[-1] == 128 and z.is_contiguous() and cos.is_contiguous() and sin.is_contiguous():
        _count("attention")
        return RopeNormScale.apply(z, cos, sin, 1.2)
    return backbone.norm(backbone.apply_rotary_emb(z, cos, sin))*1.2


def value_residual(v, gate, e):
    """v+gate.unsqueeze(-1)*e"""
    if bf16(v) and bf16(e) and gate.dtype == v.dtype and v.is_contiguous() and e.is_contiguous() and gate.is_contiguous():
        _count("value")
        return ValueResidual.apply(v, gate, e)
    return v+gate.unsqueeze(-1)*e


def attention_forward(self, x, ve, cos_sin, window_size, kv_cache):
    if kv_cache is not None: return ORIGINAL["attention"](self, x, ve, cos_sin, window_size, kv_cache)
    B, T, C = x.size()
    q, k, v, logits = projections(self, x, self.ve_gate_channels if ve is not None else 0)
    q = q.view(B, T, self.n_head, self.head_dim)
    k = k.view(B, T, self.n_kv_head, self.head_dim)
    v = v.view(B, T, self.n_kv_head, self.head_dim)
    if ve is not None:
        ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        gate = 3 * torch.sigmoid(logits)
        v = value_residual(v, gate, ve)
    cos, sin = cos_sin
    q, k = rope_norm_scale(q, cos, sin), rope_norm_scale(k, cos, sin)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    return self.c_proj(y.contiguous().view(B, T, -1))


def sparse_projections(self, x, ve, cos_sin):
    b, t, _ = x.shape
    q, k, v, logits = projections(self, x, 12 if ve is not None else 0, self.kind != "nsa")  # NSA's branch gate reads x too
    q, k, v = (z.view(b, t, self.n_head, self.head_dim) for z in (q, k, v))
    if ve is not None:
        gate = 3*torch.sigmoid(logits)
        v = value_residual(v, gate, ve.view_as(v))
    q, k = (rope_norm_scale(z, *cos_sin) for z in (q, k))
    return tuple(z.transpose(1, 2) for z in (q, k, v))


def embedding_forward(self, indices):
    """ATen's dense gradient switches algorithm at 3,072 rows; only its sorted path is reproduced."""
    w = self.weight
    if (self.training and torch.is_grad_enabled() and w.requires_grad and bf16(w) and indices.dtype == torch.int64
            and indices.numel() > 3072 and self.padding_idx is None and self.max_norm is None
            and not self.scale_grad_by_freq and not self.sparse):
        _count("embedding")
        return Lookup.apply(w, indices)
    return ORIGINAL["embedding"](self, indices)


def relu_square(h):
    if bf16(h) and h.is_contiguous():
        _count("mlp")
        return ReluSquare.apply(h)
    return F.relu(h).square()


def mlp_forward(self, x):
    return self.c_proj(relu_square(self.c_fc(x)))


def residual_mix(self, physical, x, x0):
    """resid_lambdas[physical]*x+x0_lambdas[physical]*x0"""
    a, b = self.resid_lambdas[physical], self.x0_lambdas[physical]
    if bf16(x) and x.is_contiguous() and x0.is_contiguous() and x.shape == x0.shape and torch.is_grad_enabled():
        _count("residual")
        return ResidualMix.apply(b, x0, a, x)
    return a*x+b*x0


def mixed_block(self, layer, physical, x, x0, ve, cos_sin, window):
    """layer(residual_mix(x, x0), ...) with the mix and the layer's first norm in one pass."""
    if (type(layer) is backbone.Block and "block" in ORIGINAL and wide_row(x) and x0.is_contiguous() and x.shape == x0.shape
            and x0.data_ptr() % 16 == 0 and torch.is_grad_enabled()):
        _count("addnorm")
        if not hasattr(x0, "grad_sum"): x0.grad_sum = GradSum()
        s, y = MixNorm.apply(self.x0_lambdas[physical], x0, self.resid_lambdas[physical], x, x0.grad_sum)
        return block_forward(layer, s, ve, cos_sin, window, None, y)
    return layer(residual_mix(self, physical, x, x0), ve, cos_sin, window, None)


def block(self, x, x0, idx, physical, virtual, total, alive=None):
    c = self.config
    ve = self.value_embeds[str(physical)](idx).to(x.dtype) if str(physical) in self.value_embeds else None
    window = self._window(virtual, total)
    cos_sin = self.cos[:, :idx.shape[1]], self.sin[:, :idx.shape[1]]
    layer = self.transformer.h[physical]
    out = layer(x, ve, cos_sin, window, None) if c.method == "deeploop" else mixed_block(self, layer, physical, x, x0, ve, cos_sin, window)
    if alive is not None: out = torch.where(alive[..., None], out, x)
    kind = c.layer_mixers[physical] if c.method in HYBRID else c.method
    self._record("block", physical=physical, virtual=virtual, window=window[0],
                 tokens=idx.numel(), sequence_len=idx.shape[1], ve=ve is not None,
                 mixer=kind if kind in LINEAR | SPARSE | MLA | PLUGIN_METHODS else "softmax")
    return out


def lm_forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
    """Only a training forward that goes on to the loss may leave its logits uncapped."""
    self._fused_head = targets is not None and self.training and torch.is_grad_enabled() and kv_cache is None
    try: return ORIGINAL["forward"](self, idx, targets, kv_cache, loss_reduction)
    finally: self._fused_head = False


def readout(self, x, backout):
    """Uncapped logits go only to the shared `_token_loss`; a class with its own receives the capped ones it expects."""
    if not (getattr(self, "_fused_head", False) and bf16(x) and type(self)._token_loss is token_loss):
        return ORIGINAL["readout"](self, x, backout)
    if backout is not None: x = x-self.backout_lambda.to(x.dtype)*backout
    logits = self.lm_head(backbone.norm(x))[..., :self.config.vocab_size]
    self._record("head", tokens=x.shape[0]*x.shape[1])
    if not logits.is_contiguous(): return 15*torch.tanh(logits.float()/15)
    logits.awaiting_cap = True
    return logits


def token_loss(logits, targets):
    if getattr(logits, "awaiting_cap", False):
        _count("readout")
        return CappedCrossEntropy.apply(logits, targets)
    return ORIGINAL["token_loss"](logits, targets)


def gpt_forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
    """The Dense anchor's own forward; only the training loss path differs from the original."""
    if targets is None or kv_cache is not None or loss_reduction != "mean" or not self.training or not torch.is_grad_enabled():
        return ORIGINAL["gpt_forward"](self, idx, targets, kv_cache, loss_reduction)
    B, T = idx.size()
    cos_sin = self.cos[:, :T], self.sin[:, :T]
    x = self.transformer.wte(idx)
    x = x.to(COMPUTE_DTYPE)
    x = backbone.norm(x)
    assert T > 1, "Training forward pass should have T > 1"
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0 = x
    n_layer = self.config.n_layer
    backout_layer = n_layer // 2
    x_backout = None
    for i, layer in enumerate(self.transformer.h):
        ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        x = mixed_block(self, layer, i, x, x0, ve, cos_sin, self.window_sizes[i])
        if i == backout_layer:
            x_backout = x
    if x_backout is not None:
        x = x - self.backout_lambda.to(x.dtype) * x_backout
    x = backbone.norm(x)
    logits = self.lm_head(x)[..., :self.config.vocab_size]
    if not (bf16(logits) and logits.is_contiguous()):
        logits = 15 * torch.tanh(logits.float() / 15)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="mean")
    _count("readout")
    return CappedCrossEntropy.apply(logits.view(-1, logits.size(-1)), targets.view(-1), True)


ORIGINAL = {}


def apply(names):
    if "cudnn" in names and torch.backends.cuda.flash_sdp_enabled():  # deterministic runs keep their backend
        from torch.nn.attention import SDPBackend
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch._C._set_sdp_priority_order([int(b) for b in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                                                           SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)])
    if "norm" in names:
        ORIGINAL["norm"] = backbone.norm
        for name, module in list(sys.modules.items()):
            if name.startswith("nexttransformer.") and getattr(module, "norm", None) is ORIGINAL["norm"]:
                module.norm = fast_norm
    if "attention" in names:
        ORIGINAL["attention"] = backbone.CausalSelfAttention.forward
        backbone.CausalSelfAttention.forward = attention_forward
        sparse.SparseMixer._projections = sparse_projections
    if "embedding" in names:
        ORIGINAL["embedding"] = torch.nn.Embedding.forward
        torch.nn.Embedding.forward = embedding_forward
    if "merge" in names:
        MERGE[0] = True
        MixNorm.partial_sums = True
        RopeNormScale.exact = False
        AddNorm.keep_output = True
    if "weights" in names:
        ORIGINAL["linear"] = backbone.Linear.forward
        ORIGINAL["set_step"] = lm.ReferenceLM.set_step
        backbone.Linear.forward = linear_forward
        lm.ReferenceLM.set_step = set_step
    if "addnorm" in names:
        ORIGINAL["block"] = backbone.Block.forward
        backbone.Block.forward = block_forward
    if "mlp" in names:
        backbone.MLP.forward = mlp_forward
        lm.FlexibleMLP.forward = mlp_forward
    if "residual" in names:
        lm.ReferenceLM._block = block
    else:
        globals()["residual_mix"] = lambda self, physical, x, x0: self.resid_lambdas[physical]*x+self.x0_lambdas[physical]*x0
    if "readout" in names:
        ORIGINAL["readout"] = lm.ReferenceLM._readout
        ORIGINAL["token_loss"] = lm.ReferenceLM._token_loss
        ORIGINAL["forward"] = lm.ReferenceLM.forward
        ORIGINAL["gpt_forward"] = backbone.GPT.forward
        lm.ReferenceLM._readout = readout
        lm.ReferenceLM._token_loss = staticmethod(token_loss)
        lm.ReferenceLM.forward = lm_forward
        backbone.GPT.forward = gpt_forward
