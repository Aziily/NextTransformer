"""Block AttnRes with Canon layers at all four positions (Canon-A/B/C/D).

The Block AttnRes trunk of the built-in `attnres` (Attention Residuals: every sublayer reads a softmax mixture of the
embedding, the completed group sums and the current partial sum, through one pseudo-query per sublayer), with a
Canon layer (a causal depthwise convolution over the last four positions, added to its input) at four places:
- A and C: on the input of every attention and MLP sublayer, after the depth read;
- B: on q, k and v, before rotary embedding and QK-norm;
- D: on the MLP hidden activations, before the squared ReLU.

Canon layers: Allen-Zhu, "Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers"
(2025). Their weights start at zero, so every Canon layer starts as the identity and the model starts as `attnres`.
Option `groups` is the number of AttnRes groups. Found with NextT-Gym in arena v1 Stage 2 (attempt att_000043); the
forward pass is that attempt's, see evidence/checks/arena-v1-stage2-20261102.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import depth_attention


class Canon(nn.Module):
    """Causal depthwise convolution over the last `size` positions, added to its input; starts as the identity."""

    def __init__(self, width, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, size))
        self.size = size

    def forward(self, x):
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size - 1, 0)), self.weight.to(x.dtype), groups=x.shape[-1])
        return x + y.transpose(1, 2)


class CanonAttention(CausalSelfAttention):
    """The shell's attention with a Canon layer on each of q, k and v (Canon-B)."""

    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.canon_qkv = nn.ModuleList(Canon(self.n_embd) for _ in range(3))

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("training and full-sequence evaluation only")
        B, T, C = x.size()
        q, k, v = (layer(t) for layer, t in zip(self.canon_qkv, (self.c_q(x), self.c_k(x), self.c_v(x))))
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class CanonMLP(nn.Module):
    """The shell's squared-ReLU MLP with a Canon layer on its hidden activations, before the nonlinearity (Canon-D)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.width, config.mlp_width, bias=False)
        self.c_proj = Linear(config.mlp_width, config.width, bias=False)
        self.canon = Canon(config.mlp_width)

    def forward(self, x):
        return self.c_proj(F.relu(self.canon(self.c_fc(x))).square())


def _drawn_by_init_weights():
    """Stands in for nn.Embedding.reset_parameters, see CanonAttnRes."""


def group_starts(layers, groups):
    """Physical layer index of the first block of each AttnRes group: balanced, larger groups first (as `attnres`)."""
    whole, extra = divmod(layers, groups)
    sizes = (whole + 1,) * extra + (whole,) * (groups - extra)
    starts = [0]
    for size in sizes[:-1]: starts.append(starts[-1] + size)
    return tuple(starts)


class CanonAttnRes(ReferenceLM):
    """Block AttnRes (the trunk of the built-in `attnres`) with Canon-A/C on every sublayer input."""

    @staticmethod
    def groups(config):
        return config.option("groups")

    def __init__(self, config):
        super().__init__(config)
        self.group_starts = group_starts(config.n_layer, self.groups(config))
        # One pseudo-query per attention and FFN sublayer, plus one for the final readout (as `attnres`).
        self.depth_queries = nn.Parameter(torch.empty(2 * config.n_layer + 1, config.width))
        self.canon = nn.ModuleList(Canon(config.width) for _ in range(2 * config.n_layer))
        # build_model resets plugin modules that define reset_parameters, the embeddings among them; init_weights
        # draws the embeddings anyway. Skipping that extra draw keeps the random sequence of the built-in build, so
        # seed 42 gives exactly the initial weights the Stage 2 attempts trained from.
        for module in self.modules():
            if isinstance(module, nn.Embedding): module.reset_parameters = _drawn_by_init_weights

    def _ordinary_hidden(self, idx):
        """The Block AttnRes trunk of ReferenceLM._residual_hidden, with a Canon layer on each sublayer input."""
        c = self.config
        x = self._embed_reference(idx)
        starts = set(self.group_starts)
        blocks, keys, partial = [], [], x
        for k in range(2 * c.n_layer):
            key = norm(partial)
            h = depth_attention(self.depth_queries[k], blocks + [partial], keys + [key])
            self._record("depth_read", sublayer=k, sources=len(blocks) + 1, tokens=idx.numel())
            if k % 2 == 0 and k // 2 in starts:
                blocks.append(partial); keys.append(key); partial = None
            h = self.canon[k](h)
            out = self._attention_branch(h, idx, k // 2) if k % 2 == 0 else self.transformer.h[k // 2].mlp(norm(h))
            partial = out if partial is None else partial + out
        h = depth_attention(self.depth_queries[-1], blocks + [partial], keys + [norm(partial)])
        self._record("depth_read", sublayer=2 * c.n_layer, sources=len(blocks) + 1, tokens=idx.numel())
        return h, None      # no backout, as the residual family


def mixer(config, layer):
    return CanonAttention(config, layer)


def ffn(config, layer):
    return CanonMLP(config)


model = CanonAttnRes
# The residual family's trunk does not use the shell's residual and x0 scalars or the backout.
frozen = ("resid_lambdas", "x0_lambdas", "backout_lambda")


@torch.no_grad()
def init(model):
    """Canon layers start as the identity; ReferenceLM.init_weights already zeroes `depth_queries` (uniform reads)."""
    for module in model.modules():
        if isinstance(module, Canon): nn.init.zeros_(module.weight)


def check(config):
    groups = config.option("groups")
    if not (isinstance(groups, int) and not isinstance(groups, bool) and 1 <= groups <= config.n_layer):
        raise ValueError(f"groups must be an integer from 1 to the layer count ({config.n_layer})")
