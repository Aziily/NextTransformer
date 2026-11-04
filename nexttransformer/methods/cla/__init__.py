"""Cross-Layer Attention (CLA): a layer attends with the keys and values of the layer below it.

Brandon et al., "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention" (NeurIPS 2024), CLA2: two
adjacent layers share one set of key/value activations. Only the lower layer of a pair projects keys and values and
keeps them in the decoding cache; queries, attention and output projections stay per layer.

Adaptation to the shell: the shell adds value embeddings on alternating layers, ending at the last one. Pairs follow
them: every layer without value embeddings reuses the keys and values of the layer below it, which has them, so no
value embedding is dropped. The first layer (when it has none) and the last layer keep their own keys and values. A
reusing layer's `c_k` and `c_v` are its source layer's modules, so every block keeps the shell's linear layers and every
parameter still receives a gradient.

With `share_kv: false` every layer keeps its own keys and values: the full-attention shell on the same code path, the
control for decoding measurements.

`heads_per_kv` groups query heads onto fewer key/value heads (GQA, Ainslie et al. 2023), as the paper combines CLA with
MQA and GQA: 1 keeps one key/value head per query head, 4 gives the 12 query heads 3 key/value heads, 12 is MQA. As in
the native GPT with fewer key/value heads, value embeddings and their gates take the key/value width.
"""
import torch
from torch import nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, has_ve, norm
from nexttransformer.models.lm import ReferenceLM


def kv_heads(config):
    """Key/value heads per layer: the query heads divided by the query heads that share one key/value head."""
    return config.heads // config.option("heads_per_kv")


def sources(config):
    """For every layer, the layer whose keys and values it attends with: itself, or the layer below it."""
    n = config.n_layer
    share = config.option("share_kv")
    return [i - 1 if share and i > 0 and not has_ve(i, n) else i for i in range(n)]


class Link:
    """Hands the keys and values of a source layer to the layer above it within one forward pass."""
    kv = None


class CrossLayerAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        if kv_heads(config) != self.n_kv_head:
            self.n_kv_head = kv_heads(config)
            self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            if self.ve_gate is not None:
                self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        self.source = sources(config)[layer]
        self.link = None  # set by the model on a source layer and on the layer that reuses it
        self.slot = None  # decoding-cache slot of the source layer

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        cos, sin = cos_sin
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
        owner = self.source == self.layer_idx
        if owner:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
            if ve is not None:
                ve = ve.view(B, T, self.n_kv_head, self.head_dim)
                gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
            k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
            if self.link is not None:
                self.link.kv = k, v
        else:
            if self.link.kv is None:
                raise RuntimeError(f"layer {self.layer_idx} runs before its source layer {self.source}")
            (k, v), self.link.kv = self.link.kv, None
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            y = kv_cache.attend(self.slot, q, k, v, window_size, write=owner)
        return self.c_proj(y.contiguous().view(B, T, -1))


class CrossLayerCache:
    """Decoding cache with one key/value slot per source layer; reusing layers read their source's slot."""

    def __init__(self, model, batch, length, device=None, dtype=COMPUTE_DTYPE):
        c = model.config
        if length > model.cos.shape[1]:
            raise ValueError("cache longer than the rotary table")
        shape = (model.cache_slots, batch, length, kv_heads(c), c.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length, self.position, self.embedding = length, 0, None

    @property
    def nbytes(self):
        return self.k.nbytes + self.v.nbytes

    def attend(self, slot, q, k, v, window_size, write):
        start, stop = self.position, self.position + q.shape[1]
        if write:
            self.k[slot, :, start:stop] = k
            self.v[slot, :, start:stop] = v
        # Causal attention of the new queries over every cached key, aligned to the last key.
        return flash_attn.flash_attn_func(q, self.k[slot, :, :stop], self.v[slot, :, :stop], causal=True,
                                          window_size=window_size)


class CrossLayerLM(ReferenceLM):
    """The shell with cross-layer key/value sharing; training runs the shell's forward unchanged."""

    def __init__(self, config):
        super().__init__(config)
        width = kv_heads(config) * config.head_dim
        if width != config.width:
            self.value_embeds = nn.ModuleDict({name: nn.Embedding(table.num_embeddings, width)
                                               for name, table in self.value_embeds.items()})
        layers = [block.attn for block in self.transformer.h]
        owners = sorted(set(attn.source for attn in layers))
        self.cache_slots = len(owners)
        for attn in layers:
            attn.slot = owners.index(attn.source)
            if attn.source != attn.layer_idx:
                source = layers[attn.source]
                attn.c_k, attn.c_v = source.c_k, source.c_v
                attn.link = source.link = Link()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        if kv_cache is None:
            return super().forward(idx, targets, None, loss_reduction)
        if targets is not None:
            raise ValueError("cached decoding returns logits only")
        return self._cached_logits(idx, kv_cache)

    def _cached_logits(self, idx, cache):
        """Logits of the new tokens `idx` after the cached prefix; the same function as the full forward."""
        c = self.config
        t, start = idx.shape[1], cache.position
        if start + t > cache.length:
            raise ValueError("the cache is full")
        self.last_trace = []
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        # Smear mixes each token's embedding with the previous token's, as the training forward does.
        if cache.embedding is None:
            head, rest, before = x[:, :1], x[:, 1:], x[:, :-1]
        else:
            head, rest, before = x[:, :0], x, torch.cat((cache.embedding, x[:, :-1]), 1)
        cache.embedding = x[:, -1:]
        if rest.shape[1]:
            gate = self.smear_lambda.to(x.dtype) * self.smear_gate(rest[..., :24]).sigmoid()
            x = torch.cat((head, rest + gate * before), 1)
        cos_sin = self.cos[:, start:start + t], self.sin[:, start:start + t]
        x0, backout, n = x, None, c.n_layer
        for i, block in enumerate(self.transformer.h):
            mixed = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(mixed, ve, cos_sin, self._window(i, n), cache)
            if i == n // 2:
                backout = x
        cache.position = start + t
        return self._readout(x, backout)


def check(config):
    groups = config.option("heads_per_kv")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups < 1 or config.heads % groups:
        raise ValueError(f"heads_per_kv must be a positive divisor of the {config.heads} query heads")
    if not isinstance(config.option("share_kv"), bool):
        raise ValueError("share_kv must be true or false")


def mixer(config, layer):
    return CrossLayerAttention(config, layer)


model = CrossLayerLM
