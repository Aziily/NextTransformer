"""Full AttnRes with Canon layers at all four positions: `attnres_canon_abcd` with one AttnRes group per layer.

Every block is its own group, so each sublayer reads every earlier block's output separately (the depth sources of
Full AttnRes) instead of group sums. Found with NextT-Gym in arena v1 Stage 2 (attempt att_000050, the best candidate
of that stage); the forward pass is that attempt's, see evidence/checks/arena-v1-stage2-20261102.
"""
from nexttransformer.methods.attnres_canon_abcd import CanonAttnRes, ffn, frozen, init, mixer  # noqa: F401


class FullCanonAttnRes(CanonAttnRes):
    @staticmethod
    def groups(config):
        return config.n_layer


model = FullCanonAttnRes
