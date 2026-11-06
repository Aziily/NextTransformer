"""The AttnRes-Canon plugin (attnres_full_canon_abcd) and the Canon AttnRes code it reuses.

The shared per-method tests cover forward and backward, causality, gradients and the checkpoint round trip. Here:
with Canon weights at zero (their initialization) the plugin is exactly Full AttnRes (`attnres` with one group per
layer) with the same weights, and its group layout follows `attnres`.
"""
from dataclasses import replace

import torch

from nexttransformer.methods.attnres_canon_abcd import Canon, group_starts
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model


def test_zero_canon_is_full_attnres():
    """Three layers, one AttnRes group per layer."""
    reference = build_model(replace(debug_config("attnres"), layers=3, attnres_groups=3), "cpu", seed=3)
    plugin = build_model(replace(debug_config("attnres_full_canon_abcd"), layers=3), "cpu", seed=11)
    missing, unexpected = plugin.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and missing and all("canon" in name for name in missing)
    assert all(torch.count_nonzero(m.weight) == 0 for m in plugin.modules() if isinstance(m, Canon))
    assert plugin.group_starts == reference.config.attnres_group_starts
    x = torch.randint(0, plugin.config.vocab_size, (2, plugin.config.sequence_len))
    for model in (reference, plugin): model.eval()
    assert torch.equal(reference(x), plugin(x))


def test_group_layout_matches_attnres():
    for layers in (3, 7, 23):
        for groups in range(1, layers + 1):
            c = replace(candidate_config("attnres"), layers=layers, attnres_groups=groups)
            assert group_starts(layers, groups) == c.attnres_group_starts
    assert len(build_model(candidate_config("attnres_full_canon_abcd"), "meta").group_starts) == 23
