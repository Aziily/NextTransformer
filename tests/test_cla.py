"""Cross-Layer Attention plugin: the sharing layout, the shell contract, and cached decoding."""
from dataclasses import replace

import pytest
import torch

from nexttransformer.methods.cla import CrossLayerCache, kv_heads, sources
from nexttransformer.models.backbone import GPT, GPTConfig, has_ve
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.compute import audit_graph
from nexttransformer.training.optimizer import parameter_groups


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def model_of(layers=6, share=True, groups=1):
    config = replace(debug_config("cla"), layers=layers, options={"share_kv": share, "heads_per_kv": groups})
    model = build_model(config)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.05)
            block.mlp.c_proj.weight.normal_(std=.05)
        model.smear_lambda.fill_(.3)
    return model


@pytest.mark.parametrize("layers", [23, 24])
def test_pairs_follow_the_value_embedding_layers(layers):
    config = replace(candidate_config("cla"), layers=layers)
    owners = sources(config)
    assert owners[-1] == layers - 1 and owners[0] == 0
    for layer, source in enumerate(owners):
        assert source in (layer, layer - 1)
        assert sources(config)[source] == source  # a source layer projects its own keys and values
        if has_ve(layer, layers):
            assert source == layer  # no value embedding is dropped
    assert sources(replace(config, options={"share_kv": False, "heads_per_kv": 1})) == list(range(layers))


def test_reusing_layers_hold_their_source_projections_and_every_parameter_trains():
    model = model_of()
    layers = [block.attn for block in model.transformer.h]
    assert [attn.source for attn in layers] == [0, 1, 1, 3, 3, 5] and model.cache_slots == 4
    assert layers[2].c_k is layers[1].c_k and layers[2].c_v is layers[1].c_v
    names = [name for name, _ in model.named_parameters()]
    assert "transformer.h.2.attn.c_k.weight" not in names and len(names) == len(set(names))
    groups = parameter_groups(model)
    owned = [id(p) for group in groups for p in group["params"]]
    assert sorted(owned) == sorted(id(p) for p in model.parameters() if p.requires_grad)
    x = torch.randint(model.config.vocab_size, (2, model.config.sequence_len))
    model(x, x).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # The shared projections learn from both layers that use them.
    assert layers[1].c_k.weight.grad.abs().sum() > 0 and layers[2].c_q.weight.grad.abs().sum() > 0


def test_without_sharing_the_plugin_is_the_dense_baseline():
    """share_kv false computes exactly the full-attention Dense baseline, so decoding comparisons are like for like."""
    dense = build_model(replace(debug_config("dense_full"), layers=6))
    control = build_model(replace(debug_config("cla"), layers=6, options={"share_kv": False, "heads_per_kv": 1}))
    with torch.no_grad():
        for block in dense.transformer.h:
            block.attn.c_proj.weight.normal_(std=.05)
            block.mlp.c_proj.weight.normal_(std=.05)
    control.load_state_dict(dense.state_dict())
    x = torch.randint(dense.config.vocab_size, (2, dense.config.sequence_len))
    torch.testing.assert_close(control(x, x), dense(x, x), rtol=0, atol=0)


def test_sharing_changes_the_function_and_the_control_does_not_share():
    torch.manual_seed(0)
    shared = model_of(share=True)
    torch.manual_seed(0)
    control = model_of(share=False)
    assert control.cache_slots == control.config.n_layer
    x = torch.randint(shared.config.vocab_size, (1, shared.config.sequence_len))
    with torch.no_grad():
        assert not torch.allclose(shared(x), control(x))


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("prefill", [1, 5, 12])
def test_cached_decoding_matches_the_full_forward(share, groups, prefill):
    model = model_of(share=share, groups=groups).eval()
    t = model.config.sequence_len
    x = torch.randint(model.config.vocab_size, (2, t))
    with torch.no_grad():
        full = model(x)
        cache = CrossLayerCache(model, 2, t)
        parts = [model(x[:, :prefill], kv_cache=cache)]
        parts += [model(x[:, i:i + 1], kv_cache=cache) for i in range(prefill, t)]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-5)
    assert cache.position == t
    with pytest.raises(ValueError, match="full"):
        model(x[:, :1], kv_cache=cache)


def test_the_cache_keeps_one_slot_per_source_layer_and_its_key_value_heads():
    shared, control = model_of(share=True), model_of(share=False)
    small, large = CrossLayerCache(shared, 3, 12), CrossLayerCache(control, 3, 12)
    assert small.k.shape == (4, 3, 12, 2, 24) and large.k.shape[0] == 6
    assert small.nbytes * 6 == large.nbytes * 4
    grouped = CrossLayerCache(model_of(share=True, groups=2), 3, 12)
    assert grouped.k.shape == (4, 3, 12, 1, 24) and grouped.nbytes * 2 == small.nbytes


def test_grouped_heads_without_sharing_are_the_native_gqa_attention():
    """heads_per_kv groups query heads exactly as the native GPT does with fewer key/value heads."""
    config = replace(debug_config("cla"), layers=6, options={"share_kv": False, "heads_per_kv": 2})
    assert kv_heads(config) == 1
    plugin = build_model(config)
    native = GPT(GPTConfig(sequence_len=config.sequence_len, vocab_size=config.vocab_size, n_layer=config.layers,
                           n_head=config.heads, n_kv_head=kv_heads(config), n_embd=config.width, window_pattern="L"))
    native.init_weights()
    with torch.no_grad():
        for block in plugin.transformer.h:
            block.attn.c_proj.weight.normal_(std=.05)
            block.mlp.c_proj.weight.normal_(std=.05)
    native.load_state_dict(plugin.state_dict())
    assert plugin.transformer.h[1].attn.c_k.weight.shape == (24, 48)
    assert plugin.value_embeds["1"].weight.shape[1] == 24 and plugin.transformer.h[1].attn.ve_gate.weight.shape[0] == 1
    x = torch.randint(config.vocab_size, (2, config.sequence_len))
    torch.testing.assert_close(plugin(x), native(x), rtol=0, atol=0)


def test_grouped_cross_layer_attention_trains_every_parameter():
    model = model_of(share=True, groups=2)
    assert [attn.source for attn in (block.attn for block in model.transformer.h)] == [0, 1, 1, 3, 3, 5]
    x = torch.randint(model.config.vocab_size, (2, model.config.sequence_len))
    model(x, x).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    groups = parameter_groups(model)
    assert sorted(id(p) for group in groups for p in group["params"]) == sorted(id(p) for p in model.parameters())


@pytest.mark.parametrize("groups", [0, 5, True])
def test_the_group_size_must_divide_the_query_heads(groups):
    with pytest.raises(ValueError, match="heads_per_kv"):
        replace(candidate_config("cla"), options={"share_kv": True, "heads_per_kv": groups})


def test_grouped_heads_cost_less_in_the_ledger():
    full, grouped = (audit_graph(replace(debug_config("cla"), layers=6, options={"share_kv": True, "heads_per_kv": g}))
                     for g in (1, 2))
    assert not grouped["unknown"] and grouped["phases"]["forward"] < full["phases"]["forward"]
    assert grouped["operators"]["forward/native_attention"]["calls"] == 6


def test_the_ledger_charges_every_layer_attention_and_only_owned_projections():
    shared, control = (audit_graph(replace(debug_config("cla"), layers=6, options={"share_kv": share, "heads_per_kv": 1}))
                       for share in (True, False))
    assert not shared["unknown"] and not control["unknown"]
    attention = lambda report: report["operators"]["forward/native_attention"]
    assert attention(shared)["calls"] == attention(control)["calls"] == 6
    assert attention(shared)["theory"] == attention(control)["theory"]
    assert shared["phases"]["forward"] < control["phases"]["forward"]
    assert shared["parameters"] < control["parameters"]
