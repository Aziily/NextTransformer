"""Plugin-mode tests, run by tests/test_plugins.py with NEXTTRANSFORMER_METHOD_PATHS=tests/plugins.

The three test plugins cover the mixer, ffn, model and stages hooks. The file is not named test_*.py, so
the ordinary run never collects it without the plugins registered.
"""
from dataclasses import replace
import json

import pytest
import torch

from nexttransformer import train
from nexttransformer.models.config import LINEAR, LOOPED, ModelConfig, candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.models.plugins import stage_at, stages
from nexttransformer.models.registry import BUILTIN_METHODS, FAMILY, METHODS, PLUGIN_METHODS
from nexttransformer.protocol.calibrate import Auditor
from nexttransformer.protocol.compute import audit_graph
from nexttransformer.protocol.fingerprint import method_files, published, status, traced
from nexttransformer.protocol.spec import Protocol, load_config, write_config
from tests.test_train import FakeStream, FakeTokenizer, fake_bpb, fake_core

PLUGINS = ("toy_gated_attention", "toy_glu", "toy_two_pass")


def test_plugins_are_registered_after_the_built_in_methods():
    # The test plugins join the repository's own plugins (nexttransformer/methods/), all in name order after the built-ins.
    assert set(PLUGINS) <= PLUGIN_METHODS and METHODS == BUILTIN_METHODS + tuple(sorted(PLUGIN_METHODS))
    assert FAMILY["toy_two_pass"] == "looped" and FAMILY["toy_glu"] == "others"
    assert not (LINEAR | LOOPED) & PLUGIN_METHODS  # model rules stay built-in only


def test_plugins_leave_every_published_proof_valid():
    stale = [method for method in published() if len(set(status(method))) != 1 or status(method)[1] is None]
    assert not stale


def test_options_are_declared_validated_and_round_trip(tmp_path):
    config = candidate_config("toy_glu")
    assert config.option("activation") == "silu" and config.serialized()["options"] == {"activation": "silu"}
    assert "options" not in candidate_config("gdn").serialized()
    with pytest.raises(ValueError, match="silu"): replace(config, options={"activation": "relu"})
    with pytest.raises(ValueError, match="exactly"): replace(config, options={})
    with pytest.raises(ValueError, match="plugin"): ModelConfig(method="gdn", options={"x": 1})
    with pytest.raises(ValueError, match="multiple of 2"): replace(config, layers=23)
    write_config(tmp_path/"toy_glu.json", config)
    assert load_config(tmp_path/"toy_glu.json") == config


@pytest.mark.parametrize("method", PLUGINS)
def test_each_hook_builds_and_trains_through_the_shared_shell(method):
    config = debug_config(method)
    model = build_model(config, "cpu", seed=1)
    model.train(); model.set_training_stage(0)
    x = torch.randint(config.vocab_size, (2, config.sequence_len))
    loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss) and model.last_aux_loss is not None
    named = dict(model.named_parameters())
    if method == "toy_gated_attention":
        assert config.option("gate_bias") == .5  # debug option
        # The init hook zeroes the gate; the shell's zero output projection is the first to learn.
        assert torch.count_nonzero(named["transformer.h.0.attn.gate.weight"]) == 0
        assert named["transformer.h.0.attn.c_proj.weight"].grad.abs().sum() > 0
    if method == "toy_glu":
        assert named["transformer.h.0.mlp.c_fc.weight"].shape[0] == 2*config.mlp_width
    if method == "toy_two_pass":
        assert type(model).__name__ == "TwoPass" and model.passes == 1
        assert not named["backout_lambda"].requires_grad  # the frozen hook
        model.set_training_stage(stage_at(config, 600)[1])
        assert model.passes == 2


def test_stages_drive_the_audit_and_the_trainer_cost():
    config = debug_config("toy_two_pass")
    assert stages(config) == [("single", 512), ("repeated", Protocol().steps-512)]
    assert stage_at(config, 511) == ("single", 0) and stage_at(config, 512) == ("repeated", 512)
    assert Auditor.stages(None, config) == [("single", None, 512), ("repeated", None, Protocol().steps-512)]
    single, repeated = (audit_graph(config, sequence_len=config.sequence_len, batch=2, stage=s)
                        for s in ("single", "repeated"))
    assert not single["unknown"] and not repeated["unknown"]
    assert repeated["phases"]["forward"] > 1.5*single["phases"]["forward"]
    proof = dict(stages=[dict(stage="single", depth=None, cost=dict(total=1, phases=dict(forward=1))),
                         dict(stage="repeated", depth=None, cost=dict(total=2, phases=dict(forward=2)))])
    assert train.stage_cost(proof, config, 0) == (1, 1) and train.stage_cost(proof, config, 600) == (2, 2)


@pytest.mark.parametrize("method", PLUGINS)
def test_the_fingerprint_covers_the_plugin_package_and_never_another(method):
    config = debug_config(method)
    with traced() as executed:
        model = build_model(config, "cpu")
        x = torch.zeros((1, config.sequence_len), dtype=torch.long)
        model(x, x).backward()
    files = method_files(executed)
    assert f"tests/plugins/{method}/__init__.py" in files
    assert not [f for f in files if f.startswith("tests/plugins/") and f"/{method}/" not in f]


@pytest.mark.parametrize("method", PLUGINS)
def test_a_research_smoke_run_trains_a_plugin(method, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"): monkeypatch.delenv(name, raising=False)
    config = replace(debug_config(method), vocab_size=FakeStream.vocab)
    proof = dict(formula_version="test", flops_per_token=1e9, total_training_flops=1,
                 stages=[dict(stage=name, depth=None, cost=dict(total=5e15, phases=dict(forward=2e15)))
                         for name, _ in stages(config)])
    monkeypatch.setattr(train, "review_config", lambda path: (config, proof, []))
    monkeypatch.setattr(train, "verify_resources", lambda resources, data_dir: (dict(tokenizer_sha256="t"), "manifest"))
    monkeypatch.setattr(train.RustBPETokenizer, "from_directory", staticmethod(lambda path: FakeTokenizer(64)))
    monkeypatch.setattr(train, "BestFitStream", FakeStream)
    monkeypatch.setattr(train, "evaluate_bpb", fake_bpb)
    monkeypatch.setattr(train, "evaluate_core", fake_core)
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))
    output = tmp_path/"run"
    train.main(["--config", str(path), "--output", str(output), "--runtime-dir", str(tmp_path/"runtime"),
                "--resources", str(tmp_path/"resources"), "--data-dir", str(tmp_path/"data"), "--micro-batch", "2",
                "--smoke-steps", "3", "--smoke-accumulation", "2", "--log-every", "1", "--wandb-mode", "disabled"])
    status = json.loads((output/"run_status.json").read_text())
    assert status["phase"] == "smoke_complete" and status["mode"] == "research"
