"""Trainer control flow on CPU with a debug model and synthetic rows.

Resources, tokenizer, packing and evaluators are replaced by fakes; the model,
optimizer, checkpoint, resume and segment logic are the real trainer code.
"""
from dataclasses import replace
import json
import os

import pytest
import torch

from nexttransformer import train
from nexttransformer.models.config import CHUNKED_METHODS, debug_config
from nexttransformer.models.looped import huginn_depth


class FakeTokenizer:
    def __init__(self, vocab): self.vocab = vocab
    def get_vocab_size(self): return self.vocab


class FakeStream:
    """Deterministic rows keyed by (rank, row index), with exact cursor state."""
    vocab, length = 64, 12

    def __init__(self, data_dir, tokenizer, split, rank=0, world=1, sequence_len=12):
        self.rank, self.length, self.rows = rank, sequence_len, 0

    def batch(self, size, device):
        out = []
        for row in range(self.rows, self.rows+size):
            g = torch.Generator().manual_seed(1_000_003*self.rank+row)
            out.append(torch.randint(self.vocab, (self.length+1,), generator=g))
        self.rows += size
        tokens = torch.stack(out).to(device)
        return tokens[:, :-1], tokens[:, 1:]

    def state_dict(self): return dict(rows=self.rows)
    def load_state_dict(self, state): self.rows = state["rows"]


def fake_bpb(model, tokenizer, data_dir, resources, *, batch, rank, world, device, tokens, split, smoke):
    return dict(bpb=1.0, input_tokens=tokens, split=split, seconds=0.0, smoke=smoke)


def fake_core(model, tokenizer, resources, device, *, smoke_max_per_task=None):
    return dict(core_metric=0.0, results={}, full_core=smoke_max_per_task is None)


@pytest.fixture(params=["kda", "gdn", "deltanet_hybrid", "kda_hybrid", "kda_no_ve", "mla_nope", "mha_no_ve", "qsa_hybrid", "moba_hybrid"])
def setup(request, tmp_path, monkeypatch):
    return runner(request.param, tmp_path, monkeypatch)


def runner(method, tmp_path, monkeypatch, findings=()):
    """Smoke-run `method` through train.main; `findings` are what research-mode review reports."""
    torch.set_num_threads(1)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"): monkeypatch.delenv(name, raising=False)
    config = debug_config(method)
    config = replace(config, backend="fla_chunk64" if method in CHUNKED_METHODS else config.backend,
                     vocab_size=FakeStream.vocab)
    proof = dict(formula_version="test", flops_per_token=1e9, total_training_flops=1,
                 stages=[dict(stage="joint", depth=None, cost=dict(total=5e15, phases=dict(forward=2e15)))])
    if config.has_sparse_burnin:
        config = replace(config, burnin_steps=3)
        proof["stages"].append(dict(stage="burnin", depth=None, cost=dict(total=6e15, phases=dict(forward=3e15))))
    monkeypatch.setattr(train, "admit_config", lambda path: (config, proof))
    monkeypatch.setattr(train, "review_config", lambda path: (config, proof, list(findings)))
    monkeypatch.setattr(train, "verify_resources", lambda resources, data_dir: (dict(tokenizer_sha256="t"), "manifest"))
    monkeypatch.setattr(train.RustBPETokenizer, "from_directory", staticmethod(lambda path: FakeTokenizer(64)))
    monkeypatch.setattr(train, "BestFitStream", FakeStream)
    monkeypatch.setattr(train, "evaluate_bpb", fake_bpb)
    monkeypatch.setattr(train, "evaluate_core", fake_core)
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))

    def run(output, *extra):
        train.main(["--config", str(path), "--output", str(output), "--runtime-dir", str(tmp_path/"runtime"),
                    "--resources", str(tmp_path/"resources"), "--data-dir", str(tmp_path/"data"),
                    "--micro-batch", "2", "--smoke-steps", "6", "--smoke-accumulation", "2",
                    "--rolling-every", "2", "--log-every", "1", "--wandb-mode", "disabled", *extra])
    return run


def weights(output, step):
    return torch.load(output/f"checkpoint_{step:08d}"/"model.pt", map_location="cpu", weights_only=False)["model"]


def losses(output):
    lines = (output/"metrics.jsonl").read_text().splitlines()
    return {e["step"]: e["loss"] for e in map(json.loads, lines) if e["event"] == "train"}


def test_segmented_resume_matches_continuous_run(setup, tmp_path):
    continuous, segmented = tmp_path/"continuous", tmp_path/"segmented"
    setup(continuous)
    status = json.loads((continuous/"run_status.json").read_text())
    assert status["phase"] == "smoke_complete" and status["step"] == 6
    # Rolling checkpoints are pruned to the newest two.
    assert sorted(p.name for p in continuous.glob("checkpoint_*")) == ["checkpoint_00000004", "checkpoint_00000006"]

    setup(segmented, "--max-wall-hours", "1e-12")  # ends after the first update
    assert json.loads((segmented/"run_status.json").read_text())["phase"] == "segment_complete"
    (segmented/"checkpoint_00000099").mkdir()  # an interrupted save without complete.json
    setup(segmented)
    assert (segmented/"incomplete_checkpoints").exists()
    assert losses(segmented) == losses(continuous)
    a, b = weights(continuous, 6), weights(segmented, 6)
    assert a.keys() == b.keys()
    for name in a: assert torch.equal(a[name], b[name]), name
    assert json.loads((segmented/"run_status.json").read_text())["phase"] == "smoke_complete"
    assert (segmented/"resume_00000001.json").exists()


def test_training_defaults_to_research_mode_and_records_its_findings(tmp_path, monkeypatch):
    finding = "PyTorch version differs from cost evidence; re-audit in the target environment"
    output = tmp_path/"research"
    runner("gdn", tmp_path, monkeypatch, findings=[finding])(output)
    assert json.loads((output/"provenance.json").read_text())["admission"] == \
        dict(mode="research", findings=[finding], method_source_hash=None)
    assert json.loads((output/"run_status.json").read_text())["mode"] == "research"


def test_formal_mode_is_kept_on_resume(tmp_path, monkeypatch):
    run, output = runner("gdn", tmp_path, monkeypatch), tmp_path/"formal"
    run(output, "--formal", "--max-wall-hours", "1e-12")
    assert json.loads((output/"provenance.json").read_text())["admission"] == \
        dict(mode="formal", findings=[], method_source_hash=None)
    with pytest.raises(ValueError, match="started in formal mode"):
        run(output)
    run(output, "--formal")
    status = json.loads((output/"run_status.json").read_text())
    assert status["phase"] == "smoke_complete" and status["mode"] == "formal"


def test_a_later_research_option_overrides_a_launcher_formal_one(tmp_path):
    base = ["--config", "config.json", "--output", str(tmp_path), "--micro-batch", "2"]
    assert not train.parse(base).formal and train.parse(base+["--formal"]).formal
    assert not train.parse(base+["--formal", "--research"]).formal


def test_formal_runs_accept_only_bit_identical_speedups(tmp_path):
    base = ["--config", "config.json", "--output", str(tmp_path), "--micro-batch", "2"]
    assert train.parse(base).speedups == ""
    for spec in ("exact", "norm,readout"):
        assert train.parse(base+["--formal", "--speedups", spec]).speedups == spec
    assert train.parse(base+["--speedups", "all"]).speedups == "all"
    for spec in ("all", "exact,cudnn", "merge"):
        with pytest.raises(SystemExit): train.parse(base+["--formal", "--speedups", spec])
    with pytest.raises(SystemExit): train.parse(base+["--speedups", "fast"])


def test_a_speedups_selection_holds_for_every_segment(tmp_path, monkeypatch):
    from nexttransformer.runtime import speedups
    run = runner("mla_nope", tmp_path, monkeypatch)
    # The kernels need CUDA; what is tested here is how the trainer records and compares the selection.
    monkeypatch.setattr(speedups, "install", lambda spec: sorted(speedups.selection(spec)))
    monkeypatch.setattr(speedups, "active", lambda name: False)
    run(tmp_path/"with", "--smoke-steps", "2", "--speedups", "exact")
    assert json.loads((tmp_path/"with"/"provenance.json").read_text())["software"]["speedups"] == "exact"
    with pytest.raises(ValueError, match="speedups"): run(tmp_path/"with")
    run(tmp_path/"without", "--smoke-steps", "2")
    with pytest.raises(ValueError, match="speedups"): run(tmp_path/"without", "--speedups", "exact")


def test_deterministic_attention_option_runs_an_update(setup, tmp_path, monkeypatch):
    from nanochat import flash_attention
    monkeypatch.setattr(flash_attention, "USE_FA3", False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    flash = torch.backends.cuda.flash_sdp_enabled()
    try:
        output = tmp_path/"deterministic"
        setup(output, "--deterministic-attention", "--smoke-steps", "1")
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert json.loads((output/"run_status.json").read_text())["step"] == 1
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cuda.enable_flash_sdp(flash)


def test_existing_run_is_never_restarted(setup, tmp_path):
    output = tmp_path/"run"
    setup(output)
    with pytest.raises(ValueError, match="already contains a run"):
        setup(output, "--resume", "never")


def test_stage_cost_follows_the_executed_stage():
    def proof(*stages): return dict(stages=[dict(stage=s, depth=d, cost=dict(total=t, phases=dict(forward=t//3)))
                                           for s, d, t in stages])
    kda = debug_config("kda")
    assert train.stage_cost(proof(("joint", None, 9)), kda, 100) == (9, 3)
    dsa = debug_config("dsa")
    sparse = proof(("burnin", None, 30), ("joint", None, 60))
    assert train.stage_cost(sparse, dsa, dsa.burnin_steps-1)[0] == 30
    assert train.stage_cost(sparse, dsa, dsa.burnin_steps)[0] == 60
    huginn = debug_config("huginn")
    depths = proof(*[("joint", d, 10*d) for d in range(1, huginn.max_loops+1)])
    assert train.stage_cost(depths, huginn, 7)[0] == 10*huginn_depth(huginn, 7)
    ouro = debug_config("ouro")
    stages = proof(("joint", None, 9), ("gate_only", None, 3))
    assert train.stage_cost(stages, ouro, train.gate_boundary(ouro)-1)[0] == 9
    assert train.stage_cost(stages, ouro, train.gate_boundary(ouro))[0] == 3


@pytest.mark.parametrize("keep", [0, 2, 3, 8, 20])
def test_rolling_pruning_keeps_the_newest_checkpoints(tmp_path, keep):
    saved = []
    for step in range(500, 12001, 500):  # trainer order: save, then prune
        path = tmp_path/f"checkpoint_{step:08d}"; path.mkdir(); (path/"complete.json").write_text("{}")
        saved.append(step)
        train.prune_rolling(tmp_path, keep, 46875, 5000)
        rolling = [s for s in saved if s % 5000]
        expected = sorted([s for s in saved if not s % 5000]+rolling[max(len(rolling)-keep, 0):])
        assert sorted(train.checkpoints(tmp_path)[0]) == expected, step


def test_kernel_caches_are_per_local_rank():
    env = {"TRITON_CACHE_DIR": "/ssd/kernels/", "TORCHINDUCTOR_CACHE_DIR": "/ssd/kernels/inductor", "LOCAL_RANK": "3"}
    train.rank_kernel_caches(env)
    assert env == {"TRITON_CACHE_DIR": "/ssd/kernels/rank3", "TORCHINDUCTOR_CACHE_DIR": "/ssd/kernels/inductor/rank3",
                   "LOCAL_RANK": "3"}
    single = {"TRITON_CACHE_DIR": "/ssd/kernels"}  # not launched by torchrun: unchanged
    train.rank_kernel_caches(single)
    assert single == {"TRITON_CACHE_DIR": "/ssd/kernels"}




def test_optimizer_kernels_keep_one_configuration(monkeypatch):
    from torch._inductor import config as inductor
    from nexttransformer.training.sharded import ExperimentOptimizer
    from nanochat.optim import MuonAdamW
    seen = {}
    monkeypatch.setattr(MuonAdamW, "step", lambda self: seen.update(
        pointwise=inductor.triton.autotune_pointwise, coalescing=inductor.triton.coalesce_tiling_analysis,
        scaling=inductor.dynamic_scale_rblock, limit=torch._dynamo.config.recompile_limit))
    optimizer = ExperimentOptimizer([dict(kind="adamw", params=[torch.nn.Parameter(torch.zeros(2))], lr=.1,
                                          betas=(.9, .95), eps=1e-8, weight_decay=0.)])
    before = (inductor.triton.autotune_pointwise, inductor.triton.coalesce_tiling_analysis, inductor.dynamic_scale_rblock)
    optimizer.step()
    assert seen == dict(pointwise=False, coalescing=False, scaling=False, limit=64)
    assert (inductor.triton.autotune_pointwise, inductor.triton.coalesce_tiling_analysis, inductor.dynamic_scale_rblock) == before


def test_native_attention_kernel_is_part_of_the_segment_identity(monkeypatch):
    # A resumed segment must load the same attention kernel; pure linear rows have none.
    hybrid = debug_config("gdn_hybrid")
    assert train.software_identity(hybrid)["native_attention"] == "sdpa"
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)
    assert train.software_identity(hybrid)["native_attention"] == "sdpa-noflash"
    assert "native_attention" not in train.software_identity(debug_config("gdn"))


def ouro_runner(tmp_path, monkeypatch, boundary=3):
    """Ouro with its gate-only stage moved to update `boundary` of a six-update smoke run."""
    torch.set_num_threads(1)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"): monkeypatch.delenv(name, raising=False)
    config = replace(debug_config("ouro"), vocab_size=FakeStream.vocab,
                     ouro_gate_steps=train.Protocol().steps-boundary)
    proof = dict(formula_version="test", flops_per_token=1e9, total_training_flops=1,
                 stages=[dict(stage="joint", depth=None, cost=dict(total=5e15, phases=dict(forward=2e15))),
                         dict(stage="gate_only", depth=None, cost=dict(total=2e15, phases=dict(forward=1e15)))])
    monkeypatch.setattr(train, "admit_config", lambda path: (config, proof))
    monkeypatch.setattr(train, "review_config", lambda path: (config, proof, []))
    monkeypatch.setattr(train, "verify_resources", lambda resources, data_dir: (dict(tokenizer_sha256="t"), "manifest"))
    monkeypatch.setattr(train.RustBPETokenizer, "from_directory", staticmethod(lambda path: FakeTokenizer(64)))
    monkeypatch.setattr(train, "BestFitStream", FakeStream)
    monkeypatch.setattr(train, "evaluate_bpb", fake_bpb)
    monkeypatch.setattr(train, "evaluate_core", fake_core)
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))

    def run(output, *extra):
        train.main(["--config", str(path), "--output", str(output), "--runtime-dir", str(tmp_path/"runtime"),
                    "--resources", str(tmp_path/"resources"), "--data-dir", str(tmp_path/"data"),
                    "--micro-batch", "2", "--smoke-steps", "6", "--smoke-accumulation", "2",
                    "--rolling-every", "2", "--log-every", "1", "--wandb-mode", "disabled", *extra])
    return config, run


def test_ouro_gate_only_stage_trains_only_the_gate_and_resumes_bitwise(tmp_path, monkeypatch):
    config, run = ouro_runner(tmp_path, monkeypatch)
    continuous, before, after = tmp_path/"continuous", tmp_path/"before", tmp_path/"after"
    run(continuous)
    events = [json.loads(line) for line in (continuous/"metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event"] == "stage_transition"] == [3]
    model = train.build_model(config, "cpu"); model.set_training_stage(3)
    gate = {name for name, p in model.named_parameters() if p.requires_grad}
    a4, a6 = weights(continuous, 4), weights(continuous, 6)
    changed = {name for name in a6 if not torch.equal(a4[name], a6[name])}
    assert changed and changed <= gate and gate != set(a6)
    run(before, "--max-wall-hours", "1e-12")  # resumes before the boundary and crosses it
    run(before)
    run(after, "--smoke-steps", "4")  # resumes after the boundary: gate-only optimizer from the start
    run(after)
    for output in (before, after):
        assert losses(output) == losses(continuous)
        b = weights(output, 6)
        for name in a6: assert torch.equal(a6[name], b[name]), (output.name, name)


def test_sparse_backend_is_scoped_and_part_of_the_segment_identity(tmp_path, monkeypatch):
    _, run = ouro_runner(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="sparse methods only"):
        run(tmp_path/"run", "--sparse-backend", "kernel")
    assert train.software_identity(debug_config("dsa"), "kernel")["sparse_backend"] == "kernel"
    assert "sparse_backend" not in train.software_identity(debug_config("kda"))
