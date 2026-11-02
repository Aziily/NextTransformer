from pathlib import Path
import types

import torch
from torch import nn

from launch.profile_hooks import Timeline, install


def counter(start=0.0):
    state = {"now": start}

    def tick():
        state["now"] += 1.0
        return state["now"]

    return tick


def stages(record, step):
    return [segment["stage"] for segment in record["steps"][step]["segments"]]


def test_timeline_groups_segments_by_update():
    timeline = Timeline(rank=3, world=8, clock=counter(), wall=counter(100.0))
    timeline.start_step()
    timeline.mark("data")
    timeline.mark("forward")
    timeline.mark("forward")
    timeline.mark("backward")
    timeline.start_step()
    timeline.mark("data")
    record = timeline.finish()

    assert record["rank"] == 3 and record["world"] == 8
    assert [step["step"] for step in record["steps"]] == [0, 1, 2]
    assert stages(record, 0) == ["startup"]
    assert stages(record, 1) == ["step_setup", "data", "forward", "backward"]
    assert stages(record, 2) == ["step_setup", "data"]
    segments = [segment for step in record["steps"] for segment in step["segments"]]
    assert all(segment["host_s"] == 1.0 for segment in segments)
    assert all(segment["device_s"] is None for segment in segments)
    assert record["steps"][1]["segments"][0]["start_wall"] < record["steps"][1]["segments"][1]["start_wall"]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x, y=None):
        return self.linear(x).square().mean()


class FakeOptimizer:
    def __init__(self):
        self.order = []

    def _reduce_muon(self, group, world_size):
        self.order.append("reduce")
        return {}

    def _compute_muon(self, group, info, gather_list, rank):
        self.order.append("compute")

    def _finish_gathers(self, gather_list):
        self.order.append("gather")

    def step(self):
        info = self._reduce_muon(None, 8)
        self._compute_muon(None, info, [], 0)
        self._finish_gathers([])
        return "stepped"


def fake_trainer():
    def set_schedule(optimizer, step, config):
        return 0.5

    def share_batches(stream, batch, period, share, shares, device):
        yield from [("x0", "y0"), ("x1", "y1")]

    def build_model(config, device="cpu", seed=42):
        return TinyModel()

    def build_optimizer(model, train=None):
        return FakeOptimizer()

    def evaluate_bpb(model, *args, **kwargs):
        model(torch.ones(1, 2))
        return {"bpb": 1.0}

    def evaluate_core(model, *args, **kwargs):
        return {"core": 0.1}

    def save_checkpoint(*args):
        return Path("checkpoint")

    return types.SimpleNamespace(
        set_schedule=set_schedule, share_batches=share_batches, build_model=build_model,
        build_optimizer=build_optimizer, evaluate_bpb=evaluate_bpb, evaluate_core=evaluate_core,
        save_checkpoint=save_checkpoint,
    )


def test_install_records_trainer_stages_and_restores_everything():
    trainer = fake_trainer()
    originals = dict(vars(trainer))
    clip = torch.nn.utils.clip_grad_norm_
    timeline = Timeline(clock=counter(), wall=counter(100.0))
    uninstall = install(trainer, timeline)
    try:
        model = trainer.build_model(None)
        optimizer = trainer.build_optimizer(model)
        assert trainer.evaluate_bpb(model) == {"bpb": 1.0}
        for step in range(2):
            assert trainer.set_schedule(optimizer, step, None) == 0.5
            batches = trainer.share_batches(None, 1, 2, 0, 1, "cpu")
            seen = []
            for _ in range(2):
                seen.append(next(batches))
                model(torch.ones(1, 2)).backward()
            assert next(batches, None) is None
            assert seen == [("x0", "y0"), ("x1", "y1")]
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            assert optimizer.step() == "stepped"
        trainer.evaluate_bpb(model)
        trainer.save_checkpoint()
    finally:
        uninstall()
    record = timeline.finish()

    update = [
        "step_setup", "data", "forward", "backward", "data", "forward", "backward", "data",
        "grad_norm", "wait", "optimizer.reduce", "optimizer.compute", "optimizer.gather", "wait",
    ]
    assert stages(record, 0) == ["startup", "validation", "step_tail"]
    assert stages(record, 1) == update
    assert stages(record, 2) == update + ["validation", "step_tail", "checkpoint", "step_tail"]
    assert optimizer.order == ["reduce", "compute", "gather"] * 2
    assert dict(vars(trainer)) == originals
    assert torch.nn.utils.clip_grad_norm_ is clip


class FakeProfiler:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def export_chrome_trace(self, path):
        self.events.append(("export", Path(path).name))


def test_timeline_profiles_only_the_requested_updates(tmp_path: Path):
    profiler = FakeProfiler()
    timeline = Timeline(rank=1, world=8, profile_steps=(2, 2), profiler_factory=lambda: profiler,
                        trace_dir=tmp_path, clock=counter(), wall=counter(100.0))
    timeline.start_step()
    timeline.mark("forward")
    assert profiler.events == []
    timeline.start_step()
    assert profiler.events == ["start"]
    timeline.mark("forward")
    timeline.mark("step_tail")
    assert profiler.events == ["start", "stop", ("export", "rank_0001.json.gz")]
    timeline.start_step()
    timeline.finish()
    assert profiler.events == ["start", "stop", ("export", "rank_0001.json.gz")]
    assert timeline.finish()["profiled_steps"] == [2]


class Recorder:
    def __init__(self):
        self.names = []

    def __call__(self, name):
        recorder = self

        class Range:
            def __enter__(self):
                recorder.names.append(("enter", name))

            def __exit__(self, *exc):
                recorder.names.append(("exit", name))

        return Range()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(2, 2)
        self.mlp = nn.Linear(2, 2)

    def forward(self, x):
        return self.mlp(self.attn(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([Block(), Block()])
        self.lm_head = nn.Linear(2, 1)

    def forward(self, x, y=None):
        for block in self.h:
            x = block(x)
        return self.lm_head(x).square().mean()


def test_module_annotations_cover_blocks_and_their_children_only_in_a_trace_window():
    from launch.profile_hooks import annotate_modules

    recorder = Recorder()
    timeline = Timeline(profile_steps=(2, 2), profiler_factory=FakeProfiler, annotate=recorder, clock=counter(),
                        wall=counter(100.0))
    model = Net()
    annotate_modules(model, timeline)
    timeline.start_step()
    model(torch.ones(1, 2))
    assert recorder.names == []
    timeline.start_step()
    model(torch.ones(1, 2))
    names = [name for action, name in recorder.names if action == "enter" and name.startswith("nt.module:")]
    assert names == ["nt.module:h.0|Block", "nt.module:h.0.attn|Linear", "nt.module:h.0.mlp|Linear",
                     "nt.module:h.1|Block", "nt.module:h.1.attn|Linear", "nt.module:h.1.mlp|Linear",
                     "nt.module:lm_head|Linear"]
    modules = [(action, name) for action, name in recorder.names if name.startswith("nt.module:")]
    assert sorted(name for action, name in modules if action == "enter") == \
        sorted(name for action, name in modules if action == "exit")


def test_timeline_reads_peak_memory_once_per_update_before_the_next_reset():
    memory = iter([(10, 20), (30, 40), (50, 60)])
    timeline = Timeline(clock=counter(), wall=counter(100.0), memory=lambda: next(memory))
    timeline.start_step()
    timeline.mark("forward")
    timeline.mark("step_tail")
    timeline.mark("validation")
    timeline.mark("step_tail")
    timeline.start_step()
    timeline.mark("step_tail")
    record = timeline.finish()
    assert record["steps"][1]["memory"] == {"allocated_peak": 10, "reserved_peak": 20}
    assert record["steps"][2]["memory"] == {"allocated_peak": 30, "reserved_peak": 40}
    assert "memory" not in record["steps"][0]
