"""In-process stage timing for the research profiling launcher.

    python -m launch.profile_hooks --profile-dir DIR [--profile-trace torch|nsys --profile-trace-steps FIRST LAST] \
        TRAINER ARGUMENTS ...

runs ``nexttransformer.train`` unchanged in every rank and records when the trainer enters each stage of an update:
data, forward, backward, gradient norm, host waits, the optimizer's three phases, evaluation and checkpoints, plus the
peak memory of each update. In a trace window it also names model components. The wrappers only read clocks and
memory counters, record CUDA events and annotate the trace; they never touch tensors, so the training math and
protocol are unchanged. Each rank writes ``DIR/rank_XXXX.json`` when the trainer returns.

Two research overrides exist for profiling only: ``--profile-sparse-stage joint`` runs staged sparse methods in
their joint stage from the first update, and ``--profile-flex-options`` passes FlexAttention kernel options (for
example a pipeline depth that fits the shared memory of a GPU) to the sparse kernel backend.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time


STAGE_PREFIX = "nt.stage:"
MODULE_PREFIX = "nt.module:"


class Timeline:
    """Contiguous stage segments of one rank, grouped by update; step 0 is everything before the first update."""

    def __init__(self, rank=0, world=1, device=None, profile_steps=None, profiler_factory=None, trace_dir=None,
                 annotate=None, clock=time.perf_counter, wall=time.time, memory=None):
        self.rank, self.world, self.device = rank, world, device
        self.profile_steps = profile_steps
        self.profiler_factory, self.trace_dir, self.annotate = profiler_factory, trace_dir, annotate
        self.clock, self.wall, self.memory = clock, wall, memory
        self.device_info = None
        self.step, self.stage = 0, "startup"
        self.steps = [{"step": 0, "profiled": False, "segments": []}]
        self._cuda = False
        self._open = self._boundary()
        self._annotation = self._profiler = self._record = None
        self._profiled: list[int] = []

    def _in_window(self, step):
        return self.profile_steps is not None and self.profile_steps[0] <= step <= self.profile_steps[1]

    def _boundary(self):
        event = None
        if self._cuda:
            import torch
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return self.clock(), self.wall(), event

    def _close(self, boundary):
        if self._annotation is not None:
            self._annotation.__exit__(None, None, None)
            self._annotation = None
        start, start_wall, event = self._open
        self.steps[-1]["segments"].append(dict(stage=self.stage, host_s=boundary[0]-start, start_wall=start_wall,
                                                _events=(event, boundary[2])))

    def _begin(self, stage, boundary):
        self.stage, self._open = stage, boundary
        if self._profiler is not None and self.annotate is not None:
            self._annotation = self.annotate(STAGE_PREFIX + stage)
            self._annotation.__enter__()

    def start_step(self):
        """Open the next update; the trainer calls this right after its step-start synchronization."""
        if self.device is not None and getattr(self.device, "type", None) == "cuda":
            self._cuda = True
        boundary = self._boundary()
        self._close(boundary)
        self.step += 1
        # Every rank marks the window, also ranks that record no trace: they wait for traced ranks at collectives.
        profiled = self._in_window(self.step)
        self.steps.append({"step": self.step, "profiled": profiled, "segments": []})
        if profiled and self._profiler is None and self.profiler_factory is not None and not self._profiled:
            self._profiler = self.profiler_factory()
            self._profiler.start()
        if profiled and self._profiler is not None:
            self._profiled.append(self.step)
        self._begin("step_setup", boundary)

    def mark(self, stage):
        """Start `stage` now and end the current segment; repeated marks of the current stage are ignored."""
        if stage == self.stage:
            return
        boundary = self._boundary()
        self._close(boundary)
        if stage == "step_tail":
            # The trainer resets the peak-memory counters when the next update starts.
            current = self.steps[-1]
            if self.memory is not None and self.step > 0 and "memory" not in current:
                allocated, reserved = self.memory()
                current["memory"] = {"allocated_peak": allocated, "reserved_peak": reserved}
            if self._profiler is not None and self.step >= self.profile_steps[1]:
                self._stop_profiler()
        self._begin(stage, boundary)

    @property
    def tracing(self):
        return self._profiler is not None and self.annotate is not None

    def _stop_profiler(self):
        # Export as soon as the window closes: a later crash keeps the trace and the buffers are freed.
        profiler, self._profiler = self._profiler, None
        profiler.stop()
        if self.trace_dir is not None:
            Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(Path(self.trace_dir) / f"rank_{self.rank:04d}.json.gz"))

    def finish(self):
        """Close the open segment, stop an open trace window and resolve device times; idempotent."""
        if self._record is not None:
            return self._record
        self._close(self._boundary())
        if self._profiler is not None:
            self._stop_profiler()
        resolve = True
        if self._cuda:
            import torch
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:
                # A failed run keeps its host timings; the trainer's own error is the one to report.
                resolve = False
        for step in self.steps:
            for segment in step["segments"]:
                start, end = segment.pop("_events")
                timed = resolve and start is not None and end is not None
                segment["device_s"] = start.elapsed_time(end) / 1000 if timed else None
        self._record = dict(rank=self.rank, world=self.world, profiled_steps=self._profiled, steps=self.steps,
                            device=self.device_info)
        return self._record


def annotate_modules(model, timeline):
    """Name model components in traced updates: every block of a module list, the block's direct children, and
    the other leaf modules under the root. Outside a trace window each hook only checks a flag."""
    import torch

    containers = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
    selected = []

    def walk(module, prefix, in_list):
        for name, child in module.named_children():
            path = f"{prefix}{name}"
            if isinstance(child, containers):
                walk(child, path + ".", isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)))
                continue
            selected.append((path, child))
            if in_list:
                selected.extend((f"{path}.{inner}", grandchild) for inner, grandchild in child.named_children())

    walk(model, "", False)
    open_ranges = {}

    for path, module in selected:
        label = f"{MODULE_PREFIX}{path}|{type(module).__name__}"

        def enter(module, args, label=label):
            if timeline.tracing:
                scope = timeline.annotate(label)
                scope.__enter__()
                open_ranges.setdefault(id(module), []).append(scope)

        def leave(module, args, output):
            scopes = open_ranges.get(id(module))
            if scopes:
                scopes.pop().__exit__(None, None, None)

        module.register_forward_pre_hook(enter)
        module.register_forward_hook(leave)
    return model


def _before(timeline, stage, original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        timeline.mark(stage)
        return original(*args, **kwargs)
    return wrapped


def _around(timeline, stage, after, original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        timeline.mark(stage)
        try:
            return original(*args, **kwargs)
        finally:
            timeline.mark(after)
    return wrapped


_OPTIMIZER_PHASES = (("_reduce_adamw", "optimizer.reduce"), ("_reduce_muon", "optimizer.reduce"),
                     ("_compute_adamw", "optimizer.compute"), ("_compute_muon", "optimizer.compute"),
                     ("_finish_gathers", "optimizer.gather"))


def instrument_optimizer(optimizer, timeline):
    """Mark the Muon/AdamW phases on this optimizer instance; the class and its math stay untouched."""
    if getattr(optimizer, "_profile_timeline", None) is timeline:
        return optimizer
    for name, stage in _OPTIMIZER_PHASES:
        if hasattr(optimizer, name):
            setattr(optimizer, name, _before(timeline, stage, getattr(optimizer, name)))
    step = optimizer.step

    @functools.wraps(step)
    def stepped(*args, **kwargs):
        try:
            return step(*args, **kwargs)
        finally:
            timeline.mark("wait")

    optimizer.step = stepped
    optimizer._profile_timeline = timeline
    return optimizer


def install(trainer, timeline):
    """Wrap the trainer module's stage entry points and return a function that restores every one."""
    import torch

    saved = []

    def patch(owner, name, make):
        if hasattr(owner, name):
            original = getattr(owner, name)
            saved.append((owner, name, original))
            setattr(owner, name, make(original))

    def schedule(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            timeline.start_step()
            return original(*args, **kwargs)
        return wrapped

    def batches(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            inner = original(*args, **kwargs)
            while True:
                timeline.mark("data")
                try:
                    item = next(inner)
                except StopIteration:
                    return
                timeline.mark("forward")
                yield item
        return wrapped

    def forward_done(module, args, output):
        # Evaluation reuses the model; only a training forward hands over to backward.
        if timeline.stage == "forward":
            timeline.mark("backward")

    def model(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            built = original(*args, **kwargs)
            built.register_forward_hook(forward_done)
            return annotate_modules(built, timeline)
        return wrapped

    def optimizer(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return instrument_optimizer(original(*args, **kwargs), timeline)
        return wrapped

    def transition(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            built, changed = original(*args, **kwargs)
            return instrument_optimizer(built, timeline), changed
        return wrapped

    patch(trainer, "set_schedule", schedule)
    patch(trainer, "share_batches", batches)
    patch(trainer, "build_model", model)
    patch(trainer, "build_optimizer", optimizer)
    patch(trainer, "transition_stage", transition)
    patch(trainer, "evaluate_bpb", lambda original: _around(timeline, "validation", "step_tail", original))
    patch(trainer, "evaluate_core", lambda original: _around(timeline, "core", "step_tail", original))
    patch(trainer, "save_checkpoint", lambda original: _around(timeline, "checkpoint", "step_tail", original))
    patch(trainer, "reduce_sum", lambda original: _before(timeline, "step_tail", original))
    patch(torch.nn.utils, "clip_grad_norm_", lambda original: _around(timeline, "grad_norm", "wait", original))

    def uninstall():
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)

    return uninstall


def force_joint_stage(model):
    """Keep sparse indexers out of their burn-in stage, whatever the update number.

    A bounded profile starts at update 0, where staged sparse methods train only their indexer; this profiles the
    joint stage that formal training spends almost all of its updates in. Research profiling only.
    """
    set_step = getattr(model, "set_step", None)

    def joint():
        for module in model.modules():
            if hasattr(module, "in_burnin"):
                module.in_burnin = False

    if set_step is not None:
        @functools.wraps(set_step)
        def stepped(step):
            set_step(step)
            joint()
        model.set_step = stepped
    joint()
    return model


def patch_flex_options(module, options):
    """Pass FlexAttention ``kernel_options`` (pipeline depth, block sizes) to every call through ``module``.

    Explicit options of a call win. They select a kernel configuration only; the attention computed is the same.
    Returns a function that restores the module.
    """
    original = module.flex_attention

    @functools.wraps(original)
    def flex_attention(*args, **kwargs):
        kwargs["kernel_options"] = {**options, **(kwargs.get("kernel_options") or {})}
        return original(*args, **kwargs)

    module.flex_attention = flex_attention
    if hasattr(module, "_flex_kernel"):
        module._flex_kernel.cache_clear()

    def restore():
        module.flex_attention = original
        if hasattr(module, "_flex_kernel"):
            module._flex_kernel.cache_clear()

    return restore


class _CudaProfilerRange:
    """cudaProfilerStart/Stop, the capture range of ``nsys profile --capture-range=cudaProfilerApi``."""

    def start(self):
        import torch
        torch.cuda.profiler.start()

    def stop(self):
        import torch
        torch.cuda.profiler.stop()


class _NvtxRange:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import torch
        torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *exc):
        import torch
        torch.cuda.nvtx.range_pop()


def _torch_profiler():
    import torch
    from torch.profiler import ProfilerActivity
    return torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--profile-trace", choices=("none", "torch", "nsys"), default="none")
    parser.add_argument("--profile-trace-steps", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--profile-trace-ranks", type=int, nargs="+", help="ranks that record a torch trace (default: all)")
    parser.add_argument("--profile-sparse-stage", choices=("auto", "joint"), default="auto",
                        help="joint: profile the sparse joint stage from update 1 instead of the burn-in")
    parser.add_argument("--profile-flex-options", type=json.loads, default=None,
                        help="JSON FlexAttention kernel_options for the sparse kernel backend")
    args, trainer_argv = parser.parse_known_args(argv)
    if "--formal" in trainer_argv:
        parser.error("profiling is research-only and cannot run with --formal")
    if args.profile_trace != "none" and args.profile_trace_steps is None:
        parser.error("--profile-trace needs --profile-trace-steps")
    return args, trainer_argv


def main(argv=None):
    args, trainer_argv = _parse(sys.argv[1:] if argv is None else argv)
    # Like the trainer's own entry point: resolve the site configuration before torch is imported. Kernel cache
    # directories are read when the first kernel compiles, so per-rank caches can be set after the import.
    from nexttransformer.runtime.site import bootstrap_runtime
    bootstrap_runtime(trainer_argv)
    import nexttransformer.train as trainer
    trainer.rank_kernel_caches(os.environ)
    import torch

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else None
    factory = annotate = trace_dir = None
    traced = args.profile_trace_ranks is None or rank in args.profile_trace_ranks
    if args.profile_trace == "torch" and traced:
        factory, annotate, trace_dir = _torch_profiler, torch.profiler.record_function, args.profile_dir / "torch"
    elif args.profile_trace == "nsys":
        factory, annotate = _CudaProfilerRange, _NvtxRange
    memory = (lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))) \
        if device is not None else None
    timeline = Timeline(rank, world, device=device, profile_steps=args.profile_trace_steps, profiler_factory=factory,
                        trace_dir=trace_dir, annotate=annotate, memory=memory)
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        timeline.device_info = dict(name=properties.name, major=properties.major, minor=properties.minor,
                                    sms=properties.multi_processor_count, total_memory=properties.total_memory)
    uninstall = install(trainer, timeline)
    restores = [uninstall]
    if args.profile_sparse_stage == "joint":
        build = trainer.build_model
        trainer.build_model = functools.wraps(build)(lambda *a, **k: force_joint_stage(build(*a, **k)))
        restores.append(lambda: setattr(trainer, "build_model", build))
    if args.profile_flex_options:
        import nexttransformer.models.sparse_kernels as sparse_kernels
        restores.append(patch_flex_options(sparse_kernels, args.profile_flex_options))
    try:
        trainer.main(trainer_argv)
    finally:
        for restore in reversed(restores):
            restore()
        record = timeline.finish()
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        (args.profile_dir / f"rank_{rank:04d}.json").write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
