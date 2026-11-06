"""Reproduce what `--speedups` claims on the real trainer: speed, bit identity, resume and numerics.

    python -m nexttransformer.checks.speedups tests
    python -m nexttransformer.checks.speedups timing --base ../main --gpus 8
    python -m nexttransformer.checks.speedups bits --base ../main --gpus 8
    python -m nexttransformer.checks.speedups resume --gpus 8
    python -m nexttransformer.checks.speedups numerics --gpus 8

`--base` is a checkout of the revision to compare with, from `git worktree add ../main origin/main`; without
it the reference is this checkout without `--speedups`. Arguments this tool does not know go to the trainer,
so data and caches come from `--site-config` or the usual environment. Every run is a bounded smoke run under
`.agent-work/speedups/`, and the command exits with 1 when a comparison finds a difference.

`bits` and `resume` use deterministic attention, because the default SDPA flash backward differs between two
runs of the same code. `numerics` keeps it on and measures `all` against that run-to-run noise.
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO/".agent-work/speedups"
ANCHOR = "dense_full"
# One method per family of the shared shell, chosen to reach every patched path.
FAMILIES = ("gdn", "moba_hybrid", "deeploop", "mhc", "ouro", "mla_nope", "residual_standard", "attnres")
SPARSE_KERNEL = ("moba", "moba_hybrid", "nsa")
KEPT = ("loss", "lm_loss", "aux_loss", "grad_norm", "val_bpb")
# Checkpoint entries that differ between two runs of identical results: timing, what a resumed run evaluated
# once more, and the identity of the software, which records the flag itself and the commit of each checkout.
VOLATILE = ("training_seconds", "evaluation_seconds", "evaluation_forward_flops", "smoke_val_steps",
            "reference_source_hash", "native_source_sha256", "software/speedups", "software/git_commit")


def train(name, args, repo=REPO, method=ANCHOR, steps=34, speedups="", micro=16, deterministic=False, fresh=True, extra=()):
    """One bounded run; returns its directory."""
    from nexttransformer.protocol.spec import config_path
    out = WORK/name
    if fresh: shutil.rmtree(out, ignore_errors=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(args.gpus),
               "-m", "nexttransformer.train", "--config", str(config_path(method).relative_to(REPO)), "--output", str(out),
               "--micro-batch", str(micro), "--smoke-steps", str(steps), "--rolling-every", "100000", "--log-every", "1",
               *extra, *args.trainer]
    if method in SPARSE_KERNEL: command += ["--sparse-backend", "kernel"]
    if speedups: command += ["--speedups", speedups]
    env = dict(os.environ, PYTHONPATH=str(repo))
    env.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
    if deterministic:
        command.append("--deterministic-attention")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"-- {name}", flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True, stdout=subprocess.DEVNULL)
    return out


def train_events(run):
    return [e for e in map(json.loads, open(run/"metrics.jsonl")) if e.get("event") == "train"]


def step_ms(run, warmup=4):
    return statistics.median(e["step_seconds"] for e in train_events(run)[warmup:])*1e3


def events(run):
    """Logged values by (event, step): a resumed run repeats no step, but validates at its first segment's end."""
    rows = (e for e in map(json.loads, open(run/"metrics.jsonl")) if e.get("event") in ("train", "validation"))
    return {(e["event"], e["step"]): {k: e.get(k) for k in KEPT} for e in rows}


def digests(checkpoint):
    """Every tensor and scalar of a checkpoint directory, by path."""
    import torch
    out = {}

    def walk(path, value):
        if torch.is_tensor(value):
            data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            out[path] = hashlib.sha256(data).hexdigest()
        elif isinstance(value, dict):
            for key, item in value.items(): walk(f"{path}/{key}", item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value): walk(f"{path}/{index}", item)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[path] = repr(value)

    for file in sorted(checkpoint.glob("*.pt")):
        walk(file.name, torch.load(file, map_location="cpu", weights_only=False))
    return out


def compare(a, b):
    """Differences between two runs: events both logged, and the last checkpoint without its bookkeeping."""
    left, right = events(a), events(b)
    found = [f"event {key}" for key in sorted(set(left) & set(right)) if left[key] != right[key]]
    ca, cb = (digests(sorted(run.glob("checkpoint_*"))[-1]) for run in (a, b))
    volatile = lambda path: any(word in path for word in VOLATILE)
    found += [f"checkpoint {path}" for path in sorted(set(ca) | set(cb)) if ca.get(path) != cb.get(path) and not volatile(path)]
    return found


def report(label, found):
    print(f"{label}: {'identical' if not found else f'{len(found)} differences, first {found[:3]}'}", flush=True)
    return bool(found)


def tests(args):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_speedups.py"], cwd=REPO).returncode


def timing(args):
    """Reference and optimized runs alternate, so drift shows in the reference arms."""
    steps = args.updates or 34
    arms = (("reference", args.base, ""), ("all", REPO, "all"), ("reference", args.base, ""), ("exact", REPO, "exact"),
            ("reference", args.base, ""))
    times = {}
    for index, (label, repo, speedups) in enumerate(arms):
        times.setdefault(label, []).append(step_ms(train(f"timing_{index}_{label}", args, repo, steps=steps, speedups=speedups)))
    base = statistics.mean(times["reference"])
    for label, values in times.items():
        print(f"{label:10s} {statistics.mean(values):9.1f} ms/update {base/statistics.mean(values):6.3f}x  {[round(v, 1) for v in values]}")
    return 0


def bits(args):
    failed = False
    methods = args.methods.split(",") if args.methods else (ANCHOR, *FAMILIES)
    for method in methods:
        steps = args.updates or (30 if method == ANCHOR else 6)
        kwargs = dict(method=method, steps=steps, micro=8, deterministic=True)
        reference = train(f"bits_{method}_reference", args, args.base, **kwargs)
        failed |= report(method, compare(reference, train(f"bits_{method}_exact", args, speedups="exact", **kwargs)))
    return int(failed)


def resume(args):
    """An interrupted run must equal the continuous one, also with casts cached across micro batches."""
    failed = False
    half = (args.updates or 12)//2
    for speedups in ("exact", "all"):
        kwargs = dict(speedups=speedups, micro=8, deterministic=True)
        whole = train(f"resume_{speedups}_whole", args, steps=2*half, **kwargs)
        train(f"resume_{speedups}_parts", args, steps=half, **kwargs)
        parts = train(f"resume_{speedups}_parts", args, steps=2*half, fresh=False, extra=("--resume", "auto"), **kwargs)
        failed |= report(f"resume {speedups}", compare(whole, parts))
    return int(failed)


def numerics(args):
    """`all` changes rounding only: its loss must stay within what two `exact` runs differ by."""
    steps = args.updates or 300
    runs = {name: train(f"numerics_{name}", args, steps=steps, speedups=name.split("_")[0]) for name in ("exact_1", "all_1", "exact_2", "all_2")}
    losses = {name: [e["loss"] for e in train_events(run)] for name, run in runs.items()}
    tail = max(steps//3, 1)
    gap = lambda a, b: statistics.mean(abs(x-y) for x, y in zip(losses[a][-tail:], losses[b][-tail:]))
    for name, values in losses.items(): print(f"{name:8s} mean loss of the last {tail} updates {statistics.mean(values[-tail:]):.5f}")
    print(f"mean |loss difference| over the last {tail} updates: exact-exact {gap('exact_1', 'exact_2'):.5f}, "
          f"exact-all {gap('exact_1', 'all_1'):.5f}, {gap('exact_2', 'all_2'):.5f}, all-all {gap('all_1', 'all_2'):.5f}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("tests", "timing", "bits", "resume", "numerics"))
    parser.add_argument("--base", type=Path, default=REPO, help="checkout of the reference revision")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--updates", type=int, default=0, help="updates per run (0: the command's default)")
    parser.add_argument("--methods", default="", help="bits: comma list instead of the default set")
    args, args.trainer = parser.parse_known_args(argv)
    args.base = args.base.resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
