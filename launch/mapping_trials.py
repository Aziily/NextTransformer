"""Run bounded dense_full trials under explicit GPU mappings.

This is a research-only experiment harness. It changes only process-to-device
order and selected NCCL environment variables; it never launches a formal run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def mapping_order(preset: str, world_size: int) -> tuple[int, ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if preset == "islands":
        order = tuple(range(world_size))
    elif preset == "interleaved":
        half = world_size // 2
        order = tuple(i for pair in zip(range(half), range(half, world_size)) for i in pair)
    else:
        raise ValueError(f"unknown mapping preset: {preset}")
    if len(order) != world_size or sorted(order) != list(range(world_size)):
        raise ValueError("mapping preset does not cover every visible device once")
    return order


def build_command(*, repo: Path, site_config: Path, config: Path, output: Path,
                  world_size: int, micro_batch: int, smoke_steps: int,
                  smoke_accumulation: int) -> list[str]:
    return [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(world_size),
        "-m",
        "nexttransformer.train",
        "--site-config",
        str(site_config),
        "--config",
        str(config),
        "--output",
        str(output),
        "--micro-batch",
        str(micro_batch),
        "--smoke-steps",
        str(smoke_steps),
        "--smoke-accumulation",
        str(smoke_accumulation),
        "--log-every",
        "1",
        "--rolling-every",
        "1",
        "--wandb-mode",
        "disabled",
    ]


def _run_trial(args: argparse.Namespace, preset: str) -> dict:
    order = mapping_order(preset, args.world_size)
    trial_dir = args.output / preset
    trial_dir.mkdir(parents=True, exist_ok=True)
    run_dir = trial_dir / "run"
    command = build_command(
        repo=args.repo,
        site_config=args.site_config,
        config=args.config,
        output=run_dir,
        world_size=args.world_size,
        micro_batch=args.micro_batch,
        smoke_steps=args.smoke_steps,
        smoke_accumulation=args.smoke_accumulation,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, order))
    environment.update(args.env)
    started = time.time()
    with (trial_dir / "process.log").open("w") as log:
        process = subprocess.run(command, cwd=args.repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"phase": "missing"}
    metrics_path = run_dir / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()] if metrics_path.exists() else []
    train = [row for row in metrics if row.get("event") == "train"]
    steady = train[args.warmup_steps:]
    result = {
        "preset": preset,
        "cuda_visible_devices": order,
        "exit_code": process.returncode,
        "status": status,
        "elapsed_seconds": time.time() - started,
        "steady_steps": len(steady),
        "median_step_seconds": sorted(row["step_seconds"] for row in steady)[len(steady) // 2] if steady else None,
        "median_tokens_per_second": sorted(row["tokens_per_second"] for row in steady)[len(steady) // 2] if steady else None,
        "command": command,
    }
    (trial_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--smoke-accumulation", type=int, default=1)
    parser.add_argument("--presets", nargs="+", choices=("islands", "interleaved"), default=("islands", "interleaved"))
    parser.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE environment entries")
    args = parser.parse_args(argv)
    args.env = dict(item.split("=", 1) for item in args.env)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output is nonempty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [_run_trial(args, preset) for preset in args.presets]
    (args.output / "result.json").write_text(json.dumps({"phase": "complete", "trials": results}, indent=2) + "\n")
    return 0 if all(row["exit_code"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
