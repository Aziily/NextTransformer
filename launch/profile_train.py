"""Research-only profiling launcher for the reference NextTransformer trainer.

The launcher runs bounded smoke steps, keeps warmup separate from steady-state
metrics, samples NVIDIA telemetry by GPU UUID, and can wrap the trainer in
Nsight Systems. It never adds ``--formal`` or changes the training protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

from launch.profile_report import write_report


REPO_ROOT = Path(__file__).resolve().parents[1]

_TRAIN_FIELDS = (
    "step_seconds",
    "data_seconds",
    "optimizer_seconds",
    "tokens_per_second",
    "peak_memory_bytes",
)


def read_metrics(path: Path) -> list[dict]:
    """Read trainer metrics JSONL and report malformed line numbers."""
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metrics JSON at line {line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_no} is not an object")
        rows.append(row)
    return rows


def _number(row: dict, field: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"train event is missing numeric field {field!r}")
    return float(value)


def _step_values(rows: list[dict], field: str) -> list[float]:
    return [_number(row, field) for row in rows]


def summarize_metrics(rows: list[dict], warmup_steps: int) -> dict:
    """Split train events into warmup and steady windows and aggregate timing fields."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    train = [row for row in rows if row.get("event") == "train"]
    required = warmup_steps + 1
    if len(train) < required:
        raise ValueError(f"at least {required} train events are required")
    warmup = train[:warmup_steps]
    steady = train[warmup_steps:]
    for row in train:
        for field in _TRAIN_FIELDS:
            _number(row, field)

    result = {
        "warmup": {
            "steps": len(warmup),
            "step_seconds": _step_values(warmup, "step_seconds"),
        },
        "steady": {
            "steps": len(steady),
            "step_seconds": _step_values(steady, "step_seconds"),
            "median_step_seconds": statistics.median(_step_values(steady, "step_seconds")),
            "median_tokens_per_second": statistics.median(_step_values(steady, "tokens_per_second")),
            "data_seconds_total": sum(_step_values(steady, "data_seconds")),
            "optimizer_seconds_total": sum(_step_values(steady, "optimizer_seconds")),
            "peak_memory_bytes": max(int(_number(row, "peak_memory_bytes")) for row in steady),
        },
    }
    return result


def _nvidia_command(period_ms: int) -> list[str]:
    return [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
        "-lms",
        str(period_ms),
    ]


def sample_nvidia_smi(output_path: Path, stop: threading.Event, period_ms: int = 200) -> None:
    """Sample GPU UUID, memory, utilization, power, temperature, and SM clock."""
    if period_ms <= 0:
        raise ValueError("telemetry period must be positive")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for GPU telemetry")
    process = subprocess.Popen(
        _nvidia_command(period_ms),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        with output_path.open("w") as output:
            assert process.stdout is not None
            for line in process.stdout:
                if stop.is_set():
                    break
                try:
                    values = next(csv.reader([line]))
                    index, uuid = values[:2]
                    numbers = [float(value.strip()) for value in values[2:]]
                    row = dict(
                        zip(
                            ("memory_total", "memory_used", "gpu_util", "power", "temperature", "sm_clock"),
                            numbers,
                        )
                    )
                    output.write(json.dumps({"time": time.time(), "index": int(index), "uuid": uuid.strip(), **row}) + "\n")
                    output.flush()
                except (ValueError, IndexError):
                    continue
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def summarize_telemetry(path: Path, interval_start: float, interval_end: float) -> dict:
    """Aggregate samples during the steady interval by GPU UUID."""
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if interval_start <= sample["time"] <= interval_end:
            samples.append(sample)
    by_uuid: dict[str, list[dict]] = {}
    for sample in samples:
        by_uuid.setdefault(sample["uuid"], []).append(sample)
    per_gpu = []
    for uuid, rows in sorted(by_uuid.items()):
        utilization = [row["gpu_util"] for row in rows]
        memory = [row["memory_used"] for row in rows]
        total = max(row["memory_total"] for row in rows)
        per_gpu.append(
            {
                "uuid": uuid,
                "index": rows[0]["index"],
                "samples": len(rows),
                "gpu_util_mean_pct": statistics.mean(utilization),
                "gpu_util_min_pct": min(utilization),
                "gpu_util_max_pct": max(utilization),
                "memory_used_peak_mib": max(memory),
                "memory_used_peak_pct": 100 * max(memory) / total if total else 0.0,
                "power_mean_w": statistics.mean(row["power"] for row in rows),
                "sm_clock_mean_mhz": statistics.mean(row["sm_clock"] for row in rows),
            }
        )
    return {"interval_start": interval_start, "interval_end": interval_end, "per_gpu": per_gpu}


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metadata(args: argparse.Namespace, command: list[str], started: float) -> dict:
    env_keys = ("CUDA_VISIBLE_DEVICES", "NANOCHAT_DTYPE", "OMP_NUM_THREADS", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    metadata = {
        "command": command,
        "started_at": started,
        "config": str(args.config),
        "site_config": str(args.site_config),
        "world_size": args.world_size,
        "micro_batch": args.micro_batch,
        "warmup_steps": args.warmup_steps,
        "steady_steps": args.steady_steps,
        "smoke_accumulation": args.smoke_accumulation,
        "environment": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "python": sys.version,
        "torch": _command_output([sys.executable, "-c", "import torch; print(torch.__version__)" ]),
        "cuda": _command_output([sys.executable, "-c", "import torch; print(torch.version.cuda)" ]),
    }
    return metadata


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steady-steps", type=int, default=2)
    parser.add_argument("--smoke-accumulation", type=int, default=1)
    parser.add_argument("--telemetry-period-ms", type=int, default=200)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--nsys-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--formal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.formal:
        parser.error("profiling is research-only and cannot run with --formal")
    if min(args.world_size, args.micro_batch, args.warmup_steps, args.steady_steps, args.smoke_accumulation) <= 0:
        parser.error("world size, micro-batch, steps and accumulation must be positive")
    return args


def build_command(args: argparse.Namespace, run_output: Path) -> list[str]:
    """Build the bounded trainer command and optional Nsight wrapper."""
    trainer = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(args.world_size),
        "-m",
        "nexttransformer.train",
        "--site-config",
        str(args.site_config),
        "--config",
        str(args.config),
        "--output",
        str(run_output),
        "--micro-batch",
        str(args.micro_batch),
        "--smoke-steps",
        str(args.warmup_steps + args.steady_steps),
        "--smoke-accumulation",
        str(args.smoke_accumulation),
        "--log-every",
        "1",
        "--rolling-every",
        "1",
        "--wandb-mode",
        "disabled",
    ]
    if not args.nsys:
        return trainer
    if shutil.which("nsys") is None:
        raise RuntimeError("--nsys requested but nsys is unavailable")
    nsys_output = args.nsys_output or args.output / "trace"
    nsys_output.parent.mkdir(parents=True, exist_ok=True)
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-memory-usage=true",
        "--stats=true",
        "--force-overwrite=true",
        f"--output={nsys_output}",
        *trainer,
    ]


def run_profile(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"profile output is nonempty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_output = output / "run"
    if run_output.exists() and args.overwrite:
        shutil.rmtree(run_output)
    run_output.mkdir()
    telemetry_path = output / "telemetry.jsonl"
    log_path = output / "profile.log"
    command = build_command(args, run_output)
    started = time.time()
    _json_dump(output / "metadata.json", _metadata(args, command, started))
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for profiling")
    stop = threading.Event()
    telemetry = threading.Thread(target=sample_nvidia_smi, args=(telemetry_path, stop, args.telemetry_period_ms), daemon=True)
    telemetry.start()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        exit_code = process.wait()
    stopped = time.time()
    stop.set()
    telemetry.join(timeout=10)
    metrics_path = run_output / "metrics.jsonl"
    payload: dict[str, Any] = {"status": "failed" if exit_code else "passed", "exit_code": exit_code}
    if metrics_path.exists() and exit_code == 0:
        rows = read_metrics(metrics_path)
        payload["metrics_summary"] = summarize_metrics(rows, args.warmup_steps)
    else:
        payload["error"] = "trainer exited before producing metrics.jsonl"
    payload["telemetry_summary"] = summarize_telemetry(telemetry_path, started, stopped) if telemetry_path.exists() else {"per_gpu": []}
    payload["artifacts"] = {
        "run": str(run_output),
        "metrics": str(metrics_path),
        "telemetry": str(telemetry_path),
        "log": str(log_path),
    }
    for suffix in (".nsys-rep", ".sqlite"):
        candidates = list(output.glob(f"trace{suffix}")) + list(output.glob(f"trace*{suffix}"))
        if candidates:
            payload["artifacts"][suffix.removeprefix(".")] = str(candidates[0])
    summary_path = output / "summary.json"
    _json_dump(summary_path, payload)
    write_report(summary_path, output / "report.html")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    return run_profile(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
