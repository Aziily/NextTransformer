"""Research-only profiling launcher for the reference NextTransformer trainer.

The launcher runs bounded smoke updates through ``launch.profile_hooks``, keeps warmup separate from steady-state
metrics, times every training stage with CUDA events, samples NVIDIA telemetry by GPU UUID, and can trace extra
updates after the steady ones with the PyTorch profiler or Nsight Systems. Smoke updates default to the formal
update shape for the requested micro-batch. It never adds ``--formal`` or changes the training protocol.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    # Run as a script: import this checkout, not another copy installed in the environment.
    sys.path.insert(0, str(REPO_ROOT))

from launch.profile_analysis import (diagnose, hardware_peak, memory_summary, mfu_summary, phase_bands, phase_flops,
                                     project_formal, read_timelines, steady_interval, summarize_stages, wall_accounting)
from launch.profile_trace import analyze_nsys_sqlite, analyze_torch_traces
from launch.profile_report import write_report


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


def summarize_metrics(rows: list[dict], warmup_steps: int, traced_steps: list[int] | None = None) -> dict:
    """Split train events into warmup, steady and traced windows and aggregate timing fields."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    train = [row for row in rows if row.get("event") == "train"]
    required = warmup_steps + 1
    if len(train) < required:
        raise ValueError(f"at least {required} train events are required")
    traced_steps = set(traced_steps or ())
    warmup = train[:warmup_steps]
    steady = [row for row in train[warmup_steps:] if row.get("step") not in traced_steps]
    traced = [row for row in train[warmup_steps:] if row.get("step") in traced_steps]
    if not steady:
        raise ValueError("no steady train event outside the traced updates")
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
    if traced:
        result["traced"] = {"steps": len(traced), "step_seconds": _step_values(traced, "step_seconds")}
    return result


def _nvidia_command(period_ms: int) -> list[str]:
    return [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,"
        "clocks.max.sm",
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
                            ("memory_total", "memory_used", "gpu_util", "power", "temperature", "sm_clock",
                             "sm_clock_max"),
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


def summarize_telemetry(path: Path, interval_start: float, interval_end: float,
                        steady: tuple[float, float] | None = None) -> dict:
    """Keep every sample of the profile interval for the heatmap; aggregate per GPU UUID over ``steady`` if given."""
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if interval_start <= sample["time"] <= interval_end:
            samples.append(sample)
    scoped = [sample for sample in samples if steady[0] <= sample["time"] <= steady[1]] if steady else samples
    by_uuid: dict[str, list[dict]] = {}
    for sample in scoped:
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
                "memory_total_mib": total,
                "power_mean_w": statistics.mean(row["power"] for row in rows),
                "sm_clock_mean_mhz": statistics.mean(row["sm_clock"] for row in rows),
                "sm_clock_max_mhz": max((row.get("sm_clock_max", 0) for row in rows), default=0) or None,
            }
        )
    heatmap_samples = [
        {
            "time_offset_sec": round(sample["time"] - interval_start, 3),
            "index": sample["index"],
            "uuid": sample["uuid"],
            "gpu_util": sample["gpu_util"],
            "memory_used_pct": 100 * sample["memory_used"] / sample["memory_total"] if sample["memory_total"] else None,
            "power": sample["power"],
        }
        for sample in sorted(samples, key=lambda row: (row["time"], row["index"]))
    ]
    result = {
        "interval_start": interval_start,
        "interval_end": interval_end,
        "per_gpu_scope": "steady" if steady else "captured",
        "per_gpu": per_gpu,
        "samples": heatmap_samples,
    }
    if steady:
        result["steady_offsets"] = [round(steady[0] - interval_start, 3), round(steady[1] - interval_start, 3)]
    return result


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metadata(args: argparse.Namespace, command: list[str], started: float) -> dict:
    env_keys = ("CUDA_VISIBLE_DEVICES", "NANOCHAT_DTYPE", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
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
        "trace": args.trace,
        "trace_steps": list(trace_window(args) or ()),
        "environment": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
        "git_commit": _command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
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
    parser.add_argument("--smoke-accumulation", type=int,
                        help="micro-batches per update (default: the formal update for --micro-batch)")
    parser.add_argument("--validation-tokens", type=int,
                        help="tokens of each smoke validation (default: one batch; the formal size is 41943040)")
    parser.add_argument("--trace", choices=("none", "torch", "nsys"), default="none",
                        help="trace extra updates after the steady ones (torch: operator and kernel tree of rank 0; "
                             "nsys: kernels of every rank)")
    parser.add_argument("--trace-steps", type=int, default=1, help="updates to trace after the steady ones")
    parser.add_argument("--trace-ranks", type=int, nargs="+", default=[0], help="ranks the torch trace records")
    parser.add_argument("--nsys", action="store_true", help="same as --trace nsys")
    parser.add_argument("--nsys-output", type=Path)
    parser.add_argument("--peak-tflops", type=float,
                        help="dense BF16 tensor peak per GPU for MFU (default: from the GPU architecture and clock)")
    parser.add_argument("--telemetry-period-ms", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--formal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.formal:
        parser.error("profiling is research-only and cannot run with --formal")
    if args.nsys:
        args.trace = "nsys"
    counts = (args.world_size, args.micro_batch, args.warmup_steps, args.steady_steps, args.trace_steps)
    optional = (args.smoke_accumulation, args.validation_tokens)
    if min(counts) <= 0 or any(value is not None and value <= 0 for value in optional):
        parser.error("world size, micro-batch, steps and accumulation must be positive")
    return args


def trace_window(args: argparse.Namespace) -> tuple[int, int] | None:
    """First and last traced update: they follow the steady updates, so tracing never slows a measured one."""
    if args.trace == "none":
        return None
    first = args.warmup_steps + args.steady_steps + 1
    return first, first + args.trace_steps - 1


def build_command(args: argparse.Namespace, run_output: Path) -> list[str]:
    """Build the bounded trainer command and optional Nsight wrapper."""
    window = trace_window(args)
    steps = window[1] if window else args.warmup_steps + args.steady_steps
    hooks = ["--profile-dir", str(args.output / "stages")]
    if window:
        hooks += ["--profile-trace", args.trace, "--profile-trace-steps", str(window[0]), str(window[1])]
        if args.trace == "torch":
            hooks += ["--profile-trace-ranks", *map(str, args.trace_ranks)]
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
        "launch.profile_hooks",
        *hooks,
        "--site-config",
        str(args.site_config),
        "--config",
        str(args.config),
        "--output",
        str(run_output),
        "--micro-batch",
        str(args.micro_batch),
        "--smoke-steps",
        str(steps),
        *(["--smoke-accumulation", str(args.smoke_accumulation)] if args.smoke_accumulation else []),
        *(["--smoke-val-tokens", str(args.validation_tokens)] if args.validation_tokens else []),
        "--log-every",
        "1",
        # Only the final checkpoint: a rolling checkpoint after every update would dominate the wall time.
        "--rolling-every",
        str(steps + 1),
        "--wandb-mode",
        "disabled",
    ]
    if args.trace != "nsys":
        return trainer
    if shutil.which("nsys") is None:
        raise RuntimeError("--trace nsys requested but nsys is unavailable")
    nsys_output = args.nsys_output or args.output / "trace"
    nsys_output.parent.mkdir(parents=True, exist_ok=True)
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        *_nsys_event_option(),
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--export=sqlite",
        "--force-overwrite=true",
        f"--output={nsys_output}",
        *trainer,
    ]


def _nsys_event_option() -> list[str]:
    """Disable device-side CUDA event tracing where nsys supports it: the stage timer records CUDA events, and the
    tracing adds overhead and false cross-stream dependencies."""
    help_text = _command_output(["nsys", "profile", "--help"]) or ""
    return ["--cuda-event-trace=false"] if "--cuda-event-trace" in help_text else []


def model_flops_rate(rows: list[dict], steps: list[int], world: int) -> float | None:
    """Median audited model FLOPs per second and GPU over ``steps``, from the trainer's cumulative charge."""
    train = [row for row in rows if row.get("event") == "train" and "theory_flops" in row]
    rates, previous = [], 0
    for row in train:
        if row["step"] in steps and row.get("step_seconds"):
            rates.append((row["theory_flops"] - previous) / row["step_seconds"] / world)
        previous = row["theory_flops"]
    return statistics.median(rates) if rates else None


def _attempt(errors: dict, name: str, function, *args):
    """Run one analysis; a failure is recorded in the summary instead of losing the profile."""
    try:
        return function(*args)
    except Exception as exc:  # noqa: BLE001 - every analysis failure is reported, none is fatal
        errors[name] = f"{type(exc).__name__}: {exc}"
        return None


def _cost_proof(config: Path) -> dict:
    """The cost proof a configuration's training charges, from its calibration record."""
    path = config if config.is_absolute() else REPO_ROOT / config
    record = json.loads(path.read_text())["calibration"]["cost_record"]
    return json.loads((path.parent / record).read_text())


def collect(args: argparse.Namespace, output: Path, exit_code: int, started: float, stopped: float) -> dict:
    """Summarize a finished profile directory."""
    run_output, stages_dir = output / "run", output / "stages"
    metrics_path, telemetry_path = run_output / "metrics.jsonl", output / "telemetry.jsonl"
    window = trace_window(args)
    traced = list(range(window[0], window[1] + 1)) if window else []
    payload: dict[str, Any] = {"status": "failed" if exit_code else "passed", "exit_code": exit_code}
    errors: dict[str, str] = {}
    rows = _attempt(errors, "metrics", read_metrics, metrics_path) if metrics_path.exists() else None
    if rows is None:
        payload["error"] = "trainer exited before producing metrics.jsonl"
    elif exit_code == 0:
        payload["metrics_summary"] = _attempt(errors, "metrics", summarize_metrics, rows, args.warmup_steps, traced)
    records = read_timelines(stages_dir) if stages_dir.exists() else []
    provenance = json.loads((run_output / "provenance.json").read_text()) \
        if (run_output / "provenance.json").exists() else {}
    interval = None
    if records:
        payload["stage_summary"] = _attempt(errors, "stages", summarize_stages, records, args.warmup_steps)
        payload["wall_accounting"] = _attempt(errors, "wall", wall_accounting, records[0], args.warmup_steps)
        interval = steady_interval(records[0], args.warmup_steps)
    if telemetry_path.exists():
        telemetry = _attempt(errors, "telemetry", summarize_telemetry, telemetry_path, started, stopped, interval)
        if telemetry is not None and records:
            telemetry["phase_bands"] = [dict(phase=band["phase"], start=round(band["start"] - started, 3),
                                             end=round(band["end"] - started, 3))
                                        for band in phase_bands(records[0], args.warmup_steps)]
        payload["telemetry_summary"] = telemetry or {"per_gpu": []}
    else:
        payload["telemetry_summary"] = {"per_gpu": []}
    steady = (payload.get("metrics_summary") or {}).get("steady", {})
    if provenance:
        steps = [row["step"] for row in rows or [] if row.get("event") == "train"][args.warmup_steps:]
        payload["shape"] = dict(world=provenance["world_size"], micro_batch=provenance["micro_batch"],
                                accumulation=provenance["accumulation"],
                                tokens_per_update=provenance["world_size"] * provenance["micro_batch"]
                                * provenance["accumulation"] * provenance["protocol"]["sequence_len"],
                                model_flops_per_gpu_s=model_flops_rate(rows or [], [s for s in steps if s not in traced],
                                                                       provenance["world_size"]))
        validations = [row for row in rows or [] if row.get("event") == "validation"]
        checkpoints = [segment["host_s"] for step in records[0]["steps"] for segment in step["segments"]
                       if segment["stage"] == "checkpoint"] if records else []
        payload["projection"] = _attempt(errors, "projection", project_formal, provenance["protocol"],
                                         provenance["micro_batch"], provenance["accumulation"],
                                         provenance["world_size"], steady.get("median_step_seconds"),
                                         validations[-1] if validations else None,
                                         max(checkpoints) if checkpoints else None)
    if window and exit_code == 0:
        if args.trace == "torch":
            traces = sorted((stages_dir / "torch").glob("rank_*.json.gz"))
            payload["trace_summary"] = _attempt(errors, "trace", analyze_torch_traces, traces, args.trace_steps) \
                if traces else None
        else:
            sqlite = (args.nsys_output or output / "trace").with_suffix(".sqlite")
            payload["trace_summary"] = _attempt(errors, "trace", analyze_nsys_sqlite, sqlite, args.trace_steps) \
                if sqlite.exists() else None
        if payload["trace_summary"] is None and "trace" not in errors:
            errors["trace"] = "no trace file was written"
    device = next((record["device"] for record in records if record.get("device")), None)
    if device is not None:
        device = dict(device, max_mhz=max((row.get("sm_clock_max_mhz") or 0 for row in
                                           payload["telemetry_summary"]["per_gpu"]), default=0) or None)
    clocks = [row["sm_clock_mean_mhz"] for row in payload["telemetry_summary"]["per_gpu"] if row.get("sm_clock_mean_mhz")]
    payload["hardware"] = dict(device=device, **hardware_peak(device, statistics.mean(clocks) if clocks else None,
                                                             args.peak_tflops))
    memory_total = (device or {}).get("total_memory") or max(
        (row.get("memory_total_mib", 0) for row in payload["telemetry_summary"]["per_gpu"]), default=0) * 2**20 or None
    if records:
        payload["memory_summary"] = _attempt(errors, "memory", memory_summary, records, args.warmup_steps, memory_total)
    shape = payload.get("shape")
    proof = _attempt(errors, "cost proof", _cost_proof, args.config) if shape else None
    if proof and shape:
        token_share = shape["tokens_per_update"] / provenance["protocol"]["global_batch_tokens"]
        flops = phase_flops(proof, token_share)
        payload["mfu_summary"] = _attempt(errors, "mfu", mfu_summary, flops, shape["world"],
                                          steady.get("median_step_seconds"), payload.get("stage_summary") or {},
                                          payload.get("trace_summary"), payload["hardware"]["dense_bf16_flops"],
                                          payload["hardware"]["observed_clock_flops"])
    payload["findings"] = _attempt(
        errors, "findings", diagnose, payload.get("stage_summary") or {}, payload.get("trace_summary"),
        payload.get("projection"), payload.get("wall_accounting") or [], payload["telemetry_summary"],
        (payload.get("memory_summary") or {}).get("reserved_peak") or steady.get("peak_memory_bytes"), memory_total,
        payload.get("mfu_summary")) or []
    payload["artifacts"] = {
        "run": str(run_output),
        "metrics": str(metrics_path),
        "stages": str(stages_dir),
        "telemetry": str(telemetry_path),
        "log": str(output / "profile.log"),
    }
    for path in sorted((stages_dir / "torch").glob("rank_*.json.gz")) if stages_dir.exists() else []:
        payload["artifacts"].setdefault("torch_traces", []).append(str(path))
    for suffix in (".nsys-rep", ".sqlite"):
        candidates = list(output.glob(f"trace{suffix}")) + list(output.glob(f"trace*{suffix}"))
        if candidates:
            payload["artifacts"][suffix.removeprefix(".")] = str(candidates[0])
    if errors:
        payload["analysis_errors"] = errors
    return payload


def run_profile(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    args.output = output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"profile output is nonempty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for stale in (output / "run", output / "stages"):
        if stale.exists() and args.overwrite:
            shutil.rmtree(stale)
    (output / "run").mkdir()
    telemetry_path = output / "telemetry.jsonl"
    log_path = output / "profile.log"
    command = build_command(args, output / "run")
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
    summary_path = output / "summary.json"
    _json_dump(summary_path, collect(args, output, exit_code, started, stopped))
    write_report(summary_path, output / "report.html")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    return run_profile(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
