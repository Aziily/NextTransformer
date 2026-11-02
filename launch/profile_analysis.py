"""Turn profiler stage timelines into stage, wall-clock, memory and MFU summaries and findings.

Everything here is pure Python on files the profiler wrote: per-rank stage timelines from ``launch.profile_hooks``
and the trace summary of ``launch.profile_trace``. Nothing imports torch.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

from launch.profile_trace import SMALL_KERNEL_S


EVENT_STAGES = ("validation", "core", "checkpoint")


# Stage timelines -------------------------------------------------------------------------------------------------

def _training_segments(step):
    """Segments of one update before its first evaluation or checkpoint."""
    segments = []
    for segment in step["segments"]:
        if segment["stage"] in EVENT_STAGES:
            break
        segments.append(segment)
    return segments


def _steady(record, warmup_steps):
    """Updates after warmup; updates inside a trace window are left out because tracing slows them."""
    after = [step for step in record["steps"] if step["step"] > max(warmup_steps, 0)]
    return [step for step in after if not step["profiled"]] or after


def summarize_stages(records, warmup_steps):
    """Mean per-update time of each training stage over steady updates, with the spread across ranks."""
    steady_steps = sorted({step["step"] for record in records for step in _steady(record, warmup_steps)})
    if not records or not steady_steps:
        return {"steady_steps": [], "timing": None, "step_seconds": None, "stages": []}
    device = all(segment["device_s"] is not None for record in records for step in _steady(record, warmup_steps)
                 for segment in _training_segments(step))
    key = "device_s" if device else "host_s"
    order, per_rank, host = [], defaultdict(dict), defaultdict(dict)
    for record in records:
        steps = _steady(record, warmup_steps)
        totals, host_totals = defaultdict(float), defaultdict(float)
        for step in steps:
            for segment in _training_segments(step):
                if segment["stage"] not in order:
                    order.append(segment["stage"])
                totals[segment["stage"]] += segment[key]
                host_totals[segment["stage"]] += segment["host_s"]
        for stage in totals:
            per_rank[stage][record["rank"]] = totals[stage] / len(steps)
            host[stage][record["rank"]] = host_totals[stage] / len(steps)
    rows = []
    for stage in order:
        values = per_rank[stage]
        slowest = max(values, key=values.get)
        rows.append(dict(stage=stage, seconds=statistics.mean(values.values()),
                         host_seconds=statistics.mean(host[stage].values()), rank_min=min(values.values()),
                         rank_max=values[slowest], slowest_rank=slowest,
                         per_rank=[[rank, values[rank]] for rank in sorted(values)]))
    step_seconds = sum(row["seconds"] for row in rows)
    for row in rows:
        row["share"] = row["seconds"] / step_seconds if step_seconds else 0.0
    return {"steady_steps": steady_steps, "timing": "cuda_events" if device else "host",
            "step_seconds": step_seconds, "stages": rows}


PHASES = ("startup", "warmup", "steady", "traced", *EVENT_STAGES)


def _phase(step, stage, warmup_steps):
    if stage in EVENT_STAGES:
        return stage
    if step["step"] == 0:
        return "startup"
    if step["step"] <= warmup_steps:
        return "warmup"
    return "traced" if step["profiled"] else "steady"


def wall_accounting(record, warmup_steps):
    """Host wall time of one rank split into startup, warmup, steady and traced updates, evaluation and checkpoints."""
    totals = defaultdict(float)
    for step in record["steps"]:
        for segment in step["segments"]:
            totals[_phase(step, segment["stage"], warmup_steps)] += segment["host_s"]
    return [{"phase": phase, "seconds": totals[phase]} for phase in PHASES if phase in totals]


def phase_bands(record, warmup_steps):
    """Contiguous wall-clock bands of one rank labelled by phase, for shading telemetry."""
    bands = []
    for step in record["steps"]:
        for segment in step["segments"]:
            phase = _phase(step, segment["stage"], warmup_steps)
            start, end = segment["start_wall"], segment["start_wall"] + segment["host_s"]
            if bands and bands[-1]["phase"] == phase:
                bands[-1]["end"] = end
            else:
                bands.append({"phase": phase, "start": start, "end": end})
    return bands


def steady_interval(record, warmup_steps):
    """Wall-clock start and end of the steady updates' training segments."""
    steps = [step for step in _steady(record, warmup_steps) if _training_segments(step)]
    if not steps:
        return None
    last = _training_segments(steps[-1])[-1]
    return steps[0]["segments"][0]["start_wall"], last["start_wall"] + last["host_s"]


# Hardware peak and MFU --------------------------------------------------------------------------------------------

# Dense BF16 tensor FLOPs per SM and clock with FP32 accumulation. GeForce boards of these architectures run
# FP32-accumulating tensor math at half the rate of the professional boards.
_TENSOR_FLOPS_PER_CLOCK = {(8, 0): 2048, (8, 6): 1024, (8, 7): 1024, (8, 9): 1024, (9, 0): 4096}
_HALF_RATE_ON_GEFORCE = {(8, 6), (8, 9)}
# Specification boost clocks (MHz) of known configurations, keyed by architecture, SM count and board family.
_BOOST_MHZ = {(8, 0, 108, False): 1410, (8, 6, 82, True): 1695, (8, 6, 84, True): 1860, (8, 6, 84, False): 1800,
              (8, 6, 68, True): 1710, (8, 9, 128, True): 2520, (8, 9, 142, False): 2520, (9, 0, 132, False): 1830,
              (9, 0, 114, False): 1755}


def hardware_peak(device, observed_mhz=None, override_tflops=None):
    """Dense BF16 tensor peak per GPU from its architecture, SM count and clock, with the basis of the number."""
    if override_tflops:
        return dict(dense_bf16_flops=override_tflops * 1e12, observed_clock_flops=None,
                    basis=f"--peak-tflops {override_tflops:g}")
    if not device:
        return dict(dense_bf16_flops=None, observed_clock_flops=None, basis="no device information")
    arch = (device["major"], device["minor"])
    per_clock = _TENSOR_FLOPS_PER_CLOCK.get(arch)
    if per_clock is None:
        return dict(dense_bf16_flops=None, observed_clock_flops=None,
                    basis=f"sm{arch[0]}{arch[1]} is not in the tensor-rate table; pass --peak-tflops")
    geforce = "geforce" in device.get("name", "").lower()
    if geforce and arch in _HALF_RATE_ON_GEFORCE:
        per_clock //= 2
    boost = _BOOST_MHZ.get((*arch, device["sms"], geforce))
    clock, source = (boost, "specification boost clock") if boost else (device.get("max_mhz"), "maximum SM clock")
    if not clock:
        return dict(dense_bf16_flops=None, observed_clock_flops=None, basis="unknown SM clock; pass --peak-tflops")
    peak = device["sms"] * per_clock * clock * 1e6
    return dict(dense_bf16_flops=peak,
                observed_clock_flops=device["sms"] * per_clock * observed_mhz * 1e6 if observed_mhz else None,
                basis=f"sm{arch[0]}{arch[1]}: {device['sms']} SMs × {per_clock} FLOP/clock × {clock:.0f} MHz ({source})")


def phase_flops(proof, token_share, charged_total=None):
    """Audited FLOPs of one update split by phase; forward and backward scale with the update's share of the formal
    tokens, the optimizer runs once per update. A proof with several training stages needs ``charged_total``, the
    total the trainer charged for a formal update, to pick the stage that ran; otherwise ``None``."""
    stages = [stage for stage in proof.get("stages") or [] if "cost" in stage]
    if len(stages) > 1 and charged_total is not None:
        stages = [min(stages, key=lambda stage: abs(stage["cost"]["total"] - charged_total))]
    if len(stages) != 1:
        return None
    cost = stages[0]["cost"]
    phases = cost["phases"]
    matrix, attention = defaultdict(float), defaultdict(float)
    for name, operator in cost.get("operators", {}).items():
        phase, op = name.split("/", 1)
        matrix[phase] += operator.get("matrix", 0)
        if "attention" in op:
            attention[phase] += operator.get("theory", operator.get("matrix", 0))
    result = dict(forward=phases["forward"] * token_share, backward=phases["backward"] * token_share,
                  optimizer=float(phases.get("optimizer", 0)))
    result["total"] = result["forward"] + result["backward"] + result["optimizer"]
    result["matrix"] = (matrix["forward"] + matrix["backward"]) * token_share + matrix["optimizer"]
    result["attention"] = (attention["forward"] + attention["backward"]) * token_share
    return result


def mfu_summary(flops, world, step_seconds, stages, trace, peak, clock_peak=None):
    """Model FLOPs utilization of the update and its phases, matrix-kernel efficiency, and where time goes beyond the
    ideal update at peak (from the traced GPU activity)."""
    if not flops or not step_seconds:
        return None
    achieved = flops["total"] / world / step_seconds
    result = dict(achieved_flops=achieved, mfu=achieved / peak if peak else None, peak_flops=peak,
                  clock_peak_flops=clock_peak, clock_mfu=achieved / clock_peak if clock_peak else None, phases=[],
                  kernels=[], waterfall=[])
    rows = {row["stage"]: row["seconds"] for row in stages.get("stages", [])}
    timed = dict(forward=rows.get("forward", 0.0), backward=rows.get("backward", 0.0),
                 optimizer=sum(value for name, value in rows.items() if name.startswith("optimizer.")))
    for phase, seconds in timed.items():
        rate = flops[phase] / world / seconds if seconds else None
        result["phases"].append(dict(phase=phase, seconds=seconds, flops=flops[phase] / world, achieved_flops=rate,
                                     mfu=rate / peak if rate and peak else None))
    other = (stages.get("step_seconds") or step_seconds) - sum(timed.values())
    result["phases"].append(dict(phase="rest of the update", seconds=other, flops=0.0, achieved_flops=None, mfu=None))
    gpus = (trace or {}).get("per_gpu") or []
    if not gpus:
        return result
    traced = max((trace or {}).get("steps", 1), 1)
    mean = {key: statistics.mean(row[key] for row in gpus) / traced for key in
            ("window_s", "compute_s", "exposed_communication_s", "exposed_memory_s", "idle_s", "gemm_s", "attention_s")}
    measured, sources = {}, {}
    for kind in ("gemm", "attention"):
        values = [row.get(f"{kind}_flops") for row in gpus]
        measured[kind] = statistics.mean(values) / traced if all(value is not None for value in values) else None
        sources[kind] = "trace"
        # Custom or masked attention kernels carry no shapes in the trace; the audit counts every attention layer.
        if kind == "attention" and flops.get("attention"):
            measured[kind], sources[kind] = flops["attention"] / world, "audit"
        rate = measured[kind] / mean[f"{kind}_s"] if measured[kind] and mean[f"{kind}_s"] else None
        result["kernels"].append(dict(kind=kind, seconds=mean[f"{kind}_s"], flops=measured[kind],
                                      flops_source=sources[kind], tflops=rate / 1e12 if rate else None,
                                      efficiency=rate / peak if rate and peak else None,
                                      clock_efficiency=rate / clock_peak if rate and clock_peak else None))
    if not peak:
        return result
    parts = [("ideal", flops["total"] / world / peak)]
    matrix_s = mean["gemm_s"] + mean["attention_s"]
    if measured["gemm"] is not None and measured["attention"] is not None:
        matrix_flops = measured["gemm"] + measured["attention"]
        parts += [("gemm below peak", mean["gemm_s"] - measured["gemm"] / peak),
                  ("attention below peak", mean["attention_s"] - measured["attention"] / peak)]
    else:
        matrix_flops = flops["matrix"] / world
        parts.append(("matrix kernels below peak", matrix_s - matrix_flops / peak))
    parts += [("other compute kernels", mean["compute_s"] - matrix_s - (flops["total"] / world - matrix_flops) / peak),
              ("exposed communication", mean["exposed_communication_s"]),
              ("exposed memory copies", mean["exposed_memory_s"]), ("idle", mean["idle_s"])]
    window = mean["window_s"] or 1.0
    result["waterfall"] = [dict(part=part, seconds=seconds, share=seconds / window) for part, seconds in parts]
    categories = (trace or {}).get("categories") or {}
    other = [dict(category=name, seconds=seconds) for name, seconds in categories.items()
             if name not in ("gemm", "attention", "communication", "memory") and seconds > 0]
    for row in result["waterfall"]:
        if row["part"] == "other compute kernels" and other:
            row["detail"] = sorted(other, key=lambda item: item["seconds"], reverse=True)
    result["traced_window_s"] = mean["window_s"]
    return result


def memory_summary(records, warmup_steps, total_bytes):
    """Peak allocated and reserved memory of each rank over steady updates, and the smallest headroom."""
    per_rank = []
    for record in records:
        values = [step["memory"] for step in _steady(record, warmup_steps) if step.get("memory")]
        if values:
            per_rank.append(dict(rank=record["rank"], allocated_peak=max(value["allocated_peak"] for value in values),
                                 reserved_peak=max(value["reserved_peak"] for value in values)))
    if not per_rank:
        return None
    reserved = max(row["reserved_peak"] for row in per_rank)
    return dict(per_rank=per_rank, total_bytes=total_bytes, allocated_peak=max(row["allocated_peak"] for row in per_rank),
                reserved_peak=reserved, headroom_bytes=total_bytes - reserved if total_bytes else None,
                reserved_share=reserved / total_bytes if total_bytes else None)


# Projection and diagnosis -----------------------------------------------------------------------------------------

# The trainer's default --rolling-every; protocol checkpoints (every 5,000 updates) fall on the same steps.
FORMAL_ROLLING_EVERY = 500
# Validation is extrapolated only from a measured validation of at least this share of the formal one.
MIN_VALIDATION_SHARE = 1 / 8


def project_formal(protocol, micro_batch, accumulation, world, step_seconds, validation=None, checkpoint_seconds=None):
    """Formal wall-clock implied by measured updates, only when the update shape is the formal one.

    ``validation`` is the last smoke validation event (``seconds``, ``input_tokens``); when it covers at least
    ``MIN_VALIDATION_SHARE`` of the formal validation, its token rate is scaled to the formal size.
    ``checkpoint_seconds`` is one measured checkpoint. CORE is never estimated.
    """
    parallel = protocol["data_parallel_size"]
    formal = protocol["global_batch_tokens"] // (parallel * micro_batch * protocol["sequence_len"])
    updates = protocol["total_tokens"] // protocol["global_batch_tokens"]
    result = dict(representative=world == parallel and accumulation == formal, accumulation=accumulation,
                  formal_accumulation=formal, micro_batch=micro_batch, world=world, updates=updates)
    if not result["representative"] or not step_seconds:
        return result
    result["training_hours"] = updates * step_seconds / 3600
    if "val_every" in protocol and "val_tokens" in protocol:
        result["validation_events"] = len({0, *range(protocol["val_every"], updates, protocol["val_every"]), updates})
        measured = (validation or {}).get("input_tokens", 0)
        result["validation_measured_tokens"] = measured
        # Smaller validations are dominated by fixed setup time and would not scale linearly.
        if measured >= protocol["val_tokens"] * MIN_VALIDATION_SHARE:
            per_event = validation["seconds"] * protocol["val_tokens"] / measured
            result["validation_hours"] = result["validation_events"] * per_event / 3600
    if checkpoint_seconds:
        count = len({*range(FORMAL_ROLLING_EVERY, updates, FORMAL_ROLLING_EVERY), updates})
        result.update(checkpoints=count, checkpoint_hours=count * checkpoint_seconds / 3600,
                      checkpoint_interval=FORMAL_ROLLING_EVERY)
    result["estimated_hours"] = sum(result.get(key, 0.0) for key in
                                    ("training_hours", "validation_hours", "checkpoint_hours"))
    return result


def _finding(identifier, severity, title, detail):
    return dict(id=identifier, severity=severity, title=title, detail=detail)


def diagnose(stages, trace, projection, accounting, telemetry, peak_memory_bytes, memory_total_bytes, mfu=None):
    """Ordered findings a reader should check first; each names the evidence it comes from."""
    findings = []
    waterfall = {row["part"]: row for row in (mfu or {}).get("waterfall", [])}
    losses = [row for name, row in waterfall.items() if name != "ideal" and row["share"] > 0.08]
    if losses:
        worst = max(losses, key=lambda row: row["seconds"])
        findings.append(_finding(
            "largest-mfu-loss", "warning", f"Largest MFU loss: {worst['part']}",
            f"{worst['part'].capitalize()} costs {worst['seconds']:.2f} s ({worst['share']:.0%}) of the traced update "
            f"beyond the ideal {waterfall['ideal']['seconds']:.2f} s at peak; MFU is {mfu['mfu']:.0%}."))
    if projection and not projection["representative"]:
        findings.append(_finding(
            "unrepresentative-update", "warning", "Update shape differs from formal training",
            f"This profile ran {projection['accumulation']} micro-batch(es) per update; formal training runs "
            f"{projection['formal_accumulation']} on {projection.get('world', 8)} ranks. Gradient communication and the "
            "optimizer are paid once per update, so their share here overstates formal training. Profile with "
            f"--smoke-accumulation {projection['formal_accumulation']} to estimate formal throughput."))
    gpus = (trace or {}).get("per_gpu", [])
    if gpus:
        gpu = _mean_gpu(gpus)
        exposed = gpu["exposed_communication_s"] / gpu["window_s"] if gpu["window_s"] else 0.0
        if exposed > 0.05:
            rows = sorted(trace.get("communication") or [], key=lambda row: row["exposed_s"], reverse=True)[:2]
            where = "; ".join(f"{row['collective']} {row['dtype']} in {row['stage']} {row['exposed_s']:.2f} s" for row in rows)
            findings.append(_finding(
                "exposed-communication", "warning" if exposed > 0.2 else "info",
                "Communication is not hidden behind compute",
                f"{exposed:.0%} of the traced window runs communication kernels with no compute kernel alongside "
                f"({gpu['exposed_communication_s']:.2f} s per update{'; largest: ' + where if where else ''})."))
        if gpu["window_s"] and gpu["idle_s"] / gpu["window_s"] > 0.15:
            findings.append(_finding(
                "gpu-idle", "warning", "GPUs are idle inside the traced updates",
                f"No kernel runs for {gpu['idle_s'] / gpu['window_s']:.0%} of the traced window: host launch "
                "overhead, data waits or synchronization."))
    rows = {row["stage"]: row for row in stages.get("stages", [])}
    step = stages.get("step_seconds") or 0.0
    if "data" in rows and rows["data"]["share"] > 0.05:
        findings.append(_finding("data-stall", "warning", "Input pipeline takes a visible share of each update",
                                 f"Data loading takes {rows['data']['share']:.0%} of the update."))
    skewed = [row for row in rows.values() if step and (row["rank_max"] - row["rank_min"]) / step > 0.05]
    if skewed:
        worst = max(skewed, key=lambda row: row["rank_max"] - row["rank_min"])
        findings.append(_finding(
            "rank-skew", "warning" if (worst["rank_max"] - worst["rank_min"]) / step > 0.1 else "info",
            "Ranks spend different time in the same stage",
            f"{worst['stage']} ranges from {worst['rank_min']:.3f} s to {worst['rank_max']:.3f} s per update across "
            f"ranks (slowest: rank {worst['slowest_rank']}); the others wait at the next collective."))
    optimizer = sum(row["share"] for name, row in rows.items() if name.startswith("optimizer."))
    if optimizer > 0.25:
        findings.append(_finding("optimizer-heavy", "info", "The optimizer step is a large share of the update",
                                 f"Optimizer phases take {optimizer:.0%} of the update."))
    syncs = [row for row in (trace or {}).get("syncs", []) if row["seconds"] > 0.01]
    micro = ((trace or {}).get("launch") or {}).get("micro_batches") or 0
    looped = [row for row in syncs if micro and row["calls"] >= micro and row["stage"] in ("forward", "backward", "data")]
    if looped:
        # A blocking call per micro-batch drains the launch queue, so its cost is the host-late idle time it allows.
        row = max(looped, key=lambda row: row["seconds"])
        host_idle = ((trace or {}).get("idle") or {}).get("host_s") or 0.0
        share = host_idle / step if step else 0.0
        findings.append(_finding(
            "host-sync", "warning" if share > 0.03 else "info", "The host waits for the GPU in every micro-batch",
            f"{row['api']} from {row['source']} runs {row['calls']:.0f} times per update during {row['stage']}, so the host "
            f"never queues work ahead of the GPU ({row['seconds']:.2f} s blocked per update). GPUs wait for a late host "
            f"{host_idle:.2f} s per update ({share:.1%}); removing the sync would mostly save that."))
    matrix = [row for row in (trace or {}).get("gemm", []) if row.get("tflops")]
    if mfu and mfu.get("peak_flops") and len(matrix) > 2:
        median = statistics.median(row["tflops"] for row in matrix)
        lost = [(row["seconds"] - row["flops"] / (median * 1e12), row) for row in matrix if row["tflops"] < 0.8 * median]
        if lost:
            seconds, row = max(lost, key=lambda pair: pair[0])
            if seconds > 0.01 * (stages.get("step_seconds") or 1):
                findings.append(_finding(
                    "slow-matrix-shape", "warning", "One matrix shape runs well below the others",
                    f"{row['op']} {row['shape']} in {row['stage']} runs at {row['tflops']:.0f} TFLOP/s, against a median of "
                    f"{median:.0f} over all matrix shapes; at the median it would save {seconds:.2f} s per update "
                    f"({row['calls']:.0f} calls, kernel {row['kernel']})."))
    small = (trace or {}).get("small_kernels")
    if small and small["count_share"] > 0.5:
        findings.append(_finding(
            "small-kernels", "info", "Many kernels are very short",
            f"{small['count_share']:.0%} of kernels run under {SMALL_KERNEL_S * 1e6:.0f} µs "
            f"({small['time_share']:.0%} of kernel time); launch overhead can dominate when the host falls behind."))
    total = sum(row["seconds"] for row in accounting)
    steady = sum(row["seconds"] for row in accounting if row["phase"] == "steady")
    if total and steady / total < 0.5:
        findings.append(_finding(
            "non-training-wall", "info", "Most wall time is not steady training",
            f"Steady updates are {steady / total:.0%} of the measured wall time; the rest is startup, warmup, "
            "evaluation and checkpoints. Use steady-state numbers for throughput."))
    per_gpu = telemetry.get("per_gpu", []) if telemetry else []
    utilization = {row["index"]: row["gpu_util_mean_pct"] for row in per_gpu if "gpu_util_mean_pct" in row}
    lagging = lagging_gpus(utilization)
    if lagging:
        median = statistics.median(utilization.values())
        findings.append(_finding(
            "lagging-gpu", "warning", "Low utilization on some GPUs",
            ", ".join(f"GPU {index} averaged {utilization[index]:.0f}%" for index in lagging)
            + f" versus a median of {median:.0f}% during the {telemetry.get('per_gpu_scope', 'captured')} interval."))
    if peak_memory_bytes and memory_total_bytes and peak_memory_bytes / memory_total_bytes > 0.9:
        findings.append(_finding("memory-pressure", "warning", "GPU memory is nearly full",
                                 f"Peak memory reached {peak_memory_bytes / memory_total_bytes:.0%} of device memory."))
    severity = {"warning": 0, "info": 1}
    return sorted(findings, key=lambda finding: severity[finding["severity"]])


def lagging_gpus(utilization):
    """GPUs whose mean utilization is under half the median and at least 15 points below it."""
    if len(utilization) < 2:
        return []
    median = statistics.median(utilization.values())
    return sorted(index for index, value in utilization.items() if value < 0.5 * median and median - value >= 15)


def _mean_gpu(rows):
    keys = ("window_s", "compute_s", "communication_s", "overlap_s", "memory_s", "idle_s", "exposed_communication_s")
    return {key: statistics.mean(row[key] for row in rows) for key in keys}


def read_timelines(directory):
    """Per-rank stage records written by ``launch.profile_hooks``, sorted by rank."""
    records = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("rank_*.json"))]
    return sorted(records, key=lambda record: record["rank"])
