"""Turn profiler stage timelines and traces into stage, kernel and wall-clock breakdowns.

Everything here is pure Python on files the profiler wrote: per-rank stage timelines from ``launch.profile_hooks``,
PyTorch profiler (Kineto) traces, and the SQLite export of an Nsight Systems report. Nothing imports torch.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
import statistics


EVENT_STAGES = ("validation", "core", "checkpoint")
STAGE_PREFIX = "nt.stage:"
SMALL_KERNEL_S = 10e-6
TREE_OPERATORS = 15
TREE_KERNELS = 8


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


# Kernel classification --------------------------------------------------------------------------------------------

_CATEGORIES = (
    ("communication", ("nccl",)),
    ("memory", ("memcpy", "memset")),
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("gemm", ("gemm", "cutlass", "xmma", "cublas", "matmul")),
    ("fused", ("triton",)),
    ("optimizer", ("multi_tensor_apply", "foreach", "adam", "muon")),
    ("normalization", ("norm",)),
    ("softmax", ("softmax",)),
    ("reduction", ("reduce", "reduction")),
    ("elementwise", ("elementwise", "unrolled", "vectorized", "fill", "pointwise")),
    ("copy", ("cat", "copy", "index", "gather", "scatter", "embedding", "transpose")),
)


def classify_kernel(name):
    """Coarse GPU activity category from a kernel or memory operation name."""
    lowered = name.lower()
    for category, needles in _CATEGORIES:
        if any(needle in lowered for needle in needles):
            return category
    return "other"


def _template_args(text):
    """Top-level template arguments of ``text`` that starts at its opening ``<``."""
    args, depth, start = [], 0, 1
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                args.append(text[start:index].strip())
                break
        elif char == "," and depth == 1:
            args.append(text[start:index].strip())
            start = index + 1
    return args


def short_kernel_name(name):
    """Drop the return type, call signature and template arguments unless they name the kernel."""
    text = (name[5:] if name.startswith("void ") else name).replace("(anonymous namespace)::", "")
    depth = 0
    for index, char in enumerate(text):
        if char in "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "(" and depth == 0:
            text = text[:index]
            break
    base, bracket, rest = text.partition("<")
    if not bracket:
        return base
    if base.rsplit("::", 1)[-1].startswith("Kernel"):
        args = _template_args(bracket + rest)
        if args:
            return args[0]
    return base


def _kind(category):
    return category if category in ("communication", "memory") else "compute"


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _length(intervals):
    return sum(end - start for start, end in intervals)


def _intersection(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append([start, end])
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def busy_breakdown(intervals):
    """Busy, overlapped and idle time of one GPU from ``(start, end, kind)`` activity intervals."""
    if not intervals:
        return dict(window_s=0.0, compute_s=0.0, communication_s=0.0, overlap_s=0.0, memory_s=0.0, idle_s=0.0,
                    exposed_communication_s=0.0)
    by_kind = {kind: _union([(start, end) for start, end, k in intervals if k == kind])
               for kind in ("compute", "communication", "memory")}
    window = max(end for _, end, _ in intervals) - min(start for start, _, _ in intervals)
    overlap = _length(_intersection(by_kind["compute"], by_kind["communication"]))
    communication = _length(by_kind["communication"])
    busy = _length(_union([(start, end) for start, end, _ in intervals]))
    return dict(window_s=window, compute_s=_length(by_kind["compute"]), communication_s=communication,
                overlap_s=overlap, memory_s=_length(by_kind["memory"]), idle_s=window - busy,
                exposed_communication_s=communication - overlap)


# Trace aggregation ------------------------------------------------------------------------------------------------

def _node(name):
    return {"name": name, "seconds": 0.0, "count": 0, "children": {}}


def _finish_tree(nodes, scale, depth=0):
    limit = (None, None, TREE_OPERATORS, TREE_KERNELS)[min(depth, 3)]
    ordered = sorted(nodes.values(), key=lambda node: node["seconds"], reverse=True)
    kept = ordered if limit is None else ordered[:limit]
    result = []
    for node in kept:
        result.append(dict(name=node["name"], seconds=node["seconds"] * scale, count=node["count"] * scale,
                           children=_finish_tree(node["children"], scale, depth + 1)))
    rest = ordered[len(kept):]
    if rest:
        result.append(dict(name=f"{len(rest)} more", seconds=sum(node["seconds"] for node in rest) * scale,
                           count=sum(node["count"] for node in rest) * scale, children=[]))
    return result


class _Aggregate:
    """Accumulates attributed GPU activities into the report's tree, kernel table and per-GPU breakdown."""

    def __init__(self):
        self.tree, self.kernels = {}, {}
        self.per_gpu = defaultdict(list)
        self.count = self.small = 0
        self.total = self.small_total = 0.0

    def add(self, gpu, start, end, name, stage, operator=None):
        seconds = end - start
        category = classify_kernel(name)
        short = short_kernel_name(name)
        self.per_gpu[gpu].append((start, end, _kind(category)))
        path = [stage, category] + ([operator] if operator is not None else []) + [short]
        level = self.tree
        for part in path:
            node = level.setdefault(part, _node(part))
            node["seconds"] += seconds
            node["count"] += 1
            level = node["children"]
        kernel = self.kernels.setdefault(short, dict(name=short, category=category, count=0, seconds=0.0))
        kernel["count"] += 1
        kernel["seconds"] += seconds
        if category != "memory":
            self.count += 1
            self.total += seconds
            if seconds < SMALL_KERNEL_S:
                self.small += 1
                self.small_total += seconds

    def result(self, source, steps, ranks):
        scale = 1 / (max(steps, 1) * max(ranks, 1))
        per_gpu = [dict(gpu=gpu, **busy_breakdown(rows), kernels=len(rows)) for gpu, rows in sorted(self.per_gpu.items())]
        kernels = sorted(self.kernels.values(), key=lambda row: row["seconds"], reverse=True)[:20]
        return dict(
            source=source, steps=steps, ranks=ranks, per_gpu=per_gpu, tree=_finish_tree(self.tree, scale),
            top_kernels=[dict(row, count=row["count"] * scale, seconds=row["seconds"] * scale,
                              mean_us=1e6 * row["seconds"] / row["count"]) for row in kernels],
            small_kernels=dict(count_share=self.small / self.count if self.count else 0.0,
                               time_share=self.small_total / self.total if self.total else 0.0),
        )


class _Stages:
    """Stage lookup by launch time within one process."""

    def __init__(self, ranges):
        self.ranges = sorted(ranges)
        self.starts = [start for start, _, _ in self.ranges]

    def at(self, time):
        index = bisect_right(self.starts, time) - 1
        if index >= 0 and time <= self.ranges[index][1]:
            return self.ranges[index][2]
        return "unattributed"


_SKIP_OPERATORS = ("##", "ProfilerStep", "Optimizer.", "DistributedDataParallel", STAGE_PREFIX)
# DDP launches its gradient all-reduce from autograd hooks inside other operators; name the collective instead.
_COLLECTIVES = ("nccl:", "c10d::")
_BACKWARD_PREFIX = "autograd::engine::evaluate_function: "


def _operator_name(event):
    name = event["name"].removeprefix(_BACKWARD_PREFIX)
    dims = [dim for dim in event.get("args", {}).get("Input Dims", []) if dim]
    if dims:
        name += " " + " · ".join("[" + ", ".join(map(str, dim)) + "]" for dim in dims)
    return name


def _top_level(rows):
    """Outermost operators of one thread from ``(start, end, event)`` rows sorted by start."""
    top = []
    for row in rows:
        if not top or row[0] >= top[-1][1]:
            top.append(row)
    return [row[0] for row in top], top


def _outermost(ops, time):
    """The outermost operator of one thread running at ``time``."""
    index = bisect_right(ops[0], time) - 1
    if index >= 0 and time <= ops[1][index][1]:
        return ops[1][index][2]
    return None


def analyze_torch_traces(paths, steps):
    """Stage → category → operator → kernel breakdown of PyTorch profiler traces, one file per rank."""
    aggregate = _Aggregate()
    for path in paths:
        with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as handle:
            events = json.load(handle)["traceEvents"]
        stage_ranges, threads, launches, device = defaultdict(list), defaultdict(list), {}, []
        for event in events:
            if event.get("ph") != "X":
                continue
            category, name = event.get("cat", ""), event.get("name", "")
            start, end = event["ts"], event["ts"] + event.get("dur", 0)
            if category == "user_annotation" and name.startswith(STAGE_PREFIX):
                stage_ranges[event["pid"]].append((start, end, name[len(STAGE_PREFIX):]))
            elif category == "cpu_op" and not name.startswith(_SKIP_OPERATORS):
                threads[(event["pid"], event["tid"])].append((start, end, event))
            elif category in ("cuda_runtime", "cuda_driver") and "correlation" in event.get("args", {}):
                launches[event["args"]["correlation"]] = event
            elif category in ("kernel", "gpu_memcpy", "gpu_memset"):
                device.append(event)
        stages = {pid: _Stages(ranges) for pid, ranges in stage_ranges.items()}
        ops = {key: _top_level(sorted(rows, key=lambda row: row[0])) for key, rows in threads.items()}
        collectives = {key: _top_level(sorted((row for row in rows if row[2]["name"].startswith(_COLLECTIVES)),
                                              key=lambda row: row[0])) for key, rows in threads.items()}
        for event in device:
            args = event.get("args", {})
            launch = launches.get(args.get("correlation"))
            stage, operator = "unattributed", "(no operator)"
            if launch is not None:
                if launch["pid"] in stages:
                    stage = stages[launch["pid"]].at(launch["ts"])
                key = (launch["pid"], launch["tid"])
                found = None
                if classify_kernel(event["name"]) == "communication" and collectives.get(key, ([], []))[0]:
                    found = _outermost(collectives[key], launch["ts"])
                if found is None and ops.get(key, ([], []))[0]:
                    found = _outermost(ops[key], launch["ts"])
                if found is not None:
                    operator = _operator_name(found)
            start = event["ts"] * 1e-6
            aggregate.add(args.get("device", event.get("pid")), start, start + event.get("dur", 0) * 1e-6,
                          event["name"], stage, operator)
    return aggregate.result("torch", steps, len(paths))


_MEMCPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def analyze_nsys_sqlite(path, steps):
    """Stage → category → kernel breakdown of an Nsight Systems SQLite export covering every rank."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ranges = defaultdict(list)
        if "NVTX_EVENTS" in tables:
            for start, end, tid, text in db.execute(
                    "SELECT e.start, e.end, e.globalTid, coalesce(e.text, s.value) FROM NVTX_EVENTS e "
                    "LEFT JOIN StringIds s ON s.id = e.textId WHERE e.eventType = 59"):
                if text and text.startswith(STAGE_PREFIX) and end is not None:
                    ranges[tid >> 24].append((start, end, text[len(STAGE_PREFIX):]))
        stages = {process: _Stages(rows) for process, rows in ranges.items()}
        aggregate, ranks = _Aggregate(), set()

        def attribute(process, launched):
            return stages[process].at(launched) if launched is not None and process in stages else "unattributed"

        for start, end, gpu, name, launched, process in db.execute(
                "SELECT k.start, k.end, k.deviceId, s.value, r.start, k.globalPid >> 24 "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.demangledName "
                "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
                "AND r.globalTid >> 24 = k.globalPid >> 24"):
            ranks.add(process)
            aggregate.add(gpu, start * 1e-9, end * 1e-9, name or "unknown kernel", attribute(process, launched))
        for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "Memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "Memset")):
            if table not in tables:
                continue
            kind = "m.copyKind" if label == "Memcpy" else "0"
            for start, end, gpu, copy, launched, process in db.execute(
                    f"SELECT m.start, m.end, m.deviceId, {kind}, r.start, m.globalPid >> 24 FROM {table} m "
                    "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId "
                    "AND r.globalTid >> 24 = m.globalPid >> 24"):
                name = f"{label} {_MEMCPY_KINDS.get(copy, '')}".strip()
                aggregate.add(gpu, start * 1e-9, end * 1e-9, name, attribute(process, launched))
    finally:
        db.close()
    return aggregate.result("nsys", steps, len(ranks))


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


def diagnose(stages, trace, projection, accounting, telemetry, peak_memory_bytes, memory_total_bytes):
    """Ordered findings a reader should check first; each names the evidence it comes from."""
    findings = []
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
        if gpu["window_s"] and gpu["exposed_communication_s"] / gpu["window_s"] > 0.2:
            findings.append(_finding(
                "exposed-communication", "warning", "Communication is not hidden behind compute",
                f"{gpu['exposed_communication_s'] / gpu['window_s']:.0%} of the traced window runs communication "
                f"kernels with no compute kernel alongside (overlap {gpu['overlap_s'] / gpu['window_s']:.0%})."))
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
