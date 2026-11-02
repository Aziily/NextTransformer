"""Analyse the traced updates of a profile: kernels, matrix efficiency, communication, scheduling and a timeline.

Reads PyTorch profiler (Kineto) traces, one file per rank, or the SQLite export of an Nsight Systems report. Every
GPU activity is attributed to the trainer stage that launched it (``nt.stage:`` annotations from
``launch.profile_hooks``); a PyTorch trace also names the operator, its input shapes and FLOPs, and the model
component (``nt.module:`` annotations, mapped to backward through autograd sequence numbers). Nothing imports torch.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import gzip
import json
import re
import sqlite3


STAGE_PREFIX = "nt.stage:"
MODULE_PREFIX = "nt.module:"
OUTSIDE = "(outside modules)"
SMALL_KERNEL_S = 10e-6
TREE_OPERATORS = 15
TREE_KERNELS = 8
TIMELINE_BUCKETS = 240
LAUNCH_APIS = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx",
               "cudaLaunchCooperativeKernel", "cudaGraphLaunch", "cuGraphLaunch"}
SYNC_APIS = {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize", "cuStreamSynchronize",
             "cuEventSynchronize", "cuCtxSynchronize", "cudaMemcpy", "cudaStreamWaitEvent_blocking"}


# Names and categories ---------------------------------------------------------------------------------------------

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
        if char == "<":
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


def component_of(path):
    """(component, layer) of an annotated module path: ``transformer.h.3.attn`` -> ``("attn", 3)``."""
    parts = path.split(".")
    if parts[-1].isdigit():
        return parts[-2] if len(parts) > 1 else parts[-1], int(parts[-1])
    layer = next((int(part) for part in reversed(parts[:-1]) if part.isdigit()), None)
    return parts[-1], layer


# FLOPs of matrix operators, counted like the cost audit: 2 per multiply-add, causal attention half of full, and the
# attention backward twice the forward.

_MATMUL = {"aten::mm": (0, 1), "aten::addmm": (-2, -1), "aten::bmm": (0, 1), "aten::baddbmm": (-2, -1)}
_ATTENTION = {"aten::_flash_attention_forward": (0, 1, 8, 1), "aten::_flash_attention_backward": (1, 2, 11, 2)}
MATRIX_OPERATORS = set(_MATMUL) | set(_ATTENTION)


def _dims(shape):
    return "×".join(map(str, shape))


def matmul_flops(name, dims):
    """(FLOPs, shape label) of a matrix multiply operator from its input dimensions, or (None, None)."""
    if name not in _MATMUL or len(dims) < 2:
        return None, None
    first, second = (dims[index] for index in _MATMUL[name])
    if len(first) < 2 or len(second) < 2 or first[-1] != second[-2]:
        return None, None
    batch = 1
    for size in first[:-2]:
        batch *= size
    return 2 * batch * first[-2] * first[-1] * second[-1], f"{_dims(first)} · {_dims(second)}"


def attention_flops(name, dims, concrete):
    """(FLOPs, shape label) of a FlashAttention operator; query and key are ``[batch, length, heads, dim]``."""
    if name not in _ATTENTION:
        return None, None
    query, key, causal_index, factor = _ATTENTION[name]
    if len(dims) <= max(query, key) or len(dims[query]) != 4 or len(dims[key]) != 4:
        return None, None
    batch, length, heads, dim = dims[query]
    keys = dims[key][1]
    causal = len(concrete) > causal_index and concrete[causal_index] == "True"
    flops = 4 * batch * heads * length * keys * dim * factor // (2 if causal else 1)
    return flops, f"B{batch} T{length} S{keys} H{heads} D{dim}" + (" causal" if causal else "")


# Collectives ------------------------------------------------------------------------------------------------------

_DTYPE_BYTES = {"float": 4, "float32": 4, "bfloat16": 2, "half": 2, "float16": 2, "double": 8, "int": 4,
                "long": 8, "byte": 1, "char": 1, "bool": 1}


def collective_name(name):
    """Normalized collective: ``allreduce``, ``_allgather_base`` and ``AllGather`` become ``all_reduce``/``all_gather``."""
    lowered = name.lower().replace("_", "")
    for needle, label in (("allreduce", "all_reduce"), ("allgather", "all_gather"), ("reducescatter", "reduce_scatter"),
                          ("alltoall", "all_to_all"), ("broadcast", "broadcast"), ("sendrecv", "send_recv"),
                          ("reduce", "reduce"), ("send", "send_recv"), ("recv", "send_recv")):
        if needle in lowered:
            return label
    return name.lower()


def _kernel_collective(name):
    match = re.match(r"nccl(?:Dev)?Kernel_([A-Za-z]+)", name)
    return collective_name(match.group(1)) if match else "nccl"


def busbw_factor(collective, ranks):
    """Bus-bandwidth factor of nccl-tests: algorithm bandwidth times this is what each link carries."""
    if not ranks:
        return None
    name = collective_name(collective)
    if name == "all_reduce":
        return 2 * (ranks - 1) / ranks
    if name in ("all_gather", "reduce_scatter", "all_to_all"):
        return (ranks - 1) / ranks
    return 1.0


def _message_bytes(collective, args):
    size = _DTYPE_BYTES.get(str(args.get("dtype", "")).lower())
    incoming, outgoing = args.get("In msg nelems"), args.get("Out msg nelems")
    if size is None or incoming is None:
        return None
    if collective == "all_gather":
        return size * (outgoing if outgoing is not None else incoming * args.get("Group size", 1))
    return size * incoming


# Intervals --------------------------------------------------------------------------------------------------------

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


def _covered(merged, starts, start, end):
    """Length of ``[start, end]`` covered by sorted disjoint ``merged`` intervals whose starts are ``starts``."""
    index, covered = max(bisect_right(starts, start) - 1, 0), 0.0
    while index < len(merged) and merged[index][0] < end:
        covered += max(0.0, min(end, merged[index][1]) - max(start, merged[index][0]))
        index += 1
    return covered


def busy_breakdown(intervals):
    """Busy, overlapped and idle time of one GPU from ``(start, end, kind)`` activity intervals."""
    if not intervals:
        return dict(window_s=0.0, compute_s=0.0, communication_s=0.0, overlap_s=0.0, memory_s=0.0,
                    exposed_memory_s=0.0, idle_s=0.0, exposed_communication_s=0.0)
    by_kind = {kind: _union([(start, end) for start, end, k in intervals if k == kind])
               for kind in ("compute", "communication", "memory")}
    window = max(end for _, end, _ in intervals) - min(start for start, _, _ in intervals)
    overlap = _length(_intersection(by_kind["compute"], by_kind["communication"]))
    communication = _length(by_kind["communication"])
    others = _union([*map(tuple, by_kind["compute"]), *map(tuple, by_kind["communication"])])
    memory = _length(by_kind["memory"])
    busy = _length(_union([(start, end) for start, end, _ in intervals]))
    return dict(window_s=window, compute_s=_length(by_kind["compute"]), communication_s=communication,
                overlap_s=overlap, memory_s=memory, exposed_memory_s=memory - _length(_intersection(by_kind["memory"], others)),
                idle_s=window - busy, exposed_communication_s=communication - overlap)


class _Ranges:
    """Labelled host time ranges of one thread or process; ``at`` returns the innermost range around a time."""

    def __init__(self, ranges):
        self.ranges = sorted(ranges, key=lambda row: row[0])
        self.starts = [row[0] for row in self.ranges]

    def at(self, time, default=None):
        index = bisect_right(self.starts, time) - 1
        for candidate in range(index, max(index - 64, -1), -1):
            start, end, label = self.ranges[candidate][:3]
            if start <= time <= end:
                return label
        return default


def _top_level(rows):
    """Outermost ``(start, end, event)`` rows of one thread, from rows sorted by start."""
    top = []
    for row in rows:
        if not top or row[0] >= top[-1][1]:
            top.append(row)
    return _Ranges(top)


# Aggregation ------------------------------------------------------------------------------------------------------

def _node(name):
    return {"name": name, "seconds": 0.0, "count": 0, "children": {}}


def _finish_tree(nodes, scale, depth=0):
    limit = (None, None, TREE_OPERATORS, TREE_KERNELS)[min(depth, 3)]
    ordered = sorted(nodes.values(), key=lambda node: node["seconds"], reverse=True)
    kept = ordered if limit is None else ordered[:limit]
    result = [dict(name=node["name"], seconds=node["seconds"] * scale, count=node["count"] * scale,
                   children=_finish_tree(node["children"], scale, depth + 1)) for node in kept]
    rest = ordered[len(kept):]
    if rest:
        result.append(dict(name=f"{len(rest)} more", seconds=sum(node["seconds"] for node in rest) * scale,
                           count=sum(node["count"] for node in rest) * scale, children=[]))
    return result


class _Activity:
    __slots__ = ("gpu", "start", "end", "name", "short", "category", "kind", "stage", "launched", "label")

    def __init__(self, gpu, start, end, name, stage, launched):
        self.gpu, self.start, self.end, self.name, self.stage, self.launched = gpu, start, end, name, stage, launched
        self.short, self.category = short_kernel_name(name), classify_kernel(name)
        self.kind = _kind(self.category)
        self.label = self.category


class _Aggregate:
    """Accumulates attributed GPU activities and host events into the trace summary."""

    def __init__(self):
        self.tree, self.kernels = {}, {}
        self.activities = defaultdict(list)
        self.count = self.small = 0
        self.total = self.small_total = 0.0
        self.instances, self.components = {}, defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        self.module_names, self.module_classes = set(), {}
        self.communication = []
        self.launch = dict(api_calls=0, kernels=0, memory_ops=0, host_s=0.0, micro_batches=0)
        self.syncs = defaultdict(lambda: [0, 0.0])
        self.timeline = None
        self.categories = defaultdict(float)
        self.other_operators = defaultdict(lambda: [0, 0.0])

    def add(self, gpu, start, end, name, stage, operator=None, launched=None, instance=None, component=None,
            collective=None):
        activity = _Activity(gpu, start, end, name, stage, launched)
        seconds = end - start
        self.activities[gpu].append(activity)
        path = [stage, activity.category] + ([operator] if operator is not None else []) + [activity.short]
        level = self.tree
        for part in path:
            node = level.setdefault(part, _node(part))
            node["seconds"] += seconds
            node["count"] += 1
            level = node["children"]
        kernel = self.kernels.setdefault(activity.short, dict(name=activity.short, category=activity.category,
                                                              count=0, seconds=0.0))
        kernel["count"] += 1
        kernel["seconds"] += seconds
        self.categories[activity.category] += seconds
        if activity.kind == "memory":
            self.launch["memory_ops"] += 1
            return activity
        if activity.kind == "compute" and activity.category not in ("gemm", "attention") and operator is not None:
            row = self.other_operators[(stage, operator.split(" ", 1)[0], activity.category)]
            row[0] += 1
            row[1] += seconds
        self.launch["kernels"] += 1
        self.count += 1
        self.total += seconds
        if seconds < SMALL_KERNEL_S:
            self.small += 1
            self.small_total += seconds
        if instance is not None:
            row = self.instances.setdefault(instance["key"], dict(instance, seconds=0.0, gpu=gpu, kernels=defaultdict(float),
                                                                  component=component, flops_seconds=0.0))
            row["seconds"] += seconds
            row["kernels"][activity.short] += seconds
        if activity.kind == "compute" and component is not None and stage in ("forward", "backward"):
            self.components[component][stage][0] += seconds
        if collective is not None:
            activity.label = collective["collective"]
            self.communication.append((activity, collective))
        return activity

    def sync(self, api, source, stage, seconds):
        row = self.syncs[(api, source, stage)]
        row[0] += 1
        row[1] += seconds

    def result(self, source, steps, ranks):
        scale = 1 / (max(steps, 1) * max(ranks, 1))
        per_gpu, compute_unions = [], {}
        for gpu, rows in sorted(self.activities.items()):
            breakdown = busy_breakdown([(row.start, row.end, row.kind) for row in rows])
            merged = _union([(row.start, row.end) for row in rows if row.kind == "compute"])
            compute_unions[gpu] = (merged, [row[0] for row in merged])
            matrix = defaultdict(float)
            for row in rows:
                if row.category in ("gemm", "attention"):
                    matrix[row.category] += row.end - row.start
            flops = defaultdict(float)
            for instance in self.instances.values():
                if instance["gpu"] == gpu and instance["flops"]:
                    flops[instance["kind"]] += instance["flops"]
            traced_flops = source == "torch"
            per_gpu.append(dict(gpu=gpu, **breakdown, kernels=len(rows), gemm_s=matrix["gemm"],
                                gemm_flops=flops["gemm"] if traced_flops else None, attention_s=matrix["attention"],
                                attention_flops=flops["attention"] if traced_flops else None))
        kernels = sorted(self.kernels.values(), key=lambda row: row["seconds"], reverse=True)[:20]
        launch = {key: value * scale for key, value in self.launch.items()}
        launch["per_micro_batch"] = self.launch["kernels"] / self.launch["micro_batches"] \
            if self.launch["micro_batches"] else None
        return dict(
            source=source, steps=steps, ranks=ranks, per_gpu=per_gpu, tree=_finish_tree(self.tree, scale),
            top_kernels=[dict(row, count=row["count"] * scale, seconds=row["seconds"] * scale,
                              mean_us=1e6 * row["seconds"] / row["count"]) for row in kernels],
            small_kernels=dict(count_share=self.small / self.count if self.count else 0.0,
                               time_share=self.small_total / self.total if self.total else 0.0),
            gemm=self._matrix_rows(scale), components=self._component_rows(scale),
            communication=self._communication_rows(scale, compute_unions), launch=launch,
            syncs=sorted((dict(api=api, source=source_, stage=stage, calls=calls * scale, seconds=seconds * scale)
                          for (api, source_, stage), (calls, seconds) in self.syncs.items()),
                         key=lambda row: row["seconds"], reverse=True),
            idle=self._idle(scale), timeline=self.timeline or [],
            categories={name: seconds * scale for name, seconds in sorted(self.categories.items(), key=lambda kv: -kv[1])},
            other_operators=sorted((dict(stage=stage, op=op, category=category, calls=calls * scale, seconds=seconds * scale)
                                    for (stage, op, category), (calls, seconds) in self.other_operators.items()),
                                   key=lambda row: row["seconds"], reverse=True)[:25],
        )

    def _matrix_rows(self, scale):
        groups = {}
        for instance in self.instances.values():
            if not instance["flops"]:
                continue
            key = (instance["kind"], instance["stage"], instance["op"], instance["shape"], instance["dtype"])
            row = groups.setdefault(key, dict(kind=key[0], stage=key[1], op=key[2], shape=key[3], dtype=key[4], calls=0,
                                              seconds=0.0, flops=0, kernels=defaultdict(float)))
            row["calls"] += 1
            row["seconds"] += instance["seconds"]
            row["flops"] += instance["flops"]
            for name, seconds in instance["kernels"].items():
                row["kernels"][name] += seconds
        rows = []
        for row in groups.values():
            kernels = row.pop("kernels")
            rows.append(dict(row, kernel=max(kernels, key=kernels.get) if kernels else None,
                             tflops=row["flops"] / row["seconds"] / 1e12 if row["seconds"] else None,
                             calls=row["calls"] * scale, seconds=row["seconds"] * scale, flops=row["flops"] * scale))
        return sorted(rows, key=lambda row: row["seconds"], reverse=True)

    def _component_rows(self, scale):
        for instance in self.instances.values():
            if instance["component"] is not None and instance["flops"]:
                self.components[instance["component"]]["flops"][0] += instance["flops"]
        blocks = {name for name in self.module_names if any(other.startswith(name + ".") for other in self.module_names)}
        # A hybrid model puts different mixers at the same position (``attn``); name those by their class.
        classes = defaultdict(set)
        for path in self.components:
            if path != OUTSIDE and path not in blocks:
                classes[component_of(path)[0]].add(self.module_classes.get(path))
        rows = {}
        for path, values in self.components.items():
            name, layer = (OUTSIDE, None) if path == OUTSIDE else component_of(path)
            if path in blocks:
                name = f"{self.module_classes.get(path, name)} (outside its submodules)"
            elif len(classes.get(name, ())) > 1 and self.module_classes.get(path):
                name = f"{name} ({self.module_classes[path]})"
            row = rows.setdefault(name, dict(name=name, forward_s=0.0, backward_s=0.0, flops=0, layers={}))
            forward, backward, flops = values["forward"][0], values["backward"][0], values["flops"][0]
            row["forward_s"] += forward * scale
            row["backward_s"] += backward * scale
            row["flops"] += flops * scale
            if layer is not None:
                entry = row["layers"].setdefault(layer, dict(layer=layer, forward_s=0.0, backward_s=0.0))
                entry["forward_s"] += forward * scale
                entry["backward_s"] += backward * scale
        result = [dict(row, layers=[row["layers"][key] for key in sorted(row["layers"])]) for row in rows.values()]
        return sorted(result, key=lambda row: row["forward_s"] + row["backward_s"], reverse=True)

    def _communication_rows(self, scale, compute_unions):
        groups = {}
        for activity, collective in self.communication:
            merged, starts = compute_unions.get(activity.gpu, ([], []))
            seconds = activity.end - activity.start
            exposed = seconds - _covered(merged, starts, activity.start, activity.end)
            # NCCL launches one generic kernel per collective and data type, so its name does not identify the protocol.
            key = (collective["collective"], collective["dtype"], activity.stage, collective["kernel"])
            row = groups.setdefault(key, dict(collective=key[0], dtype=key[1], stage=key[2], kernel=key[3], calls=0,
                                              seconds=0.0, exposed_s=0.0, bytes=0, ranks=collective["ranks"]))
            row["calls"] += 1
            row["seconds"] += seconds
            row["exposed_s"] += exposed
            row["bytes"] = None if row["bytes"] is None or collective["bytes"] is None else row["bytes"] + collective["bytes"]
        rows = []
        for row in groups.values():
            algbw = row["bytes"] / row["seconds"] / 1e9 if row["bytes"] and row["seconds"] else None
            factor = busbw_factor(row["collective"], row["ranks"])
            rows.append(dict(row, algbw_gbps=algbw, busbw_gbps=algbw * factor if algbw is not None and factor else None,
                             calls=row["calls"] * scale, seconds=row["seconds"] * scale, exposed_s=row["exposed_s"] * scale,
                             bytes=row["bytes"] * scale if row["bytes"] is not None else None))
        return sorted(rows, key=lambda row: row["seconds"], reverse=True)

    def _idle(self, scale):
        totals, gaps = dict(total_s=0.0, host_s=0.0, launch_s=0.0, dependency_s=0.0), []
        count = 0
        for gpu, rows in self.activities.items():
            rows = sorted(rows, key=lambda row: row.start)
            origin, busy = rows[0].start, rows[0].end
            for row in rows[1:]:
                if row.start > busy:
                    gap = row.start - busy
                    if row.launched is None or row.launched <= busy:
                        parts = dict(host_s=0.0, launch_s=0.0, dependency_s=gap)
                    else:
                        host = min(row.launched - busy, gap)
                        parts = dict(host_s=host, launch_s=gap - host, dependency_s=0.0)
                    for key, value in parts.items():
                        totals[key] += value
                    totals["total_s"] += gap
                    count += 1
                    cause = max(parts, key=parts.get).removesuffix("_s")
                    gaps.append(dict(gpu=gpu, seconds=gap, start_s=busy - origin, cause=cause, stage=row.stage,
                                     next=row.short))
                busy = max(busy, row.end)
        largest = sorted(gaps, key=lambda row: row["seconds"], reverse=True)[:8]
        return dict({key: value * scale for key, value in totals.items()}, gaps=count * scale, largest=largest)

    def build_timeline(self, gpu, forward, backward):
        """Lanes of one GPU over the traced update, its middle micro-batch and the update tail."""
        rows = sorted(self.activities.get(gpu, []), key=lambda row: row.start)
        if not rows:
            return
        origin, finish = rows[0].start, max(row.end for row in rows)
        windows = [("update", origin, finish)]
        if forward and len(forward) == len(backward):
            middle = len(forward) // 2
            low, high = forward[middle][0], backward[middle][1]
            inside = [row for row in rows if row.launched is not None and low <= row.launched <= high]
            if inside:
                windows.append((f"micro-batch {middle + 1} of {len(forward)}", min(row.start for row in inside),
                                max(row.end for row in inside)))
        if backward:
            tail = [row for row in rows if row.launched is not None and row.launched >= backward[-1][0]]
            if tail:
                windows.append(("update tail", min(row.start for row in tail), finish))
        self.timeline = [_window(name, rows, origin, start, end) for name, start, end in windows]


def _window(name, rows, origin, start, end, buckets=TIMELINE_BUCKETS):
    width = (end - start) / buckets if end > start else 1.0
    lanes = {}
    for lane, select, label in (("stage", lambda row: True, lambda row: row.stage),
                                ("compute", lambda row: row.kind == "compute", lambda row: row.category),
                                ("communication", lambda row: row.kind == "communication", lambda row: row.label),
                                ("memory", lambda row: row.kind == "memory", lambda row: "memory")):
        cells = [defaultdict(float) for _ in range(buckets)]
        for row in rows:
            if row.end <= start or row.start >= end or not select(row):
                continue
            low, high = max(row.start, start), min(row.end, end)
            first, last = int((low - start) / width), min(int((high - start) / width), buckets - 1)
            for index in range(first, last + 1):
                left = start + index * width
                cells[index][label(row)] += max(0.0, min(high, left + width) - max(low, left))
        runs = []
        for index, cell in enumerate(cells):
            if not cell:
                continue
            top = max(cell, key=cell.get)
            intensity = min(1.0, sum(cell.values()) / width)
            if runs and runs[-1][2] == top and runs[-1][1] == index:
                previous = runs[-1]
                previous[3] = (previous[3] * (previous[1] - previous[0]) + intensity) / (previous[1] - previous[0] + 1)
                previous[1] = index + 1
            else:
                runs.append([index, index + 1, top, intensity])
        lanes[lane] = [[first, last, label_, round(max(intensity, 1e-3), 3)] for first, last, label_, intensity in runs]
    # Busy share of matrix (GEMM, attention) and other compute kernels in each bucket, capped at a full bucket.
    mix = [[0.0, 0.0] for _ in range(buckets)]
    for row in rows:
        if row.kind != "compute" or row.end <= start or row.start >= end:
            continue
        low, high = max(row.start, start), min(row.end, end)
        slot = 0 if row.category in ("gemm", "attention") else 1
        for index in range(int((low - start) / width), min(int((high - start) / width), buckets - 1) + 1):
            left = start + index * width
            mix[index][slot] += max(0.0, min(high, left + width) - max(low, left)) / width
    for cell in mix:
        total = cell[0] + cell[1]
        if total > 1:
            cell[0], cell[1] = cell[0] / total, cell[1] / total
    return dict(name=name, start_s=start - origin, seconds=end - start, buckets=buckets, lanes=lanes,
                mix=[[round(matrix, 3), round(other, 3)] for matrix, other in mix])


# PyTorch profiler traces ------------------------------------------------------------------------------------------

_SKIP_OPERATORS = ("##", "ProfilerStep", "Optimizer.", "DistributedDataParallel", STAGE_PREFIX, MODULE_PREFIX)
# DDP launches its gradient all-reduce from autograd hooks inside other operators; name the collective instead.
_COLLECTIVES = ("nccl:", "c10d::")
_BACKWARD_PREFIX = "autograd::engine::evaluate_function: "


def _operator_name(event):
    name = event["name"].removeprefix(_BACKWARD_PREFIX)
    dims = [dim for dim in event.get("args", {}).get("Input Dims", []) if dim]
    if dims:
        name += " " + " · ".join("[" + ", ".join(map(str, dim)) + "]" for dim in dims)
    return name


def _matrix_instance(event):
    args = event.get("args", {})
    dims, name = args.get("Input Dims", []), event["name"]
    if name in _MATMUL:
        flops, shape = matmul_flops(name, dims)
        kind = "gemm"
    else:
        flops, shape = attention_flops(name, dims, args.get("Concrete Inputs", []))
        kind = "attention"
    types = args.get("Input type") or [None]
    return dict(key=id(event), kind=kind, op=name, shape=shape, dtype=types[0], flops=flops)


def _thread_ranges(rows):
    return {key: _top_level(sorted(values, key=lambda row: row[0])) for key, values in rows.items()}


def _torch_trace(path, aggregate, timeline):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as handle:
        events = json.load(handle)["traceEvents"]
    stage_rows, module_rows = defaultdict(list), defaultdict(list)
    operators, collectives, matrices, backward_ops = (defaultdict(list) for _ in range(4))
    sequenced, launches, device, syncs = [], {}, [], []
    for event in events:
        if event.get("ph") != "X":
            continue
        category, name, args = event.get("cat", ""), event.get("name", ""), event.get("args", {})
        start = event["ts"]
        end = start + event.get("dur", 0)
        thread = (event.get("pid"), event.get("tid"))
        if category == "user_annotation" and name.startswith(STAGE_PREFIX):
            stage_rows[event["pid"]].append((start, end, name[len(STAGE_PREFIX):]))
        elif category == "user_annotation" and name.startswith(MODULE_PREFIX):
            path, _, kind = name[len(MODULE_PREFIX):].partition("|")
            module_rows[thread].append((start, end, path))
            if kind:
                aggregate.module_classes[path] = kind
        elif category == "cpu_op":
            if name.startswith(_SKIP_OPERATORS):
                continue
            row = (start, end, event)
            operators[thread].append(row)
            if name.startswith(_COLLECTIVES):
                collectives[thread].append(row)
            if name in MATRIX_OPERATORS:
                matrices[thread].append(row)
            sequence = args.get("Sequence number")
            if name.startswith(_BACKWARD_PREFIX):
                if sequence is not None:
                    backward_ops[thread].append((start, end, sequence))
            elif sequence is not None:
                sequenced.append((thread, start, sequence))
        elif category in ("cuda_runtime", "cuda_driver"):
            if "correlation" in args:
                launches[args["correlation"]] = event
            if name in LAUNCH_APIS:
                aggregate.launch["api_calls"] += 1
                aggregate.launch["host_s"] += event.get("dur", 0) * 1e-6
            elif name in SYNC_APIS:
                syncs.append(event)
        elif category in ("kernel", "gpu_memcpy", "gpu_memset"):
            device.append(event)
    stages = {pid: _Ranges(rows) for pid, rows in stage_rows.items()}
    modules = {thread: _Ranges(rows) for thread, rows in module_rows.items()}
    for rows in module_rows.values():
        aggregate.module_names.update(row[2] for row in rows)
    operators, collectives, matrices = _thread_ranges(operators), _thread_ranges(collectives), _thread_ranges(matrices)
    backward_ops = {thread: _Ranges(rows) for thread, rows in backward_ops.items()}
    origin = {}
    for thread, start, sequence in sequenced:
        if sequence not in origin and thread in modules:
            origin[sequence] = modules[thread].at(start)
    instances = {}

    def stage_at(pid, time):
        return stages[pid].at(time, "unattributed") if pid in stages else "unattributed"

    for pid, rows in stage_rows.items():
        aggregate.launch["micro_batches"] += sum(1 for row in rows if row[2] == "forward")
    for event in syncs:
        thread = (event.get("pid"), event.get("tid"))
        found = operators[thread].at(event["ts"]) if thread in operators else None
        aggregate.sync(event["name"], found["name"] if found else "(outside operators)", stage_at(event["pid"], event["ts"]),
                       event.get("dur", 0) * 1e-6)
    for event in device:
        args = event.get("args", {})
        launch = launches.get(args.get("correlation"))
        stage, operator, instance, component, launched = "unattributed", "(no operator)", None, None, None
        category = classify_kernel(event["name"])
        if launch is not None:
            launched = launch["ts"] * 1e-6
            thread = (launch["pid"], launch["tid"])
            stage = stage_at(launch["pid"], launch["ts"])
            found = None
            if category == "communication" and thread in collectives:
                found = collectives[thread].at(launch["ts"])
            if found is None and thread in operators:
                found = operators[thread].at(launch["ts"])
            if found is not None:
                operator = _operator_name(found)
            # Only the matrix kernels of a matrix operator count toward its time; casts it launches do not.
            matrix = matrices[thread].at(launch["ts"]) if thread in matrices and category in ("gemm", "attention") else None
            if matrix is not None:
                instance = instances.get(id(matrix))
                if instance is None:
                    instance = instances[id(matrix)] = dict(_matrix_instance(matrix), stage=stage)
            if stage == "forward" and modules:
                component = modules[thread].at(launch["ts"], OUTSIDE) if thread in modules else OUTSIDE
            elif stage == "backward" and modules:
                sequence = backward_ops[thread].at(launch["ts"]) if thread in backward_ops else None
                component = origin.get(sequence) or OUTSIDE
        collective = None
        if category == "communication":
            name = collective_name(args["Collective name"]) if args.get("Collective name") else _kernel_collective(event["name"])
            collective = dict(collective=name, dtype=args.get("dtype", "?"), kernel=short_kernel_name(event["name"]),
                              bytes=_message_bytes(name, args), ranks=args.get("Group size"))
        start = event["ts"] * 1e-6
        aggregate.add(args.get("device", event.get("pid")), start, start + event.get("dur", 0) * 1e-6, event["name"], stage,
                      operator, launched, instance, component, collective)
    if timeline and device:
        main = max(stage_rows, key=lambda pid: len(stage_rows[pid])) if stage_rows else None
        rows = sorted(stage_rows.get(main, []))
        gpu = device[0].get("args", {}).get("device", device[0].get("pid"))
        aggregate.build_timeline(gpu, [(start * 1e-6, end * 1e-6) for start, end, label in rows if label == "forward"],
                                 [(start * 1e-6, end * 1e-6) for start, end, label in rows if label == "backward"])


def analyze_torch_traces(paths, steps):
    """Kernels, matrix efficiency, components, communication and scheduling of PyTorch profiler traces."""
    aggregate = _Aggregate()
    for index, path in enumerate(paths):
        _torch_trace(path, aggregate, timeline=index == 0)
    return aggregate.result("torch", steps, len(paths))


# Nsight Systems ---------------------------------------------------------------------------------------------------

_MEMCPY_KINDS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def _api(name):
    return re.sub(r"_v\d+$", "", name or "")


def analyze_nsys_sqlite(path, steps):
    """Kernels, communication and scheduling of an Nsight Systems SQLite export covering every rank."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ranges = defaultdict(list)
        if "NVTX_EVENTS" in tables:
            for start, end, tid, text in db.execute(
                    "SELECT e.start, e.end, e.globalTid, coalesce(e.text, s.value) FROM NVTX_EVENTS e "
                    "LEFT JOIN StringIds s ON s.id = e.textId WHERE e.eventType = 59"):
                if text and text.startswith(STAGE_PREFIX) and end is not None:
                    ranges[tid >> 24].append((start * 1e-9, end * 1e-9, text[len(STAGE_PREFIX):]))
        stages = {process: _Ranges(rows) for process, rows in ranges.items()}
        aggregate, ranks = _Aggregate(), set()

        def stage_at(process, time):
            return stages[process].at(time, "unattributed") if time is not None and process in stages else "unattributed"

        first = {}
        for start, end, gpu, name, launched, process in db.execute(
                "SELECT k.start, k.end, k.deviceId, s.value, r.start, k.globalPid >> 24 "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.demangledName "
                "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
                "AND r.globalTid >> 24 = k.globalPid >> 24"):
            ranks.add(process)
            first.setdefault(gpu, process)
            name = name or "unknown kernel"
            launched = launched * 1e-9 if launched is not None else None
            collective = None
            if classify_kernel(name) == "communication":
                collective = dict(collective=_kernel_collective(name), dtype="?", bytes=None, ranks=None,
                                  kernel=short_kernel_name(name))
            aggregate.add(gpu, start * 1e-9, end * 1e-9, name, stage_at(process, launched), launched=launched,
                          collective=collective)
        for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "Memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "Memset")):
            if table not in tables:
                continue
            kind = "m.copyKind" if label == "Memcpy" else "0"
            for start, end, gpu, copy, launched, process in db.execute(
                    f"SELECT m.start, m.end, m.deviceId, {kind}, r.start, m.globalPid >> 24 FROM {table} m "
                    "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId "
                    "AND r.globalTid >> 24 = m.globalPid >> 24"):
                launched = launched * 1e-9 if launched is not None else None
                aggregate.add(gpu, start * 1e-9, end * 1e-9, f"{label} {_MEMCPY_KINDS.get(copy, '')}".strip(),
                              stage_at(process, launched), launched=launched)
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            for start, end, tid, name in db.execute(
                    "SELECT r.start, r.end, r.globalTid, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "JOIN StringIds s ON s.id = r.nameId WHERE s.value LIKE '%Launch%' OR s.value LIKE '%Synchronize%'"):
                api = _api(name)
                if api in LAUNCH_APIS:
                    aggregate.launch["api_calls"] += 1
                    aggregate.launch["host_s"] += (end - start) * 1e-9
                elif api in SYNC_APIS:
                    aggregate.sync(api, "(not traced)", stage_at(tid >> 24, start * 1e-9), (end - start) * 1e-9)
        for process, rows in ranges.items():
            aggregate.launch["micro_batches"] += sum(1 for row in rows if row[2] == "forward")
        if first:
            gpu = min(first)
            rows = sorted(ranges.get(first[gpu], []))
            aggregate.build_timeline(gpu, [row[:2] for row in rows if row[2] == "forward"],
                                     [row[:2] for row in rows if row[2] == "backward"])
    finally:
        db.close()
    return aggregate.result("nsys", steps, len(ranks))
