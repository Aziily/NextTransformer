"""Local, read-only monitor for run directories of the reference trainer.

    python launch/run_monitor.py RUNS [RUNS ...]              # serve http://127.0.0.1:8765
    python launch/run_monitor.py RUNS --once OUTPUT_DIR       # write static pages once

RUNS is a run directory or a folder that contains run directories. The monitor reads only what the trainer already
writes (``metrics.jsonl``, ``provenance.json``, ``run_status.json``, ``budget.json``, ``latest.json`` and the checkpoint
folders) and reads them again on every request. It never writes into a run directory and never opens ``run.lock``, so it
cannot delay or block a trainer. It imports neither torch nor the trainer: any Python 3.10 environment can watch a run.

A run page shows progress against the protocol's evaluation schedule, the losses, throughput, the compute ledger,
evaluations and health checks; the comparison page overlays up to eight runs.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "runs"
# Protocol values used when a run directory has no provenance yet; a run's own provenance always wins.
DEFAULT_PROTOCOL = dict(total_tokens=49_152_000_000, global_batch_tokens=1_048_576, val_every=250, val_tokens=41_943_040,
                        core_steps=[8000, 16000, 24000, 32000, 40000, 46875], preflight_tolerance=0.01,
                        cumulative_tolerance=0.02)
CARD_ALIASES = {"dense": "sssl"}  # SSSL keeps the historical method id `dense`.
SKIP_FOLDERS = ("checkpoint_", "replayed_events", ".", "__pycache__", "node_modules", "stages", "wandb")
RUN_FILES = ("metrics.jsonl", "provenance.json", "run_status.json", "budget.json", "latest.json")
PHASES = {"training_complete": ("complete", "Training complete"), "smoke_complete": ("complete", "Smoke run complete"),
          "segment_complete": ("segment", "Segment ended, resume to continue"), "stopped": ("stopped", "Stopped")}
MAX_COMPARE = 8
QUIET_FLOOR_S = 15 * 60


# ---------------------------------------------------------------------------------------------------------- reading

def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_events(path: Path):
    """Events of ``metrics.jsonl`` and the number of unreadable lines; a line still being written is not counted."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return [], 0
    events, malformed = [], 0
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            malformed += number != len(lines) - 1
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, malformed


def is_run(folder: Path) -> bool:
    return (folder / "metrics.jsonl").is_file() or (folder / "provenance.json").is_file()


def find_runs(roots, depth=3):
    """Run directories under the roots as {id: path}; ids are paths relative to their root, prefixed by the root's name
    when several roots are watched. A run directory's own subfolders are not searched."""
    found = {}
    roots = [Path(root).resolve() for root in roots]
    for root in roots:
        prefix = root.name if len(roots) > 1 else ""
        stack = [(root, 0)]
        while stack:
            folder, level = stack.pop()
            if is_run(folder):
                relative = folder.relative_to(root).as_posix()
                identifier = "/".join(part for part in (prefix, "" if relative == "." else relative) if part) or folder.name
                while identifier in found:
                    identifier += "~"
                found[identifier] = folder
                continue
            if level >= depth:
                continue
            try:
                children = sorted(child for child in folder.iterdir()
                                  if child.is_dir() and not child.name.startswith(SKIP_FOLDERS))
            except OSError:
                continue
            stack.extend((child, level + 1) for child in reversed(children))
    return found


def _signature(folder: Path):
    parts = []
    for name in RUN_FILES:
        try:
            stat = (folder / name).stat()
            parts.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            parts.append((name, None, None))
    try:
        parts.append(("folder", folder.stat().st_mtime_ns, None))
    except OSError:
        pass
    return tuple(parts)


_CACHE: dict = {}


def load_run(folder: Path) -> dict:
    """Everything the trainer wrote into one run directory, cached until one of its files changes."""
    signature = _signature(folder)
    cached = _CACHE.get(folder)
    if cached and cached[0] == signature:
        return cached[1]
    events, malformed = read_events(folder / "metrics.jsonl")
    checkpoints = []
    try:
        for child in folder.iterdir():
            if child.name.startswith("checkpoint_") and (child / "complete.json").is_file():
                try:
                    checkpoints.append(int(child.name.removeprefix("checkpoint_")))
                except ValueError:
                    pass
    except OSError:
        pass
    mtimes = {name: (folder / name).stat().st_mtime for name in ("metrics.jsonl", "run_status.json", "provenance.json")
              if (folder / name).is_file()}
    run = dict(path=str(folder), events=events, malformed=malformed, provenance=read_json(folder / "provenance.json") or {},
               status=read_json(folder / "run_status.json"), budget=read_json(folder / "budget.json"),
               latest=read_json(folder / "latest.json"), checkpoints=sorted(checkpoints), mtimes=mtimes)
    _CACHE[folder] = (signature, run)
    return run


# -------------------------------------------------------------------------------------------------------- analysis

def _command_value(command, flag):
    for index, part in enumerate(command or []):
        if part == flag and index + 1 < len(command):
            return command[index + 1]
        if isinstance(part, str) and part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return None


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _median(values):
    values = [value for value in values if _finite(value)]
    return statistics.median(values) if values else None


def schedule(provenance: dict) -> dict:
    """Total updates and the validation and CORE steps this run is expected to evaluate at."""
    protocol = dict(DEFAULT_PROTOCOL, **(provenance.get("protocol") or {}))
    formal_steps = protocol["total_tokens"] // protocol["global_batch_tokens"]
    command = provenance.get("command") or []
    smoke_steps = _command_value(command, "--smoke-steps")
    smoke = bool(provenance.get("smoke")) or bool(smoke_steps and int(smoke_steps) > 0)
    if smoke and smoke_steps:
        total = min(formal_steps, int(smoke_steps))
        validation = [0, total]
        core = [total] if (_command_value(command, "--smoke-core-per-task") or "0") != "0" else []
    else:
        total = formal_steps
        validation = sorted({0, *range(protocol["val_every"], total, protocol["val_every"]), total})
        core = [step for step in protocol["core_steps"] if step <= total]
    return dict(total=total, smoke=smoke, validation=validation, core=core, protocol=protocol,
                log_every=int(_command_value(command, "--log-every") or 10))


def _status(run: dict, quiet_after: float, now: float):
    """(kind, label) of a run: complete, segment, stopped, running, quiet or empty."""
    mtimes, status = run["mtimes"], run["status"]
    newest = max(mtimes.values(), default=None)
    if status and mtimes.get("run_status.json", 0) >= mtimes.get("metrics.jsonl", 0) - 1:
        return PHASES.get(status.get("phase"), ("stopped", str(status.get("phase", "Finished")).replace("_", " ").capitalize()))
    if newest is None or not run["events"]:
        return "empty", "No events yet"
    age = now - newest
    if age > quiet_after:
        return "quiet", f"No events for {_duration(age)}"
    return "running", "Running"


def _published(method, results=RESULTS):
    """Validation curve of the published run of the same method, if this checkout has one."""
    if not method or results is None:
        return None
    record = read_json(Path(results) / f"{CARD_ALIASES.get(method, method)}.json")
    if not record or not record.get("val"):
        return None
    return dict(name=record.get("name") or method, val=[[point["step"], point["value"]] for point in record["val"]])


def _spike(values, factor):
    """Index of the last value above ``factor`` times the median of the twenty before it, or None."""
    for index in range(len(values) - 1, 20, -1):
        reference = _median(values[index - 20:index])
        if reference and _finite(values[index]) and values[index] > factor * reference:
            return index
    return None


def summarize(identifier: str, run: dict, now: float | None = None, peak_tflops: float | None = None,
              results=RESULTS) -> dict:
    """A JSON-ready summary of one run: identity, progress, estimates, series for the charts and health findings."""
    now = time.time() if now is None else now
    provenance, events = run["provenance"], run["events"]
    plan = schedule(provenance)
    world = provenance.get("world_size") or 8
    train = [event for event in events if event.get("event") == "train" and _finite(event.get("step"))]
    validation = [event for event in events if event.get("event") == "validation" and event.get("split", "val") == "val"]
    core = [event for event in events if event.get("event") == "core"]
    anomalies = [event for event in events if event.get("event") == "budget_anomaly"]
    transitions = [event for event in events if event.get("event") == "stage_transition"]
    final = next((event for event in reversed(events) if event.get("event") == "final_bpb"), None)
    last = train[-1] if train else {}
    step = max([int(last.get("step", 0)), int((run["status"] or {}).get("step") or 0), *run["checkpoints"]], default=0)

    # Update 1 includes compilation and warmup; it is not a steady sample.
    steady = [event for event in train if event["step"] > 1][-50:]
    step_seconds = _median(event.get("step_seconds") for event in steady)
    quiet_after = max(QUIET_FLOOR_S, 3 * plan["log_every"] * (step_seconds or 0))
    kind, label = _status(run, quiet_after, now)
    finished = kind in ("complete", "segment", "stopped")

    done_val = {int(event["step"]) for event in validation if _finite(event.get("step"))}
    done_core = {int(event["step"]) for event in core if _finite(event.get("step"))}
    val_seconds = _median(event.get("seconds") for event in validation)
    core_seconds = _median(event.get("seconds") for event in core)
    remaining_val = [s for s in plan["validation"] if s >= step and s not in done_val]
    remaining_core = [s for s in plan["core"] if s >= step and s not in done_core]
    estimate, excluded = None, []
    if not finished and step_seconds and step < plan["total"]:
        estimate = (plan["total"] - step) * step_seconds + len(remaining_val) * (val_seconds or 0)
        if remaining_val and val_seconds is None:
            excluded.append("validation")
        if remaining_core:
            if core_seconds is None:
                excluded.append("CORE")
            else:
                estimate += len(remaining_core) * core_seconds
        excluded.append("checkpoints")

    # MFU from the audited cost charged per update; the trainer reports it only for GPUs in its peak table.
    mfu, previous = [], None
    for event in train:
        value = event.get("mfu_theory")
        if not _finite(value) and peak_tflops and previous and _finite(event.get("theory_flops")) \
                and _finite(previous.get("theory_flops")) and event["step"] > previous["step"] \
                and _finite(event.get("step_seconds")) and event["step_seconds"] > 0:
            cost = (event["theory_flops"] - previous["theory_flops"]) / (event["step"] - previous["step"])
            value = cost / (event["step_seconds"] * world * peak_tflops * 1e12)
        if _finite(value) and event["step"] > 1:
            mfu.append([event["step"], value])
        previous = event

    def series(key, scale=1.0):
        return [[event["step"], event[key] * scale] for event in train if _finite(event.get(key))]

    lm = series("lm_loss") or series("loss")
    aux = series("aux_loss")
    charts = dict(
        lm_loss=lm, aux_loss=aux if any(value for _, value in aux) else [], grad_norm=series("grad_norm"),
        step_seconds=[point for point in series("step_seconds") if point[0] > 1],
        tokens_per_second=[point for point in series("tokens_per_second") if point[0] > 1], mfu=mfu,
        drift=series("theory_relative_error", 100.0), memory=series("peak_memory_bytes", 1 / 2**30),
        lr=series("lr_multiplier"),
        val_bpb=[[event["step"], event["val_bpb"]] for event in validation if _finite(event.get("val_bpb"))],
        core=[[event["step"], event["core_metric"]] for event in core if _finite(event.get("core_metric"))])

    model = provenance.get("model") or {}
    method = model.get("method")
    config = provenance.get("config")
    name = Path(run["path"]).name
    if name == "run":
        name = Path(run["path"]).parent.name
    summary = dict(
        id=identifier, name=name, path=run["path"], method=method, config=Path(config).name if config else None,
        mode=(provenance.get("admission") or {}).get("mode") or ("research" if plan["smoke"] else None), smoke=plan["smoke"],
        world=world, micro_batch=provenance.get("micro_batch"), accumulation=provenance.get("accumulation"),
        parameters=provenance.get("parameters"), device=provenance.get("device"),
        cost=provenance.get("cost") or {}, status=kind, status_label=label,
        step=step, total=plan["total"], input_tokens=last.get("input_tokens"), schedule=dict(
            validation=plan["validation"], core=plan["core"], done_validation=sorted(done_val), done_core=sorted(done_core)),
        checkpoints=run["checkpoints"], latest=last, lm_loss=lm[-1][1] if lm else None,
        val_bpb=charts["val_bpb"][-1][1] if charts["val_bpb"] else None,
        core_metric=charts["core"][-1][1] if charts["core"] else None, final=final,
        step_seconds=step_seconds, tokens_per_second=_median(event.get("tokens_per_second") for event in steady),
        mfu=_median(value for _, value in mfu[-50:]), training_seconds=(last.get("gpu_hours") or 0) * 3600 / world if last else 0,
        gpu_hours=last.get("gpu_hours"), estimate_seconds=estimate, estimate_excludes=excluded,
        remaining_validation=len(remaining_val), remaining_core=len(remaining_core), validation_seconds=val_seconds,
        core_seconds=core_seconds, drift=last.get("theory_relative_error"), anomalies=anomalies[-20:],
        transitions=transitions, validations=validation, cores=core, updated=max(run["mtimes"].values(), default=None),
        malformed=run["malformed"], charts=charts, published=_published(method, results))
    summary["findings"] = diagnose(summary, plan, steady, quiet_after, now)
    return summary


def _finding(level, title, detail):
    return dict(level=level, title=title, detail=detail)


def diagnose(summary: dict, plan: dict, steady: list, quiet_after: float, now: float) -> list:
    """Health findings, most severe first. Every finding names the evidence it comes from."""
    findings = []
    latest, charts = summary["latest"], summary["charts"]
    bad = [key for key in ("loss", "lm_loss", "grad_norm") if key in latest and not _finite(latest[key])]
    if bad:
        findings.append(_finding("critical", "Non-finite training values",
                                 f"{', '.join(bad)} at update {latest.get('step')} is not a finite number."))
    drift, proof = summary["drift"], summary["cost"].get("relative_error")
    tolerance = plan["protocol"]
    # The audit judges the whole run. A staged method (burn-in, gate-only, sampled depth) starts away from the anchor
    # and reaches the proof's whole-run figure only at the end, so the tolerance applies to that figure when it exists.
    judged, basis = (proof, "the admitted proof averages") if _finite(proof) else (drift, "charged FLOPs per token differ by")
    if _finite(judged):
        if abs(judged) > tolerance["cumulative_tolerance"]:
            findings.append(_finding("critical", "Compute ledger outside the cumulative tolerance",
                                     f"Over the full run {basis} {judged:+.2%} from the anchor; an audit accepts at most "
                                     f"{tolerance['cumulative_tolerance']:.0%}."))
        elif abs(judged) > tolerance["preflight_tolerance"]:
            findings.append(_finding("serious", "Compute ledger outside the admission tolerance",
                                     f"Over the full run {basis} {judged:+.2%} from the anchor; admission allows "
                                     f"{tolerance['preflight_tolerance']:.0%}."))
    if _finite(proof) and _finite(drift) and abs(drift) > tolerance["preflight_tolerance"] \
            and abs(proof) <= tolerance["preflight_tolerance"]:
        findings.append(_finding("info", "Staged compute", f"Charged FLOPs per token so far are {drift:+.2%} from the "
                                 f"anchor; the admitted proof averages {proof:+.2%} over the full run, so earlier stages "
                                 "of this method cost more or less than later ones."))
    if summary["anomalies"]:
        event = summary["anomalies"][-1]
        findings.append(_finding("warning", f"{len(summary['anomalies'])} budget anomaly event(s)",
                                 f"The rolling cost of the last {tolerance.get('val_every', 250)} updates left the routing "
                                 f"warning band, latest at update {event.get('step')} "
                                 f"({event.get('rolling_relative_error', 0):+.2%})."))
    if summary["status"] == "quiet":
        pending = [s for s in summary["schedule"]["core"] if s == summary["step"] and s not in summary["schedule"]["done_core"]]
        detail = (f"metrics.jsonl has not changed for {_duration(now - summary['updated'])}" if summary["updated"] else
                  "metrics.jsonl has not changed")
        detail += "; CORE is scheduled at this update and can take a while." if pending else \
            f"; expected a train event every {_duration(quiet_after / 3)}. The process may have stopped."
        findings.append(_finding("warning", summary["status_label"], detail))
    losses = [value for _, value in charts["lm_loss"]]
    index = _spike(losses, 1.5)
    if index is not None:
        findings.append(_finding("warning", "Loss spike", f"Language-model loss {losses[index]:.4f} at update "
                                 f"{charts['lm_loss'][index][0]} is above 1.5 times the median of the 20 events before."))
    norms = [value for _, value in charts["grad_norm"]]
    index = _spike(norms, 4.0)
    if index is not None:
        findings.append(_finding("warning", "Gradient-norm spike", f"Gradient norm {norms[index]:.3g} at update "
                                 f"{charts['grad_norm'][index][0]} is above 4 times the median of the 20 events before."))
    seconds = [event.get("step_seconds") for event in steady]
    if len(seconds) >= 20:
        early, late = _median(seconds[:len(seconds) // 2]), _median(seconds[-10:])
        if early and late and late > 1.2 * early:
            findings.append(_finding("warning", "Updates are slowing down", f"The last 10 logged updates take "
                                     f"{late:.2f} s at the median, {late / early - 1:.0%} more than earlier in the window."))
    if summary["malformed"]:
        findings.append(_finding("warning", "Unreadable lines in metrics.jsonl",
                                 f"{summary['malformed']} line(s) are not JSON and were skipped."))
    if summary["smoke"] or summary["mode"] == "research":
        findings.append(_finding("info", "Research run", "Not trained with --formal; research runs are never published. "
                                 "Smoke validations use fewer tokens than formal ones." if summary["smoke"] else
                                 "Not trained with --formal; research runs are never published."))
    if not [f for f in findings if f["level"] in ("critical", "serious", "warning")] and summary["step"]:
        findings.insert(0, _finding("good", "No problems found", "Losses are finite, the compute ledger is inside its "
                                    "tolerance and events arrive on time."))
    order = dict(critical=0, serious=1, warning=2, good=3, info=4)
    return sorted(findings, key=lambda finding: order[finding["level"]])


# ------------------------------------------------------------------------------------------------------ formatting

def _text(value) -> str:
    return html.escape("" if value is None else str(value))


def _duration(seconds) -> str:
    if not _finite(seconds):
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _number(value, digits=4) -> str:
    return f"{value:.{digits}f}" if _finite(value) else "—"


def _count(value) -> str:
    return f"{int(value):,}" if _finite(value) else "—"


def _compact(value) -> str:
    if not _finite(value):
        return "—"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= size:
            scaled = value / size
            return f"{scaled:.0f}{suffix}" if scaled >= 100 or scaled == int(scaled) else f"{scaled:.1f}{suffix}"
    return f"{value:g}"


def _tick(value) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return _compact(value)
    if magnitude >= 10 or value == int(value):
        return f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    return f"{value:.3g}"


# ----------------------------------------------------------------------------------------------------------- charts

W, H = 760, 230
LEFT, TOP, BOTTOM = 58, 14, 32


def _nice(lo, hi, count=5):
    """Round tick values covering [lo, hi]."""
    if not (_finite(lo) and _finite(hi)) or hi <= lo:
        return [lo] if _finite(lo) else []
    raw = (hi - lo) / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    ticks, value = [], start
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _log_ticks(lo, hi):
    ticks = []
    for exponent in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10 ** exponent
            if lo <= value <= hi:
                ticks.append(value)
    if len(ticks) < 3:
        ticks = sorted({*ticks, *[value for value in _nice(lo, hi, 3) if value > 0]})
    return ticks


def line_chart(identifier, title, series, unit="", log=False, x_max=None, note="", bands=(), digits=4, height=H):
    """An SVG line chart with one y axis, a crosshair readout and a table view.

    ``series`` is a list of dicts with ``name``, ``points`` ([[step, value], ...]) and optional ``slot`` (1-8, the fixed
    categorical order), ``dashed`` and ``markers``. ``bands`` are horizontal reference ranges (``lo``, ``hi``, ``label``).
    """
    drawn = [dict(item, points=[p for p in item["points"] if _finite(p[1]) and (not log or p[1] > 0)])
             for item in series]
    drawn = [item for item in drawn if item["points"]]
    head = f'<figure class="chart" id="{_text(identifier)}"><figcaption><span class="chart-title">{_text(title)}</span>'
    if not drawn:
        return head + f'</figcaption><p class="empty">No data yet.</p></figure>'
    labelled = 2 <= len(drawn) <= 4
    right = 132 if labelled else 18
    plot_w, plot_h = W - LEFT - right, height - TOP - BOTTOM
    xs = [p[0] for item in drawn for p in item["points"]]
    ys = [p[1] for item in drawn for p in item["points"]] + [v for band in bands for v in (band["lo"], band["hi"])]
    x_lo, x_hi = 0, max(x_max or 0, max(xs), 1)
    y_lo, y_hi = min(ys), max(ys)
    if log:
        y_lo, y_hi = y_lo / 1.15, y_hi * 1.15
        ticks = _log_ticks(y_lo, y_hi)
        ly_lo, ly_hi = math.log10(y_lo), math.log10(y_hi)
        y_of = lambda v: TOP + plot_h - (math.log10(v) - ly_lo) / (ly_hi - ly_lo) * plot_h
    else:
        pad = (y_hi - y_lo) * 0.08 or abs(y_hi) * 0.05 or 1.0
        y_lo, y_hi = y_lo - pad, y_hi + pad
        ticks = [t for t in _nice(y_lo, y_hi) if y_lo <= t <= y_hi]
        y_of = lambda v: TOP + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h
    x_of = lambda v: LEFT + (v - x_lo) / (x_hi - x_lo) * plot_w

    parts = [f'<svg class="plot" viewBox="0 0 {W} {height}" role="img" aria-label="{_text(title)}">']
    for band in bands:
        top, bottom = y_of(min(max(band["hi"], y_lo), y_hi)), y_of(min(max(band["lo"], y_lo), y_hi))
        parts.append(f'<rect class="band" x="{LEFT}" y="{top:.1f}" width="{plot_w}" height="{max(bottom - top, 0):.1f}">'
                     f'<title>{_text(band["label"])}</title></rect>')
    for tick in ticks:
        y = y_of(tick)
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="axis" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{_text(_tick(tick))}</text>')
    for tick in _nice(x_lo, x_hi, 6):
        x = x_of(tick)
        parts.append(f'<text class="axis" x="{x:.1f}" y="{height - 10}" text-anchor="middle">{_text(_compact(tick))}</text>')
    parts.append(f'<line class="baseline" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{TOP + plot_h}" y2="{TOP + plot_h}"/>')
    ends = []
    for index, item in enumerate(drawn):
        slot = item.get("slot") or index + 1
        points = " ".join(f"{x_of(x):.1f},{y_of(y):.1f}" for x, y in item["points"])
        dash = ' stroke-dasharray="5 4"' if item.get("dashed") else ""
        if len(item["points"]) > 1:
            parts.append(f'<polyline class="series s{slot}" points="{points}"{dash}/>')
        if item.get("markers") or len(item["points"]) == 1:
            parts.extend(f'<circle class="dot s{slot}" cx="{x_of(x):.1f}" cy="{y_of(y):.1f}" r="4"/>' for x, y in item["points"])
        x, y = item["points"][-1]
        ends.append([y_of(y), x_of(x), item["name"]])
    if labelled:
        # Direct labels at the line ends, nudged apart so they never overlap.
        ends.sort()
        for index in range(1, len(ends)):
            ends[index][0] = max(ends[index][0], ends[index - 1][0] + 17)
        for y, _, name in ends:
            parts.append(f'<text class="label" x="{LEFT + plot_w + 8}" y="{min(y, TOP + plot_h) + 4:.1f}">{_text(name)}</text>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{TOP}" y2="{TOP + plot_h}"/></svg>')

    legend = ""
    if len(drawn) >= 2:
        legend = '<div class="legend">' + "".join(
            f'<span><i class="key s{item.get("slot") or i + 1}{" dashed" if item.get("dashed") else ""}"></i>{_text(item["name"])}</span>'
            for i, item in enumerate(drawn)) + "</div>"
    data = dict(lo=x_lo, hi=x_hi, left=LEFT, width=plot_w, total=W, unit=unit, digits=digits,
                series=[dict(name=item["name"], points=item["points"]) for item in drawn])
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    caption = f'<span class="readout" aria-live="polite"></span></figcaption>'
    footer = f'<p class="chart-note">{_text(note)}</p>' if note else ""
    return (head + caption + legend + "".join(parts) + footer + _chart_table(drawn, unit, digits) +
            f'<script type="application/json" class="chart-data">{payload}</script></figure>')


def _chart_table(series, unit, digits, rows=40):
    """Table view: every point of sparse series, an even sample of dense ones, always the last point."""
    steps = sorted({p[0] for item in series for p in item["points"]})
    if len(steps) > rows:
        stride = math.ceil(len(steps) / rows)
        steps = sorted({*steps[::stride], steps[-1]})
    lookup = [{p[0]: p[1] for p in item["points"]} for item in series]
    head = "".join(f"<th>{_text(item['name'])}</th>" for item in series)
    body = "".join("<tr><td>" + _count(step) + "</td>" + "".join(
        f"<td>{_number(values[step], digits) if step in values else ''}</td>" for values in lookup) + "</tr>" for step in steps)
    unit_note = f" ({_text(unit)})" if unit else ""
    return (f'<details class="table-view"><summary>Table view{unit_note}</summary><table><thead><tr><th>Update</th>{head}'
            f"</tr></thead><tbody>{body}</tbody></table></details>")


def contract_track(summary: dict) -> str:
    """Progress against the run's schedule: every validation event above the track, CORE events below it."""
    total, step = max(summary["total"], 1), summary["step"]
    plan = summary["schedule"]
    width, left, right = 1000, 10, 10
    span = width - left - right
    x_of = lambda s: left + min(max(s, 0), total) / total * span
    done_val, done_core = set(plan["done_validation"]), set(plan["done_core"])
    parts = [f'<svg class="track" viewBox="0 0 {width} 78" role="img" aria-label="Update {step} of {total}">',
             f'<rect class="track-bed" x="{left}" y="28" width="{span}" height="10" rx="5"/>',
             f'<rect class="track-fill" x="{left}" y="28" width="{max(x_of(step) - left, 0):.1f}" height="10" rx="5"/>']
    for s in plan["validation"]:
        state = "done" if s in done_val else "missed" if s < step else "pending"
        parts.append(f'<line class="val-tick {state}" x1="{x_of(s):.1f}" x2="{x_of(s):.1f}" y1="12" y2="24">'
                     f'<title>Validation at update {s:,}: {state}</title></line>')
    for s in plan["core"]:
        state = "done" if s in done_core else "missed" if s < step else "pending"
        x = x_of(s)
        parts.append(f'<circle class="core-dot {state}" cx="{x:.1f}" cy="50" r="5"><title>CORE at update {s:,}: {state}'
                     f'</title></circle><text class="axis" x="{min(max(x, 24), width - 24):.1f}" y="72" '
                     f'text-anchor="middle">{_text(_compact(s))}</text>')
    x = x_of(step)
    parts.append(f'<line class="now" x1="{x:.1f}" x2="{x:.1f}" y1="6" y2="44"/></svg>')
    legend = ('<div class="legend"><span><i class="key tick"></i>Validation event</span>'
              '<span><i class="key ring"></i>CORE event</span><span><i class="key fill"></i>Done</span></div>')
    return "".join(parts) + legend


# ------------------------------------------------------------------------------------------------------------ pages

ICONS = dict(critical="✖", serious="▲", warning="▲", good="✔", info="ℹ", running="●", quiet="◐", complete="✔",
             segment="‖", stopped="■", empty="○")
STATUS_TONE = dict(running="good", quiet="warning", complete="good", segment="info", stopped="serious", empty="info")


def _pill(kind, label):
    return f'<span class="pill {STATUS_TONE.get(kind, "info")}"><span aria-hidden="true">{ICONS.get(kind, "")}</span> {_text(label)}</span>'


def _tiles(items):
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><span class="tile-label">{_text(label)}</span><span class="tile-value">{_text(value)}</span>'
        f'<span class="tile-note">{_text(note)}</span></div>' for label, value, note in items) + "</div>"


def _section(identifier, title, body, lede=""):
    lede = f'<p class="lede">{_text(lede)}</p>' if lede else ""
    return f'<section class="panel" id="{identifier}"><h2>{_text(title)}</h2>{lede}{body}</section>'


def _estimate(summary) -> tuple[str, str]:
    if summary["status"] in ("complete", "segment", "stopped"):
        return "—", summary["status_label"]
    if summary["status"] == "quiet":
        return "—", "no recent events"
    if summary["estimate_seconds"] is None:
        return "—", "needs two logged updates"
    excluded = ", ".join(summary["estimate_excludes"])
    finish = time.strftime("%b %d, %H:%M", time.localtime(time.time() + summary["estimate_seconds"]))
    return _duration(summary["estimate_seconds"]), f"around {finish}; excludes {excluded}" if excluded else f"around {finish}"


def _findings(summary):
    rows = "".join(f'<li class="finding {f["level"]}"><span class="icon" aria-hidden="true">{ICONS[f["level"]]}</span>'
                   f'<div><strong>{_text(f["title"])}</strong><p>{_text(f["detail"])}</p></div></li>'
                   for f in summary["findings"])
    return f'<ul class="findings">{rows}</ul>'


def _facts(summary):
    cost = summary["cost"]
    rows = [("Method", summary["method"]), ("Configuration", summary["config"]), ("Mode", summary["mode"]),
            ("Ranks", summary["world"]), ("Micro-batch", summary["micro_batch"]),
            ("Micro-batches per update", summary["accumulation"]),
            ("Parameters", _compact(summary["parameters"]) if summary["parameters"] else None), ("Device", summary["device"]),
            ("Charged GFLOPs per token", _number(cost["flops_per_token"] / 1e9, 3) if cost.get("flops_per_token") else None),
            ("Checkpoints", ", ".join(_compact(s) for s in summary["checkpoints"][-6:]) or None),
            ("Run directory", summary["path"])]
    return '<dl class="facts">' + "".join(f"<div><dt>{_text(k)}</dt><dd>{_text(v)}</dd></div>" for k, v in rows if v) + "</dl>"


def _evaluations(summary):
    rows = [(event["step"], "Validation BPB", _number(event.get("val_bpb")), event.get("input_tokens"), event.get("seconds"))
            for event in summary["validations"]]
    rows += [(event["step"], "CORE", _number(event.get("core_metric")), None, event.get("seconds")) for event in summary["cores"]]
    final = summary["final"]
    if final:
        rows.append((final["step"], "Final BPB (val / train)", f"{_number(final.get('val_bpb'))} / {_number(final.get('train_bpb'))}",
                     None, None))
    if not rows:
        return '<p class="empty">No evaluations yet.</p>'
    body = "".join(f"<tr><td>{_count(s)}</td><td>{_text(kind)}</td><td>{value}</td><td>{_compact(tokens) if tokens else ''}</td>"
                   f"<td>{_duration(seconds) if seconds else ''}</td></tr>" for s, kind, value, tokens, seconds in
                   sorted(rows, key=lambda row: (-row[0], row[1]))[:60])
    return ("<table><thead><tr><th>Update</th><th>Event</th><th>Value</th><th>Tokens</th><th>Time</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def render_run(summary: dict, base="/", refresh=10) -> str:
    charts = summary["charts"]
    total = summary["total"]
    estimate, estimate_note = _estimate(summary)
    share = summary["step"] / total if total else 0
    head = (f'<header class="masthead"><p class="crumb"><a href="{base}">All runs</a></p><h1>{_text(summary["name"])}</h1>'
            f'<p class="headline">{_pill(summary["status"], summary["status_label"])} Update {_count(summary["step"])} of '
            f'{_count(total)}, {share:.1%} of the {"smoke" if summary["smoke"] else "protocol"} budget.</p></header>')
    tiles = _tiles([
        ("Language-model loss", _number(summary["lm_loss"]), f"update {_count(summary['latest'].get('step'))}"
         if summary["latest"] else "no train event yet"),
        ("Validation BPB", _number(summary["val_bpb"]), f"{len(summary['validations'])} of {len(summary['schedule']['validation'])} events"),
        ("Throughput", f"{_compact(summary['tokens_per_second'])} tok/s" if summary["tokens_per_second"] else "—",
         f"{_number(summary['step_seconds'], 2)} s per update" if summary["step_seconds"] else "steady updates only"),
        ("Time left", estimate, estimate_note)])
    overview = (f'<section class="overview">{tiles}{contract_track(summary)}</section>')
    # Charts span the updates logged so far; the track above shows the whole budget.
    extent = max(summary["step"], 1)
    published = summary["published"]
    val_series = [dict(name="This run", points=charts["val_bpb"], slot=1, markers=True)]
    reference = [point for point in (published or {}).get("val", []) if point[0] <= extent]
    if len(reference) >= 2 and not summary["smoke"]:
        val_series.append(dict(name=f"Published {published['name']}", points=reference, slot=2, dashed=True))
    tolerance = 100 * DEFAULT_PROTOCOL["preflight_tolerance"]
    drift_series = [dict(name="Charged so far", points=charts["drift"], slot=1)]
    proof = summary["cost"].get("relative_error")
    if _finite(proof) and charts["drift"]:
        drift_series.append(dict(name="Admitted proof, full run", slot=2, dashed=True,
                                 points=[[0, 100 * proof], [extent, 100 * proof]]))
    sections = [
        _section("health", "Health", _findings(summary)),
        _section("loss", "Training", line_chart("lm-loss", "Language-model loss", [dict(name="Language-model loss",
                 points=charts["lm_loss"])], log=True) +
                 (line_chart("aux-loss", "Auxiliary loss", [dict(name="Auxiliary loss", points=charts["aux_loss"])])
                  if charts["aux_loss"] else "") +
                 line_chart("grad-norm", "Gradient norm", [dict(name="Gradient norm", points=charts["grad_norm"])], log=True,
                            digits=3) +
                 line_chart("lr", "Learning-rate multiplier", [dict(name="Learning-rate multiplier", points=charts["lr"])],
                            digits=3, height=170)),
        _section("evaluation", "Evaluation",
                 line_chart("val-bpb", "Validation bits per byte", val_series,
                            note="Smoke validations use fewer tokens than formal ones and are noisier." if summary["smoke"] else "")
                 + (line_chart("core", "CORE-22", [dict(name="CORE", points=charts["core"], markers=True)])
                    if charts["core"] else "") + _evaluations(summary)),
        _section("speed", "Speed", line_chart("step-seconds", "Seconds per update", [dict(name="Seconds per update",
                 points=charts["step_seconds"])], unit="s", digits=2) +
                 line_chart("tokens", "Training tokens per second", [dict(name="Tokens per second",
                            points=charts["tokens_per_second"])], digits=0) +
                 (line_chart("mfu", "Model FLOPs utilization", [dict(name="MFU", points=[[s, 100 * v] for s, v in charts["mfu"]])],
                             unit="%", digits=1) if charts["mfu"] else
                  '<p class="chart-note">MFU needs the GPU peak: the trainer knows it only for some data-centre GPUs. '
                  'Pass --peak-tflops to the monitor.</p>') +
                 line_chart("memory", "Peak allocated memory", [dict(name="Peak memory", points=charts["memory"])],
                            unit="GiB", digits=2)),
        _section("ledger", "Compute ledger", line_chart(
            "drift", "Charged FLOPs per token versus the anchor", drift_series,
            unit="%", digits=3, bands=[dict(lo=-tolerance, hi=tolerance, label="Admission tolerance")],
            note="The shaded band is the 1% admission tolerance; a finished run must stay within 2% cumulatively. "
                 "Staged methods start away from the anchor and approach the proof's full-run figure."),
            "Cumulative training FLOPs the trainer charged per input token, relative to the SSSL anchor."),
        _section("details", "Run details", _facts(summary))]
    nav = "".join(f'<a href="#{key}">{label}</a>' for key, label in (
        ("health", "Health"), ("loss", "Training"), ("evaluation", "Evaluation"), ("speed", "Speed"),
        ("ledger", "Ledger"), ("details", "Details")))
    return _page(summary["name"], head + overview + "".join(sections), nav, refresh)


def render_index(summaries: list, link, compare_link="compare", refresh=10) -> str:
    counts = {kind: sum(1 for s in summaries if s["status"] == kind) for kind in ("running", "quiet", "complete")}
    tiles = _tiles([("Running", str(counts["running"]), "events in the last minutes"),
                    ("Quiet", str(counts["quiet"]), "no recent events"),
                    ("Finished", str(counts["complete"]), "complete or smoke runs"),
                    ("Watched", str(len(summaries)), "run directories")])
    if not summaries:
        body = '<p class="empty">No run directories found. Pass a run directory or a folder of runs.</p>'
    else:
        rows = []
        for s in summaries:
            share = s["step"] / s["total"] if s["total"] else 0
            estimate, _ = _estimate(s)
            worst = next((f for f in s["findings"] if f["level"] in ("critical", "serious", "warning")
                          and f["title"] != s["status_label"]), None)
            flag = (f' <span class="flag {worst["level"]}" title="{_text(worst["title"])}"><span aria-hidden="true">'
                    f'{ICONS[worst["level"]]}</span> {_text(worst["title"])}</span>') if worst else ""
            rows.append(
                f'<tr><td><input type="checkbox" name="run" value="{_text(s["id"])}" aria-label="Compare {_text(s["name"])}"></td>'
                f'<td><a href="{link(s["id"])}">{_text(s["name"])}</a><div class="sub">{_text(s["id"])}</div></td>'
                f'<td>{_text(s["method"] or "—")}</td><td>{_pill(s["status"], s["status_label"])}{flag}</td>'
                f'<td><div class="mini"><span style="width:{100 * min(share, 1):.1f}%"></span></div>'
                f'<div class="sub">{_count(s["step"])} / {_count(s["total"])}</div></td>'
                f'<td class="num">{_number(s["lm_loss"])}</td><td class="num">{_number(s["val_bpb"])}</td>'
                f'<td class="num">{_compact(s["tokens_per_second"])}</td><td class="num">{_text(estimate)}</td>'
                f'<td class="num">{_text(_duration(time.time() - s["updated"]) + " ago" if s["updated"] else "—")}</td></tr>')
        body = (f'<form action="{compare_link}" method="get" class="runs"><table><thead><tr><th></th><th>Run</th><th>Method</th>'
                '<th>Status</th><th>Progress</th><th class="num">LM loss</th><th class="num">Val BPB</th>'
                '<th class="num">Tok/s</th><th class="num">Time left</th><th class="num">Last event</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table><button type="submit">Compare selected runs</button></form>')
    head = ('<header class="masthead"><h1>Training runs</h1><p class="headline">Live view of run directories, read from the '
            'files the trainer writes. Nothing here writes to a run.</p></header>')
    return _page("Training runs", head + f'<section class="overview">{tiles}</section>' + _section("runs", "Runs", body),
                 "", refresh)


def render_compare(summaries: list, link, base="/", refresh=10) -> str:
    summaries = summaries[:MAX_COMPARE]
    head = (f'<header class="masthead"><p class="crumb"><a href="{base}">All runs</a></p><h1>Compare runs</h1>'
            f'<p class="headline">{len(summaries)} run(s). Choose among candidates with training loss or the dev split; the '
            'validation shard and CORE are reserved for formal runs.</p></header>')
    if not summaries:
        return _page("Compare runs", head + '<p class="empty">Select runs on the overview page.</p>', "", refresh)
    total = max(s["total"] for s in summaries)

    def series(key, scale=1.0):
        return [dict(name=s["name"], slot=i + 1, points=[[x, y * scale] for x, y in s["charts"][key]])
                for i, s in enumerate(summaries)]

    common = None
    for s in summaries:
        steps = {x for x, _ in s["charts"]["lm_loss"]}
        common = steps if common is None else common & steps
    matched = max(common) if common else None
    reference = None
    rows = []
    for i, s in enumerate(summaries):
        values = dict(map(tuple, s["charts"]["lm_loss"]))
        value = values.get(matched)
        if i == 0:
            reference = value
        delta = value - reference if _finite(value) and _finite(reference) and i else None
        rows.append(f'<tr><td><i class="key s{i + 1}"></i> <a href="{link(s["id"])}">{_text(s["name"])}</a></td>'
                    f'<td>{_text(s["method"] or "—")}</td><td class="num">{_number(value)}</td>'
                    f'<td class="num">{_number(delta) if delta is not None else "—"}</td>'
                    f'<td class="num">{_number(s["step_seconds"], 2)}</td><td class="num">{_count(s["step"])}</td></tr>')
    table = (f'<table><thead><tr><th>Run</th><th>Method</th><th class="num">LM loss at update {_count(matched)}</th>'
             f'<th class="num">Δ vs first run</th><th class="num">s per update</th><th class="num">Updates</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>' if matched is not None else
             '<p class="empty">The runs share no logged update yet.</p>')
    body = _section("matched", "Matched update", table, "Training loss at the last update every run has logged; the "
                    "first selected run is the reference.")
    body += _section("curves", "Curves", line_chart("cmp-loss", "Language-model loss", series("lm_loss"), log=True) +
                     line_chart("cmp-speed", "Seconds per update", series("step_seconds"), unit="s", digits=2) +
                     line_chart("cmp-memory", "Peak allocated memory", series("memory"), unit="GiB", digits=2))
    return _page("Compare runs", head + body, "", refresh)


LIGHT = ("--page:#eef1f5;--surface:#ffffff;--raise:#f5f7fa;--ink:#0f1722;--ink-2:#465263;--ink-3:#6f7b8b;--line:#e2e7ee;"
         "--grid:#e6eaf0;--axis:#c3c9d2;--track:#e3e8ef;--band:rgba(42,120,214,.10);"
         "--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948")
DARK = ("color-scheme:dark;--page:#0c1218;--surface:#141b24;--raise:#1a2330;--ink:#edf1f6;--ink-2:#b2bdca;--ink-3:#7e8a99;"
        "--line:#243040;--grid:#1f2a37;--axis:#344255;--track:#1d2733;--band:rgba(57,135,229,.16);"
        "--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767")
STATUS = "--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b"

STYLE = (":root{color-scheme:light;" + LIGHT + ";" + STATUS + ";--font:'Avenir Next','Segoe UI Variable Text','Segoe UI',"
         "system-ui,-apple-system,sans-serif}"
         "@media (prefers-color-scheme:dark){:root:where(:not([data-theme='light'])){" + DARK + "}}"
         ":root[data-theme='dark']{" + DARK + "}" + """
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--font);-webkit-font-smoothing:antialiased}
a{color:var(--s1)} :focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.shell{max-width:1120px;margin:0 auto;padding:22px 28px 64px}
.topbar{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;align-items:center;gap:16px;
  margin:0 -28px 24px;padding:11px 28px;background:color-mix(in oklab,var(--page) 88%,transparent);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line);font-size:.86rem}
.topbar nav{display:flex;gap:18px;flex-wrap:wrap} .topbar nav a{color:var(--ink-2);text-decoration:none}
.topbar nav a:hover{color:var(--ink)} .topbar .right{display:flex;gap:10px;align-items:center;color:var(--ink-3)}
.theme,.runs button{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:999px;
  padding:5px 14px;font:inherit;cursor:pointer}
.runs button{margin-top:14px;color:var(--ink);font-weight:600}
.masthead{margin:6px 0 20px} .masthead h1{font-size:2.1rem;line-height:1.12;font-weight:650;letter-spacing:-.02em;margin:0 0 8px;
  word-break:break-word} .crumb{margin:0 0 6px;font-size:.86rem} .crumb a{color:var(--ink-3);text-decoration:none}
.headline{font-size:1.08rem;color:var(--ink-2);margin:0;max-width:72ch;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.overview,.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px}
.overview{padding:20px 24px 14px;margin-bottom:18px} .panel{padding:20px 24px 16px;margin:0 0 18px}
.panel h2{font-size:1.15rem;margin:0 0 4px;font-weight:650} .lede{color:var(--ink-3);margin:0 0 12px;font-size:.9rem}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:12px}
.tile{padding:0 18px;border-left:1px solid var(--line);display:flex;flex-direction:column;min-width:0}
.tile:first-child{border-left:0;padding-left:0} .tile-label{font-size:.8rem;color:var(--ink-3)}
.tile-value{font-size:1.7rem;font-weight:650;letter-spacing:-.015em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile-note{font-size:.78rem;color:var(--ink-3)}
.pill{display:inline-flex;gap:6px;align-items:center;border-radius:999px;padding:2px 11px;font-size:.84rem;font-weight:600;
  background:color-mix(in oklab,var(--tone) 16%,transparent);color:var(--ink);border:1px solid color-mix(in oklab,var(--tone) 45%,transparent)}
.pill span{color:var(--tone)} .flag{display:block;font-size:.78rem;color:var(--ink-2);margin-top:3px} .flag span{color:var(--tone)}
.good{--tone:var(--good)} .warning{--tone:var(--warning)} .serious{--tone:var(--serious)} .critical{--tone:var(--critical)}
.info{--tone:var(--ink-3)}
.track{width:100%;height:auto;display:block;margin-top:6px} .track-bed{fill:var(--track)} .track-fill{fill:var(--s1)}
.val-tick{stroke:var(--axis);stroke-width:1.2} .val-tick.done{stroke:var(--s1)} .val-tick.missed{stroke:var(--critical)}
.core-dot{fill:var(--surface);stroke:var(--axis);stroke-width:2} .core-dot.done{fill:var(--s1);stroke:var(--surface)}
.core-dot.missed{stroke:var(--critical)} .now{stroke:var(--ink);stroke-width:2}
.findings{list-style:none;margin:0;padding:0;display:grid;gap:10px}
.finding{display:flex;gap:12px;align-items:flex-start;padding:10px 14px;border-radius:10px;background:var(--raise);
  border-left:3px solid var(--tone)} .finding .icon{color:var(--tone);font-size:1rem;line-height:1.5}
.finding p{margin:2px 0 0;color:var(--ink-2);font-size:.9rem}
.chart{margin:14px 0 18px} .chart figcaption{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
.chart-title{font-weight:600;font-size:.95rem} .readout{font-size:.84rem;color:var(--ink-2);min-height:1.3em}
.readout b{color:var(--ink);font-weight:650}
.plot{width:100%;height:auto;display:block;touch-action:none} .grid{stroke:var(--grid);stroke-width:1}
.baseline{stroke:var(--axis);stroke-width:1} .axis{fill:var(--ink-3);font-size:11px;font-variant-numeric:tabular-nums}
.label{fill:var(--ink-2);font-size:11.5px} .band{fill:var(--band)}
.series{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round} .dot{stroke:var(--surface);stroke-width:2}
.crosshair{stroke:var(--ink-3);stroke-width:1;opacity:0}
.s1{stroke:var(--s1);--key:var(--s1)} .s2{stroke:var(--s2);--key:var(--s2)} .s3{stroke:var(--s3);--key:var(--s3)}
.s4{stroke:var(--s4);--key:var(--s4)} .s5{stroke:var(--s5);--key:var(--s5)} .s6{stroke:var(--s6);--key:var(--s6)}
.s7{stroke:var(--s7);--key:var(--s7)} .s8{stroke:var(--s8);--key:var(--s8)}
circle.s1{fill:var(--s1)} circle.s2{fill:var(--s2)} circle.s3{fill:var(--s3)} circle.s4{fill:var(--s4)}
circle.s5{fill:var(--s5)} circle.s6{fill:var(--s6)} circle.s7{fill:var(--s7)} circle.s8{fill:var(--s8)}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.82rem;color:var(--ink-2);margin:4px 0}
.legend span{display:inline-flex;align-items:center;gap:7px}
.key{display:inline-block;width:16px;height:0;border-top:2px solid var(--key,var(--ink-3))} .key.dashed{border-top-style:dashed}
.key.tick{width:2px;height:12px;border:0;background:var(--s1)} .key.ring{width:10px;height:10px;border:2px solid var(--axis);border-radius:50%}
.key.fill{width:16px;height:8px;border:0;border-radius:4px;background:var(--s1)}
.chart-note,.empty{color:var(--ink-3);font-size:.84rem;margin:6px 0 0}
.table-view summary{cursor:pointer;color:var(--ink-3);font-size:.82rem;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:.86rem} th{text-align:left;color:var(--ink-3);font-weight:600}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top} .num,td:not(:first-child){font-variant-numeric:tabular-nums}
th.num,td.num{text-align:right} .sub{color:var(--ink-3);font-size:.76rem;word-break:break-all}
.mini{width:120px;height:6px;border-radius:3px;background:var(--track);margin-top:7px;overflow:hidden}
.mini span{display:block;height:100%;background:var(--s1);border-radius:3px}
.runs{overflow-x:auto}
.facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 28px;margin:0;font-size:.88rem}
.facts div{display:flex;flex-direction:column} .facts dt{color:var(--ink-3)} .facts dd{margin:0;word-break:break-all}
.footer{color:var(--ink-3);font-size:.8rem;margin-top:28px}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr));row-gap:14px}.tile:nth-child(3){border-left:0;padding-left:0}
  .tile-value{font-size:1.3rem}
  .facts{grid-template-columns:1fr}.shell{padding:16px 16px 48px}.topbar{margin:0 -16px 18px;padding:10px 16px}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
""")

SCRIPT = """
(() => {
  const root = document.documentElement, button = document.querySelector('.theme');
  let stored = null; try { stored = localStorage.getItem('nt-monitor-theme'); } catch (error) {}
  if (stored) root.dataset.theme = stored;
  const dark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const label = () => { if (button) button.textContent = dark() ? 'Light theme' : 'Dark theme'; };
  if (button) button.addEventListener('click', () => { root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem('nt-monitor-theme', root.dataset.theme); } catch (error) {} label(); });
  label();
  const nearest = (points, x) => { let lo = 0, hi = points.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (points[mid][0] < x) lo = mid; else hi = mid; }
    return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? points[lo] : points[hi]; };
  const charts = () => document.querySelectorAll('figure.chart').forEach(figure => {
    const svg = figure.querySelector('svg.plot'), node = figure.querySelector('.chart-data');
    if (!svg || !node || svg.dataset.bound) return;
    svg.dataset.bound = '1';
    const data = JSON.parse(node.textContent), line = svg.querySelector('.crosshair'), out = figure.querySelector('.readout');
    const show = clientX => {
      const box = svg.getBoundingClientRect(), px = (clientX - box.left) / box.width * data.total;
      if (px < data.left || px > data.left + data.width) { hide(); return; }
      const step = data.lo + (px - data.left) / data.width * (data.hi - data.lo);
      const hits = data.series.map(s => [s.name, nearest(s.points, step)]);
      const at = hits[0][1][0], x = data.left + (at - data.lo) / (data.hi - data.lo) * data.width;
      line.setAttribute('x1', x); line.setAttribute('x2', x); line.style.opacity = 0.7;
      out.textContent = '';
      const head = document.createElement('span'); head.textContent = 'Update ' + Math.round(at).toLocaleString() + '  ';
      out.appendChild(head);
      hits.forEach(([name, point], index) => {
        const value = document.createElement('b');
        value.textContent = point[1].toFixed(data.digits) + (data.unit ? ' ' + data.unit : '');
        out.appendChild(value);
        out.appendChild(document.createTextNode(' ' + name + (point[0] !== at ? ' at ' + Math.round(point[0]).toLocaleString() : '')
          + (index < hits.length - 1 ? ',  ' : '')));
      });
    };
    const hide = () => { line.style.opacity = 0; out.textContent = ''; };
    svg.addEventListener('pointermove', event => show(event.clientX));
    svg.addEventListener('pointerleave', hide);
  });
  charts();
  const period = Number(document.body.dataset.refresh || 0);
  if (period > 0) setInterval(async () => {
    if (document.hidden || document.querySelector('figure.chart svg:hover')) return;
    try {
      const response = await fetch(location.href, { cache: 'no-store' });
      if (!response.ok) return;
      const next = new DOMParser().parseFromString(await response.text(), 'text/html').querySelector('main');
      const open = [...document.querySelectorAll('main details[open]')].map(node => node.closest('[id]')?.id);
      const checked = [...document.querySelectorAll('main input[type=checkbox]:checked')].map(node => node.value);
      document.querySelector('main').replaceWith(next);
      open.forEach(id => { const node = id && document.getElementById(id); if (node) node.querySelector('details')?.setAttribute('open', ''); });
      document.querySelectorAll('main input[type=checkbox]').forEach(node => { node.checked = checked.includes(node.value); });
      document.querySelector('.stamp').textContent = 'Updated ' + new Date().toLocaleTimeString();
      charts();
    } catch (error) {}
  }, period * 1000);
})();
"""


def _page(title, body, nav="", refresh=10) -> str:
    stamp = time.strftime("%H:%M:%S")
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>{_text(title)}</title><style>{STYLE}</style></head><body data-refresh="{int(refresh)}">'
            f'<div class="shell"><div class="topbar"><nav>{nav}</nav><div class="right"><span class="stamp">Updated {stamp}'
            f'</span><button class="theme" type="button">Dark theme</button></div></div><main>{body}</main>'
            '<p class="footer">Read-only view of run directories; it never writes to a run or takes its lock. Research '
            'runs are never published.</p></div>'
            f'<script>{SCRIPT}</script></body></html>\n')


# ----------------------------------------------------------------------------------------------------------- server

class Monitor:
    """Watched roots and options; every call scans the roots and reads the runs again."""

    def __init__(self, roots, depth=3, peak_tflops=None, results=RESULTS, refresh=10):
        self.roots, self.depth, self.peak_tflops, self.results, self.refresh = roots, depth, peak_tflops, results, refresh

    def summaries(self, ids=None):
        runs = find_runs(self.roots, self.depth)
        chosen = [i for i in ids if i in runs] if ids is not None else list(runs)
        now = time.time()
        result = [summarize(i, load_run(runs[i]), now, self.peak_tflops, self.results) for i in chosen]
        if ids is None:
            rank = dict(running=0, quiet=1, segment=2, empty=3, stopped=4, complete=5)
            result.sort(key=lambda s: (rank.get(s["status"], 9), -(s["updated"] or 0)))
        return result


def _handler(monitor: Monitor, quiet=True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if not quiet:
                super().log_message(format, *args)

        def _send(self, status, body, kind="text/html; charset=utf-8"):
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            path, query = unquote(url.path), parse_qs(url.query)
            link = lambda identifier: "/run/" + quote(identifier)
            try:
                if path == "/":
                    return self._send(200, render_index(monitor.summaries(), link, "/compare", monitor.refresh))
                if path == "/compare":
                    return self._send(200, render_compare(monitor.summaries(query.get("run", [])), link, "/", monitor.refresh))
                if path.startswith("/run/"):
                    found = monitor.summaries([path.removeprefix("/run/")])
                    if found:
                        return self._send(200, render_run(found[0], "/", monitor.refresh))
                if path == "/api/runs":
                    rows = [{k: v for k, v in s.items() if k not in ("charts", "validations", "cores")} for s in monitor.summaries()]
                    return self._send(200, json.dumps(rows), "application/json")
                if path.startswith("/api/run/"):
                    found = monitor.summaries([path.removeprefix("/api/run/")])
                    if found:
                        return self._send(200, json.dumps(found[0]), "application/json")
                self._send(404, _page("Not found", '<p class="empty">No such page or run.</p>', "", 0))
            except BrokenPipeError:
                pass

    return Handler


def serve(monitor: Monitor, host="127.0.0.1", port=8765, quiet=True):
    server = ThreadingHTTPServer((host, port), _handler(monitor, quiet))
    return server


def write_site(monitor: Monitor, output: Path) -> list[Path]:
    """Static pages of every run, the overview and a comparison of the first eight runs."""
    output.mkdir(parents=True, exist_ok=True)
    summaries = monitor.summaries()
    names = {s["id"]: "run-" + "".join(c if c.isalnum() or c in "-_" else "-" for c in s["id"]) + ".html" for s in summaries}
    link = lambda identifier: names[identifier]
    written = []
    for s in summaries:
        path = output / names[s["id"]]
        path.write_text(render_run(s, "index.html", 0))
        written.append(path)
    (output / "compare.html").write_text(render_compare(summaries, link, "index.html", 0))
    (output / "index.html").write_text(render_index(summaries, link, "compare.html", 0))
    return [output / "index.html", output / "compare.html", *written]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", type=Path, help="run directories or folders that contain them")
    parser.add_argument("--host", default="127.0.0.1", help="interface to serve on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--depth", type=int, default=3, help="folder levels searched below each root")
    parser.add_argument("--refresh", type=int, default=10, help="seconds between page refreshes (0: off)")
    parser.add_argument("--peak-tflops", type=float, help="dense BF16 peak per GPU, for MFU when the trainer reports none")
    parser.add_argument("--published", type=Path, default=RESULTS,
                        help="folder of published run records for reference curves (default: this checkout)")
    parser.add_argument("--once", type=Path, metavar="OUTPUT", help="write static pages to OUTPUT and exit")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)
    missing = [str(root) for root in args.roots if not root.is_dir()]
    if missing:
        parser.error("not a directory: " + ", ".join(missing))
    monitor = Monitor(args.roots, args.depth, args.peak_tflops, args.published if args.published.is_dir() else None,
                      args.refresh)
    if args.once:
        pages = write_site(monitor, args.once)
        print(f"wrote {len(pages)} page(s) to {args.once / 'index.html'}")
        return
    server = serve(monitor, args.host, args.port, not args.verbose)
    print(f"watching {len(find_runs(args.roots, args.depth))} run(s); open http://{args.host}:{server.server_address[1]}/",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
