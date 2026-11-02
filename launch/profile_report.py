"""Render profiler JSON as self-contained HTML: one page per profile and an index that compares profiles.

Pages need no network, server or external asset. Charts are inline SVG and CSS drawn from the summary in two hues:
blue for time spent on model FLOPs, orange for time lost beyond them. Light and dark themes follow the system and can
be switched. Every chart has a table view, and hover readouts never hold the only copy of a value.

    python -m launch.profile_report RUN/summary.json ... --output SITE_DIR
"""

from __future__ import annotations

from collections import defaultdict
import html
import json
import math
from pathlib import Path

from launch.profile_analysis import lagging_gpus


# Labels -----------------------------------------------------------------------------------------------------------

PHASE_LABELS = {"startup": "Startup", "warmup": "Warmup", "steady": "Steady", "traced": "Traced",
                "validation": "Validation", "core": "CORE", "checkpoint": "Checkpoint"}
PHASE_TONES = {"steady": "use", "traced": "use-2", "warmup": "loss-2", "startup": "idle", "validation": "loss-3",
               "core": "loss-3", "checkpoint": "loss"}
STAGE_LABELS = {"step_setup": "Step setup", "data": "Data", "forward": "Forward", "backward": "Backward",
                "grad_norm": "Gradient norm", "wait": "Host wait", "optimizer.reduce": "Optimizer reduce",
                "optimizer.compute": "Optimizer compute", "optimizer.gather": "Optimizer gather",
                "step_tail": "Step tail", "unattributed": "Unattributed"}
WATERFALL_PARTS = {"ideal": "Model FLOPs at peak", "gemm below peak": "GEMM below peak",
                   "attention below peak": "Attention below peak", "matrix kernels below peak": "Matrix kernels below peak",
                   "other compute kernels": "Other compute kernels", "exposed communication": "Exposed communication",
                   "exposed memory copies": "Exposed memory copies", "idle": "GPU idle"}
WATERFALL_NOTES = {
    "ideal": "the audited FLOPs of the update at the dense peak",
    "gemm below peak": "matrix multiplies run slower than peak",
    "attention below peak": "attention kernels run slower than peak",
    "matrix kernels below peak": "matrix multiplies and attention run slower than peak",
    "other compute kernels": "elementwise, normalization, casts, copies and the optimizer",
    "exposed communication": "collectives with no compute kernel alongside",
    "exposed memory copies": "copies and sets with no other kernel alongside",
    "idle": "no kernel runs: the host is late or the GPU waits on a dependency",
}
# Tones: "use" is time on model FLOPs, "loss" time beyond them, "idle" nothing running.
WATERFALL_TONES = {"ideal": "use", "gemm below peak": "loss-2", "attention below peak": "loss-2",
                   "matrix kernels below peak": "loss-2", "other compute kernels": "loss",
                   "exposed communication": "loss-3", "exposed memory copies": "loss-3", "idle": "idle"}


# Formatting -------------------------------------------------------------------------------------------------------

def _text(value) -> str:
    return html.escape(str(value))


def _seconds(value) -> str:
    if value is None:
        return "—"
    if abs(value) >= 100:
        return f"{value:,.0f} s"
    if abs(value) >= 1:
        return f"{value:,.2f} s"
    if abs(value) >= 1e-3:
        return f"{value * 1e3:,.1f} ms"
    return f"{value * 1e6:,.0f} µs"


def _percent(value, digits=1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def _tflops(value) -> str:
    return "—" if not value else f"{value / 1e12:,.1f} TFLOP/s"


def _gib(value) -> str:
    return "—" if value is None else f"{value / 2**30:,.2f} GiB"


def _amount(value, unit, suffix, digits=1, empty="—") -> str:
    return empty if not value else f"{value / unit:,.{digits}f} {suffix}".rstrip()


def _compact(value) -> str:
    for unit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= unit:
            return f"{value / unit:,.1f}{suffix}"
    return f"{value:,.0f}"


def _stage(name) -> str:
    return STAGE_LABELS.get(name, name)


def _tip(*parts) -> str:
    """A hover readout: value first, then its label."""
    return _text(" · ".join(str(part) for part in parts if part not in (None, "")))


# Chart primitives -------------------------------------------------------------------------------------------------

def _segments(parts, size="md", label_min=0.07):
    """A full-width stacked bar of ``(key, label, seconds, tone)`` parts with a 2px surface gap between segments."""
    total = sum(max(seconds, 0) for _, _, seconds, _ in parts) or 1.0
    items = []
    for key, label, seconds, tone in parts:
        share = max(seconds, 0) / total
        if share <= 0:
            continue
        inner = f'<span class="seg-label">{_text(_percent(share, 0))}</span>' if share >= label_min else ""
        items.append(f'<span class="seg tone-{tone}" data-part="{_text(key)}" style="flex-grow:{share:.5f}" tabindex="0" '
                     f'title="{_tip(_seconds(seconds), _percent(share), label)}">{inner}</span>')
    return f'<div class="stack stack-{size}" role="img" aria-label="Stacked bar">{"".join(items)}</div>'


def _legend(items):
    return '<ul class="legend">' + "".join(
        f'<li><span class="key tone-{tone}"></span>{_text(label)}</li>' for label, tone in items) + "</ul>"


def _bars(rows, value_format, max_value=None, tone_of=None, note_of=None, bar_label=""):
    """Horizontal bars of ``(label, value)``; labels and values in text ink, bars in their tone."""
    if not rows:
        return ""
    top = max_value or max(value for _, value in rows) or 1.0
    items = []
    for index, (label, value) in enumerate(rows):
        tone = tone_of(index, label) if tone_of else "use"
        note = note_of(index, label) if note_of else ""
        width = max(0.0, min(1.0, value / top)) * 100
        items.append(
            f'<div class="bar-row" tabindex="0" title="{_tip(value_format(value), label, note)}">'
            f'<span class="bar-label">{_text(label)}</span>'
            f'<span class="bar-track"><span class="bar tone-{tone}" style="width:{width:.3f}%"></span></span>'
            f'<span class="bar-value">{_text(value_format(value))}</span>'
            f'<span class="bar-note">{_text(note)}</span></div>')
    return f'<div class="bars" aria-label="{_text(bar_label)}">{"".join(items)}</div>'


def _split_bar_row(label, parts, top, value, note=""):
    """One bar made of ``(seconds, tone, name)`` parts, e.g. exposed and hidden communication."""
    spans = "".join(f'<span class="bar tone-{tone}" style="width:{100 * max(seconds, 0) / top:.3f}%"></span>'
                    for seconds, tone, _ in parts)
    detail = ", ".join(f"{_gib(seconds) if 'GiB' in value else _seconds(seconds)} {name}" for seconds, _, name in parts if seconds)
    return (f'<div class="bar-row" tabindex="0" title="{_tip(value, label, detail)}"><span class="bar-label">{_text(label)}</span>'
            f'<span class="bar-track">{spans}</span><span class="bar-value">{_text(value)}</span>'
            f'<span class="bar-note">{_text(note)}</span></div>')


def _meter(value, label, caption):
    """A half-ring gauge of a ratio between 0 and 1."""
    ratio = max(0.0, min(1.0, value or 0.0))
    radius, cx, cy = 80, 100, 96

    def point(fraction):
        angle = math.pi * (1 - fraction)
        return cx + radius * math.cos(angle), cy - radius * math.sin(angle)

    x0, y0 = point(0)
    x1, y1 = point(1)
    xv, yv = point(ratio)
    track = f"M {x0:.1f} {y0:.1f} A {radius} {radius} 0 0 1 {x1:.1f} {y1:.1f}"
    fill = f"M {x0:.1f} {y0:.1f} A {radius} {radius} 0 0 1 {xv:.1f} {yv:.1f}" if ratio > 0 else ""
    return (f'<figure class="meter"><svg viewBox="0 0 200 108" role="img" aria-label="{_text(label)} {_percent(value)}">'
            f'<path class="meter-track" d="{track}"/><path class="meter-fill" d="{fill}"/>'
            f'<text class="meter-value" x="100" y="90">{_text(_percent(value))}</text></svg>'
            f'<figcaption><b>{_text(label)}</b><span>{_text(caption)}</span></figcaption></figure>')


def _table(headers, rows, numeric=()):
    head = "".join(f'<th{" class=num" if index in numeric else ""}>{_text(header)}</th>' for index, header in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if index in numeric else ""}>{cell}</td>' for index, cell in enumerate(row))
                   + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table_view(headers, rows, numeric=(), title="Table view", identifier=None):
    attribute = f' id="{identifier}"' if identifier else ""
    return (f'<details class="table-view"{attribute}><summary>{_text(title)}</summary>'
            f'<div class="table-scroll">{_table(headers, rows, numeric)}</div></details>')


def _tiles(items, small=False):
    return f'<div class="tiles{" small" if small else ""}">' + "".join(
        f'<div class="tile"><span class="tile-label">{_text(label)}</span><span class="tile-value">{_text(value)}</span>'
        f'<span class="tile-note">{_text(note)}</span></div>' for label, value, note in items) + "</div>"


def _section(identifier, title, lede, body, extra_class=""):
    return (f'<section id="{identifier}" class="panel {extra_class}"><header><h2>{_text(title)}</h2>'
            + (f'<p class="lede">{lede}</p>' if lede else "") + f"</header>{body}</section>")


# Page head --------------------------------------------------------------------------------------------------------

def _title(summary):
    run = summary.get("run") or {}
    return run.get("method") or (Path(run["config"]).stem if run.get("config") else "Training profile")


def _headline(summary):
    steady = (summary.get("metrics_summary") or {}).get("steady") or {}
    mfu = summary.get("mfu_summary") or {}
    losses = [row for row in mfu.get("waterfall", []) if row["part"] != "ideal"]
    step = steady.get("median_step_seconds")
    if not step:
        return "The trainer did not reach a steady update, so this profile has no timing summary."
    sentence = f"An update takes {_seconds(step)}"
    if mfu.get("mfu") is not None:
        sentence += f" at {_percent(mfu['mfu'], 0)} MFU"
    if losses:
        worst = max(losses, key=lambda row: row["seconds"])
        sentence += f"; the largest loss is {WATERFALL_PARTS.get(worst['part'], worst['part']).lower()} ({_seconds(worst['seconds'])})"
    return sentence + "."


def _identity(summary):
    run = summary.get("run") or {}
    shape = summary.get("shape") or {}
    device = (summary.get("hardware") or {}).get("device") or {}
    software = summary.get("software") or {}
    facts = []
    if run.get("config"):
        facts.append(("Configuration", Path(run["config"]).name))
    if shape:
        facts.append(("Update", f'{shape["world"]} ranks × micro-batch {shape["micro_batch"]} × {shape["accumulation"]} '
                                f'= {shape["tokens_per_update"]:,} tokens per update'))
    if device.get("name"):
        facts.append(("GPU", f'{device["name"]}, sm{device.get("major")}{device.get("minor")}'))
    if software:
        facts.append(("Software", ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in software.items())))
    facts.append(("Status", str(summary.get("status", "unknown"))))
    return '<dl class="facts">' + "".join(f"<div><dt>{_text(label)}</dt><dd>{_text(value)}</dd></div>" for label, value in facts) + "</dl>"


def _detail(rows) -> str:
    if not rows:
        return ""
    return " (" + ", ".join(f'{_text(row["category"])} {_seconds(row["seconds"])}' for row in rows[:5]) + ")"


def _overview(summary):
    """Stat row and the hero bar: the traced update split into time on model FLOPs and each loss."""
    mfu = summary.get("mfu_summary") or {}
    waterfall = mfu.get("waterfall") or []
    steady = (summary.get("metrics_summary") or {}).get("steady") or {}
    shape = summary.get("shape") or {}
    flops = shape.get("model_flops_per_gpu_s")
    memory = (summary.get("memory_summary") or {}).get("reserved_peak") or steady.get("peak_memory_bytes")
    tiles = _tiles([("Update", _seconds(steady.get("median_step_seconds")), "median of steady updates"),
                    ("Throughput", f'{_compact(steady.get("median_tokens_per_second") or 0)} tok/s', "all ranks"),
                    ("MFU", _percent(mfu.get("mfu")), f'{flops / 1e12:,.1f} model TFLOP/s per GPU' if flops else "peak unknown"),
                    ("Peak memory", _gib(memory), "reserved, largest rank")])
    if not waterfall:
        return f'<div class="overview">{tiles}</div>'
    parts = [(row["part"], WATERFALL_PARTS.get(row["part"], row["part"]), max(row["seconds"], 0),
              WATERFALL_TONES.get(row["part"], "loss")) for row in waterfall]
    rows = [[f'<span class="key tone-{WATERFALL_TONES.get(row["part"], "loss")}"></span>{_text(WATERFALL_PARTS.get(row["part"], row["part"]))}',
             _seconds(max(row["seconds"], 0)), _percent(row["share"]),
             _text(WATERFALL_NOTES.get(row["part"], "")) + _detail(row.get("detail"))] for row in waterfall]
    return (f'<div class="overview" id="mfu-loss">{tiles}<figure class="hero-bar"><figcaption>Where the traced update goes '
            f'({_seconds(mfu.get("traced_window_s"))} on rank 0): model FLOPs at peak, then everything lost beyond them'
            f'</figcaption>{_segments(parts, size="lg")}{_legend([(label, tone) for _, label, _, tone in parts])}</figure>'
            f'<div class="table-scroll">{_table(["Part", "Time", "Share", "What it is"], rows, numeric=(1, 2))}</div></div>')


# Sections ---------------------------------------------------------------------------------------------------------

def _findings(summary):
    findings = summary.get("findings") or []
    if not findings:
        return ""
    items = "".join(
        f'<li class="finding {_text(finding["severity"])}" data-finding="{_text(finding["id"])}">'
        f'<span class="badge">{"Warning" if finding["severity"] == "warning" else "Note"}</span>'
        f'<div><b>{_text(finding["title"])}</b><p>{_text(finding["detail"])}</p></div></li>' for finding in findings)
    return _section("findings", "What to look at first", "Ordered by severity; each finding names the evidence it comes from.",
                    f'<ol class="findings">{items}</ol>')


def _phase_name(row):
    return _stage(row["phase"]) if row["phase"] in STAGE_LABELS else row["phase"].capitalize()


def _mfu(summary):
    mfu = summary.get("mfu_summary")
    if not mfu:
        return ""
    hardware = summary.get("hardware") or {}
    meters = [_meter(mfu.get("mfu"), "MFU", f'{_tflops(mfu["achieved_flops"])} per GPU of {_tflops(mfu.get("peak_flops"))} dense BF16')]
    if mfu.get("clock_mfu") is not None:
        meters.append(_meter(mfu["clock_mfu"], "MFU at the measured clock",
                             f'peak {_tflops(mfu.get("clock_peak_flops"))} at the mean SM clock of steady updates'))
    phases = [row for row in mfu["phases"] if row["mfu"] is not None]
    phase_bars = _bars([(_phase_name(row), row["mfu"]) for row in phases], _percent, max_value=1.0,
                       note_of=lambda index, label: f'{_seconds(phases[index]["seconds"])}, {_tflops(phases[index]["achieved_flops"])}',
                       bar_label="MFU by phase")
    kernels = [row for row in mfu.get("kernels", []) if row.get("efficiency") is not None]
    kernel_bars = _bars([("GEMM kernels" if row["kind"] == "gemm" else "Attention kernels", row["efficiency"]) for row in kernels],
                        _percent, max_value=1.0,
                        note_of=lambda index, label: f'{_seconds(kernels[index]["seconds"])} at {kernels[index]["tflops"]:,.1f} TFLOP/s',
                        bar_label="Matrix kernels against peak") if kernels else ""
    phase_rows = [[_text(_phase_name(row)), _seconds(row["seconds"]), _amount(row["flops"], 1e12, "TFLOP"),
                   _tflops(row["achieved_flops"]), _percent(row["mfu"])] for row in mfu["phases"]]
    kernel_rows = [[("GEMM" if row["kind"] == "gemm" else "Attention") + " kernels", _seconds(row["seconds"]),
                    _amount(row["flops"], 1e12, "TFLOP"), _amount(row["tflops"], 1, "TFLOP/s"), _percent(row["efficiency"]),
                    _text(row.get("flops_source", ""))] for row in mfu.get("kernels", [])]
    body = (f'<div class="meters">{"".join(meters)}</div><div class="split"><div><h3>MFU by phase</h3>{phase_bars}</div>'
            + (f'<div><h3>Matrix kernels against peak</h3>{kernel_bars}</div>' if kernel_bars else "") + "</div>"
            + _table_view(["Phase", "Time", "FLOPs per GPU", "Rate", "MFU"], phase_rows, numeric=(1, 2, 3, 4))
            + (_table_view(["Kernels", "Time", "FLOPs", "Rate", "Of peak", "FLOPs from"], kernel_rows, numeric=(1, 2, 3, 4),
                           title="Matrix kernels") if kernel_rows else ""))
    lede = (f'Audited model FLOPs of an update, the cost the trainer charges, over the dense BF16 tensor peak '
            f'({_text(hardware.get("basis", "unknown"))}). GEMM FLOPs come from traced shapes, attention FLOPs from the audit.')
    return _section("mfu", "Model FLOPs utilization", lede, body)


def _projection(summary):
    projection = summary.get("projection")
    if not projection or not projection.get("representative") or "estimated_hours" not in projection:
        return ""
    parts = [("training", "Training updates", projection["training_hours"], "use")]
    if "validation_hours" in projection:
        parts.append(("validation", "Validation", projection["validation_hours"], "use-2"))
    if "checkpoint_hours" in projection:
        parts.append(("checkpoint", "Checkpoints", projection["checkpoint_hours"], "loss-3"))
    total = projection["estimated_hours"]
    missing = "CORE-22 evaluation is not measured and comes on top."
    if "validation_hours" not in projection and "validation_events" in projection:
        missing = (f'Validation ({projection["validation_events"]} events) and CORE-22 are not included: this profile '
                   f'validated {projection.get("validation_measured_tokens", 0):,} tokens. Rerun with '
                   f'<code>--validation-tokens 41943040</code> to measure validation.')
    rows = [[_text(label), f"{hours:,.1f} h"] for _, label, hours, _ in parts]
    body = (f'<p class="big-number">{total:,.1f} h<span>≈ {total / 24:,.1f} days for {projection["updates"]:,} updates</span></p>'
            f'{_segments(parts, size="sm", label_min=0.2)}{_legend([(label, tone) for _, label, _, tone in parts])}'
            f'<p class="lede">{missing}</p>{_table_view(["Part", "Time"], rows, numeric=(1,))}')
    return _section("projection", "Formal run estimate", "From the median steady update on this machine.", body)


def _stage_tone(stage):
    if stage in ("forward", "backward"):
        return "use"
    if stage.startswith("optimizer") or stage == "grad_norm":
        return "use-2"
    return "idle"


def _stages(summary):
    stages = summary.get("stage_summary") or {}
    rows = stages.get("stages") or []
    if not rows:
        return _section("stages", "Where an update spends its time",
                        "Stage timings are unavailable: the trainer exited early or the profile predates the stage timer.", "")
    shown = [row for row in rows if row["seconds"] >= 1e-3]
    bars = _bars([(_stage(row["stage"]), row["seconds"]) for row in shown], _seconds,
                 tone_of=lambda index, label: _stage_tone(shown[index]["stage"]),
                 note_of=lambda index, label: f'{_percent(shown[index]["share"])} of the update',
                 bar_label="Stage time per update")
    ranks = sorted({rank for row in rows for rank, _ in row.get("per_rank", [])})
    grid = ""
    if len(ranks) > 1:
        lines = []
        for row in shown:
            values = dict(row.get("per_rank", []))
            low, spread = row["rank_min"], row["rank_max"] - row["rank_min"]
            cells = "".join(
                f'<span class="cell" style="--v:{(values.get(rank, low) - low) / spread if spread else 0:.3f}" '
                f'title="{_tip(_seconds(values.get(rank)), _stage(row["stage"]), f"rank {rank}")}"></span>' for rank in ranks)
            lines.append(f'<div class="grid-row"><span class="grid-label">{_text(_stage(row["stage"]))}</span>'
                         f'<span class="grid-cells" style="--n:{len(ranks)}">{cells}</span>'
                         f'<span class="grid-note">{_seconds(spread)}</span></div>')
        head = "".join(f"<span>{rank}</span>" for rank in ranks)
        grid = (f'<h3>Rank spread</h3><p class="lede">Within each stage, darker cells are slower ranks; the right column is '
                f'the spread from fastest to slowest. Every rank waits for the slowest at the next collective.</p>'
                f'<div class="rank-grid"><div class="grid-row head"><span class="grid-label">Rank</span>'
                f'<span class="grid-cells" style="--n:{len(ranks)}">{head}</span><span class="grid-note">Spread</span></div>'
                f'{"".join(lines)}</div>')
    table = [[_text(_stage(row["stage"])), _seconds(row["seconds"]), _percent(row["share"]), _seconds(row["host_seconds"]),
              _seconds(row["rank_min"]), _seconds(row["rank_max"]), str(row["slowest_rank"])] for row in rows]
    timing = "CUDA events on each rank's stream" if stages.get("timing") == "cuda_events" else "host clocks"
    lede = (f'Mean of steady updates {", ".join(map(str, stages["steady_steps"]))} over all ranks, timed with {timing}: '
            f'{_seconds(stages["step_seconds"])} per update. A stage runs from the trainer entering it to entering the next.')
    return _section("stages", "Where an update spends its time", lede,
                    bars + grid + _table_view(["Stage", "Time", "Share", "Host", "Fastest rank", "Slowest rank", "Slowest"],
                                              table, numeric=(1, 2, 3, 4, 5)))


def _timeline(trace):
    windows = (trace or {}).get("timeline") or []
    if not windows:
        return ""
    blocks = []
    for index, window in enumerate(windows):
        buckets = window["buckets"] or 1
        width, plot = 1000 / buckets, 60
        mix = window.get("mix") or []
        areas = ""
        if mix:
            matrix = " ".join(f"L {(i + 0.5) * width:.1f} {plot - plot * value[0]:.1f}" for i, value in enumerate(mix))
            both = " ".join(f"L {(i + 0.5) * width:.1f} {plot - plot * min(1, value[0] + value[1]):.1f}" for i, value in enumerate(mix))
            areas = (f'<path class="area tone-loss" d="M 0 {plot} {both} L 1000 {plot} Z"/>'
                     f'<path class="area tone-use" d="M 0 {plot} {matrix} L 1000 {plot} Z"/>')
        lanes = []
        for lane, top, height in (("stage", 70, 12), ("communication", 90, 9), ("memory", 105, 9)):
            for first, last, label, intensity in window["lanes"].get(lane, []):
                tone = _stage_tone(str(label)) if lane == "stage" else "loss" if lane == "communication" else "loss-3"
                lanes.append(f'<rect class="tone-{tone}" x="{first * width:.1f}" y="{top}" width="{max((last - first) * width - 1, 0.6):.1f}" '
                             f'height="{height}" rx="2" opacity="{0.45 + 0.55 * intensity:.2f}" data-lane="{lane}" data-label="{_text(label)}">'
                             f'<title>{_tip(_stage(label), lane)}</title></rect>')
        ticks = "".join(f'<text class="tick" x="{fraction * 1000:.0f}" y="132" text-anchor="{anchor}">'
                        f'{_text(_seconds(window["start_s"] + fraction * window["seconds"]))}</text>'
                        for fraction, anchor in ((0, "start"), (0.25, "middle"), (0.5, "middle"), (0.75, "middle"), (1, "end")))
        names = "".join(f'<text class="lane-name" x="-10" y="{y}" text-anchor="end">{name}</text>'
                        for name, y in (("Busy", 34), ("Stage", 80), ("Comm", 98), ("Memory", 113)))
        svg = (f'<svg class="timeline" viewBox="-72 -4 1078 142" role="img" aria-label="{_text(window["name"])} timeline">'
               f'<line class="grid" x1="0" x2="1000" y1="{plot}" y2="{plot}"/><line class="grid" x1="0" x2="1000" y1="0" y2="0"/>'
               f'{areas}{"".join(lanes)}{names}{ticks}<line class="crosshair" x1="0" x2="0" y1="-4" y2="118"/></svg>')
        blocks.append(f'<details class="window"{" open" if index == 0 else ""} data-start="{window["start_s"]}" '
                      f'data-seconds="{window["seconds"]}"><summary>{_text(window["name"].capitalize())}'
                      f'<span>{_seconds(window["seconds"])} starting at {_seconds(window["start_s"])}</span></summary>'
                      f'{svg}<p class="readout" aria-live="polite"></p></details>')
    legend = _legend([("Matrix kernels busy", "use"), ("Other kernels busy", "loss"), ("Forward and backward", "use"),
                      ("Optimizer", "use-2"), ("Communication", "loss"), ("Memory copies", "loss-3")])
    return _section("timeline", "Timeline",
                    "Rank 0's traced update, its middle micro-batch, and the tail from the last backward through the optimizer. "
                    "The upper band is how busy each slice is with matrix kernels (blue) and other kernels (orange); point at it "
                    "for the time.", "".join(blocks) + legend)


def _activity(trace):
    rows = (trace or {}).get("per_gpu") or []
    if not rows:
        return ""
    lines = []
    for row in rows:
        parts = [("compute", "Compute only", row["compute_s"] - row["overlap_s"], "use"),
                 ("overlap", "Compute with communication", row["overlap_s"], "use-2"),
                 ("communication", "Exposed communication", row["exposed_communication_s"], "loss"),
                 ("memory", "Exposed memory copies", row.get("exposed_memory_s", row["memory_s"]), "loss-3"),
                 ("idle", "Idle", row["idle_s"], "idle")]
        lines.append(f'<div class="gpu-row"><span class="grid-label">GPU {row["gpu"]}</span>{_segments(parts, size="sm", label_min=0.08)}</div>')
    table = [[f'GPU {row["gpu"]}', _seconds(row["window_s"]), _seconds(row["compute_s"]), _seconds(row["overlap_s"]),
              _seconds(row["exposed_communication_s"]), _seconds(row["idle_s"])] for row in rows]
    legend = _legend([("Compute only", "use"), ("Compute with communication", "use-2"), ("Exposed communication", "loss"),
                      ("Exposed memory copies", "loss-3"), ("Idle", "idle")])
    return _section("gpu-activity", "What the GPUs were doing",
                    "Each traced GPU's update by what its kernels did. Exposed communication has no compute kernel alongside.",
                    "".join(lines) + legend + _table_view(["GPU", "Window", "Compute", "Overlap", "Exposed communication", "Idle"],
                                                          table, numeric=(1, 2, 3, 4, 5)))


def _communication(trace):
    rows = (trace or {}).get("communication") or []
    if not rows:
        return ""
    top = max(row["seconds"] for row in rows) or 1.0
    bars = "".join(_split_bar_row(f'{row["collective"]} {row["dtype"]} in {_stage(row["stage"])}',
                                  [(row["exposed_s"], "loss", "exposed"), (max(row["seconds"] - row["exposed_s"], 0), "use-2", "hidden")],
                                  top, _seconds(row["seconds"]), _amount(row["busbw_gbps"], 1, "GB/s bus")) for row in rows)
    table = [[_text(row["collective"]), _text(row["dtype"]), _text(_stage(row["stage"])), f'{row["calls"]:,.0f}',
              _amount(row["bytes"], 1e9, "GB", 2), _seconds(row["seconds"]), _seconds(row["exposed_s"]),
              _amount(row["algbw_gbps"], 1, "GB/s"), _amount(row["busbw_gbps"], 1, "GB/s"),
              f'<span class="mono">{_text(row["kernel"])}</span>'] for row in rows]
    return _section("communication", "Communication",
                    "Collectives per traced update and rank. Exposed time (orange) has no compute alongside it; hidden time "
                    "(light blue) overlaps compute. Bus bandwidth follows nccl-tests.",
                    f'<div class="bars">{bars}</div>' + _legend([("Exposed", "loss"), ("Hidden behind compute", "use-2")])
                    + _table_view(["Collective", "Type", "Stage", "Calls", "Bytes", "Time", "Exposed", "Algorithm bandwidth",
                                   "Bus bandwidth", "Kernel"], table, numeric=(3, 4, 5, 6, 7, 8)))


def _scheduling(trace):
    launch, idle, syncs = (trace or {}).get("launch"), (trace or {}).get("idle"), (trace or {}).get("syncs") or []
    if not launch:
        return ""
    per_micro = launch.get("per_micro_batch")
    tiles = [("Kernels per update", f'{launch["kernels"]:,.0f}', f'{per_micro:,.0f} kernels per micro-batch' if per_micro else ""),
             ("Host launch time", _seconds(launch["host_s"]), f'{launch["api_calls"]:,.0f} launch calls'),
             ("Memory operations", f'{launch["memory_ops"]:,.0f}', "copies and sets")]
    if idle:
        tiles.append(("GPU idle", _seconds(idle["total_s"]), f'{idle["gaps"]:,.0f} gaps between kernels'))
    body = _tiles(tiles, small=True)
    if idle and idle["total_s"]:
        parts = [("host", "Host late", idle["host_s"], "loss"), ("launch", "Launch latency", idle["launch_s"], "loss-2"),
                 ("dependency", "Waiting on a dependency", idle["dependency_s"], "idle")]
        largest = "".join(f'<li>{_seconds(row["seconds"])} at {row["start_s"]:.3f} s, {_text(row["cause"])}, before '
                          f'<span class="mono">{_text(row["next"])}</span> in {_text(_stage(row["stage"]))}</li>'
                          for row in idle.get("largest", [])[:6])
        body += (f'<h3>Why GPUs idle</h3>{_segments(parts, size="sm", label_min=0.15)}'
                 f'{_legend([(label, tone) for _, label, _, tone in parts])}'
                 f'<details class="table-view"><summary>Largest gaps</summary><ul class="plain">{largest}</ul></details>')
    if syncs:
        rows = [[f'<span class="mono">{_text(row["api"])}</span>', f'<span class="mono">{_text(row["source"])}</span>',
                 _text(_stage(row["stage"])), f'{row["calls"]:,.0f}', _seconds(row["seconds"])] for row in syncs[:10]]
        body += ('<h3>Host synchronizations</h3><p class="lede">Calls that block the host until the GPU catches up; while it '
                 'waits it cannot queue the next kernels.</p><div class="table-scroll">'
                 + _table(["API", "From", "Stage", "Calls", "Host blocked"], rows, numeric=(3, 4)) + "</div>")
    return _section("scheduling", "Scheduling and synchronization", "Per traced update and rank.", body)


def _components(trace):
    rows = (trace or {}).get("components") or []
    if not rows:
        return ""
    top = max(row["forward_s"] + row["backward_s"] for row in rows) or 1.0
    bars = "".join(_split_bar_row(row["name"], [(row["forward_s"], "use", "forward"), (row["backward_s"], "use-2", "backward")],
                                  top, _seconds(row["forward_s"] + row["backward_s"]),
                                  _amount(row["flops"], 1e12, "TFLOP matrix", empty="")) for row in rows)
    layered = [row for row in rows if len(row["layers"]) > 1]
    heat = ""
    if layered:
        layers = sorted({layer["layer"] for row in layered for layer in row["layers"]})
        high = max(layer["forward_s"] + layer["backward_s"] for row in layered for layer in row["layers"]) or 1.0
        lines = []
        for row in layered:
            by_layer = {layer["layer"]: layer for layer in row["layers"]}
            cells = "".join(
                f'<span class="cell" style="--v:{(by_layer[layer]["forward_s"] + by_layer[layer]["backward_s"]) / high:.3f}" '
                f'title="{_tip(_seconds(by_layer[layer]["forward_s"] + by_layer[layer]["backward_s"]), row["name"], f"layer {layer}")}"></span>'
                if layer in by_layer else '<span class="cell empty"></span>' for layer in layers)
            lines.append(f'<div class="grid-row"><span class="grid-label">{_text(row["name"])}</span>'
                         f'<span class="grid-cells" style="--n:{len(layers)}">{cells}</span><span></span></div>')
        head = "".join(f"<span>{layer if layer % 4 == 0 else ''}</span>" for layer in layers)
        heat = (f'<h3>By layer</h3><p class="lede">Forward plus backward GPU time of each layer; darker is slower, an empty '
                f'cell means the component is not in that layer.</p><div class="rank-grid layers"><div class="grid-row head">'
                f'<span class="grid-label">Layer</span><span class="grid-cells" style="--n:{len(layers)}">{head}</span><span></span></div>'
                f'{"".join(lines)}</div>')
    table = [[_text(row["name"]), _seconds(row["forward_s"]), _seconds(row["backward_s"]),
              _amount(row["flops"], 1e12, "TFLOP")] for row in rows]
    layer_rows = [[_text(row["name"]), f'layer {layer["layer"]}', _seconds(layer["forward_s"]), _seconds(layer["backward_s"])]
                  for row in rows for layer in row["layers"]]
    return _section("components", "Model components",
                    "GPU compute time per traced update in each part of the model: forward (blue) and backward (light blue). "
                    "Backward kernels belong to the module whose forward created them.",
                    f'<div class="bars">{bars}</div>' + _legend([("Forward", "use"), ("Backward", "use-2")]) + heat
                    + _table_view(["Component", "Forward", "Backward", "Matrix FLOPs"], table, numeric=(1, 2, 3))
                    + (_table_view(["Component", "Layer", "Forward", "Backward"], layer_rows, numeric=(2, 3), title="Per layer")
                       if layer_rows else ""))


def _tree_rows(nodes, total, depth):
    items = []
    for node in nodes:
        label = _stage(node["name"]) if depth == 0 else node["name"]
        share = node["seconds"] / total if total else 0.0
        line = (f'<span class="name" title="{_text(label)}">{_text(label)}</span>'
                f'<span class="bar-track"><span class="bar tone-use" style="width:{100 * share:.3f}%"></span></span>'
                f'<span class="num">{_seconds(node["seconds"])}</span><span class="num">{_percent(share)}</span>'
                f'<span class="num muted">{node["count"]:,.0f}</span>')
        if node["children"]:
            items.append(f'<details data-depth="{depth}"><summary>{line}</summary>{_tree_rows(node["children"], total, depth + 1)}</details>')
        else:
            items.append(f'<div class="leaf" data-depth="{depth}">{line}</div>')
    return "".join(items)


def _category_tone(index, label):
    if label in ("gemm", "attention"):
        return "use"
    return "loss-3" if label in ("communication", "memory") else "loss"


def _kernels(summary):
    trace = summary.get("trace_summary")
    if not trace:
        return _section("kernel-tree", "Operators and kernels",
                        "No trace in this profile. Rerun with <code>--trace torch</code> for a stage, category, operator and "
                        "kernel breakdown of rank 0, or <code>--trace nsys</code> for the kernels of every rank.", "")
    total = sum(node["seconds"] for node in trace["tree"])
    categories = sorted(((name, seconds) for name, seconds in (trace.get("categories") or {}).items() if seconds > 0),
                        key=lambda item: item[1], reverse=True)
    category_bars = _bars(categories, _seconds, tone_of=_category_tone, bar_label="GPU time by kernel category") if categories else ""
    gemm = trace.get("gemm") or []
    rated = [row for row in gemm if row.get("tflops")][:8]
    shape_bars = _bars([(f'{_stage(row["stage"])}: {row["shape"]}', row["tflops"]) for row in rated],
                       lambda value: f"{value:,.0f} TFLOP/s", bar_label="Rate of the slowest-costing matrix shapes",
                       note_of=lambda index, label: _seconds(rated[index]["seconds"])) if rated else ""
    small = trace.get("small_kernels", {})
    source = ("PyTorch profiler; operators are the CPU ops that launched each kernel" if trace["source"] == "torch"
              else "Nsight Systems, every rank")
    tree = (f'<h3>Drill down</h3><div class="tree"><div class="tree-head"><span>Stage, category, operator, kernel</span>'
            f'<span></span><span class="num">Time</span><span class="num">Share</span><span class="num">Launches</span></div>'
            f'{_tree_rows(trace["tree"], total, 0)}</div>')
    top_rows = [[f'<span class="mono">{_text(row["name"])}</span>', _text(row["category"]), f'{row["count"]:,.0f}',
                 _seconds(row["seconds"]), f'{row["mean_us"]:,.1f} µs'] for row in trace.get("top_kernels") or []]
    gemm_rows = [[("GEMM" if row["kind"] == "gemm" else "Attention"), _text(_stage(row["stage"])),
                  f'<span class="mono">{_text(row["op"])}</span>', f'<span class="mono">{_text(row["shape"])}</span>',
                  f'{row["calls"]:,.0f}', _seconds(row["seconds"]), _amount(row["tflops"], 1, ""),
                  f'<span class="mono">{_text(row["kernel"] or "")}</span>'] for row in gemm[:30]]
    other_rows = [[_text(_stage(row["stage"])), f'<span class="mono">{_text(row["op"])}</span>', _text(row["category"]),
                   f'{row["calls"]:,.0f}', _seconds(row["seconds"])] for row in trace.get("other_operators") or []]
    body = (f'<div class="split"><div><h3>GPU time by kernel category</h3>{category_bars}</div>'
            + (f'<div><h3>Matrix shapes by rate</h3>{shape_bars}</div>' if shape_bars else "") + f"</div>{tree}"
            + (_table_view(["Kernel", "Category", "Launches", "Time", "Mean"], top_rows, numeric=(2, 3, 4), title="Top kernels",
                           identifier="top-kernels") if top_rows else "")
            + (_table_view(["Kind", "Stage", "Operator", "Shape", "Calls", "Time", "TFLOP/s", "Kernel"], gemm_rows,
                           numeric=(4, 5, 6), title="Matrix shapes", identifier="matrix-shapes") if gemm_rows else "")
            + (_table_view(["Stage", "Operator", "Kernels", "Launches", "Time"], other_rows, numeric=(3, 4),
                           title="Non-matrix operators", identifier="other-operators") if other_rows else ""))
    lede = (f'GPU time per traced update and rank ({source}; {trace["steps"]} update(s), {trace["ranks"]} rank(s)). '
            f'{_percent(small.get("count_share", 0), 0)} of kernels run under 10 µs. Each kernel belongs to the stage that '
            f'launched it, so an asynchronous collective appears where it was issued.')
    return _section("kernel-tree", "Operators and kernels", lede, body)


def _memory(summary):
    memory = summary.get("memory_summary")
    if not memory:
        return ""
    total = memory.get("total_bytes") or 1
    bars = "".join(_split_bar_row(f'Rank {row["rank"]}', [(row["allocated_peak"], "use", "allocated"),
                                                          (max(row["reserved_peak"] - row["allocated_peak"], 0), "use-2", "cache")],
                                  total, _gib(row["reserved_peak"]), _percent(row["reserved_peak"] / total, 0))
                   for row in memory["per_rank"])
    table = [[f'Rank {row["rank"]}', _gib(row["allocated_peak"]), _gib(row["reserved_peak"])] for row in memory["per_rank"]]
    lede = (f'Allocator peaks over steady updates against {_gib(memory.get("total_bytes"))} of device memory, which is the '
            f'full width of each bar. Headroom at the reserved peak: {_gib(memory.get("headroom_bytes"))}.')
    return _section("memory", "Memory", lede, f'<div class="bars">{bars}</div>'
                    + _legend([("Allocated by tensors", "use"), ("Allocator cache", "use-2")])
                    + _table_view(["Rank", "Allocated", "Reserved"], table, numeric=(1, 2)))


def _wall(summary):
    rows = summary.get("wall_accounting") or []
    if not rows:
        return ""
    parts = [(row["phase"], PHASE_LABELS.get(row["phase"], row["phase"]), row["seconds"], PHASE_TONES.get(row["phase"], "idle"))
             for row in rows]
    total = sum(row["seconds"] for row in rows)
    return _section("wall-time", "Wall time of this profile",
                    f'Rank 0 from trainer start to exit: {_seconds(total)}. Only steady updates measure throughput.',
                    _segments(parts) + _legend([(label, tone) for _, label, _, tone in parts])
                    + _table_view(["Phase", "Time"], [[_text(label), _seconds(seconds)] for _, label, seconds, _ in parts], numeric=(1,)))


# Telemetry heatmap ------------------------------------------------------------------------------------------------

def _lagging(rows):
    return set(lagging_gpus({int(row.get("index", 0)): float(row["gpu_util_mean_pct"])
                             for row in rows if isinstance(row.get("gpu_util_mean_pct"), (int, float))}))


def _fallback_samples(per_gpu):
    """One honest time bucket for summaries produced before per-sample telemetry."""
    return [dict(time_offset_sec=0.0, index=row.get("index", 0), uuid=row.get("uuid", ""), gpu_util=row.get("gpu_util_mean_pct"),
                 memory_used_pct=row.get("memory_used_peak_pct"), power=row.get("power_mean_w")) for row in per_gpu]


def _heatmap_payload(summary):
    telemetry = summary.get("telemetry_summary", {})
    per_gpu = telemetry.get("per_gpu", [])
    samples = telemetry.get("samples") or _fallback_samples(per_gpu)
    valid = [sample for sample in samples if isinstance(sample.get("time_offset_sec"), (int, float))]
    offsets = sorted({round(float(sample["time_offset_sec"]), 3) for sample in valid}) or [0.0]
    if len(offsets) > 80:
        start, end = offsets[0], offsets[-1]
        width = (end - start) / 80 or 1.0
        offsets = [round(start + (index + 0.5) * width, 3) for index in range(80)]

    def bucket_for(offset):
        return min(range(len(offsets)), key=lambda index: abs(offsets[index] - offset))

    values, grouped = defaultdict(dict), defaultdict(list)
    for sample in valid:
        grouped[(int(sample.get("index", 0)), bucket_for(round(float(sample["time_offset_sec"]), 3)))].append(sample)
    for key, rows in grouped.items():
        for metric in ("gpu_util", "memory_used_pct", "power"):
            numbers = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
            if numbers:
                values[key][metric] = sum(numbers) / len(numbers)
        if "gpu_util" in values[key]:
            values[key]["wait_idle_ratio"] = 100.0 - values[key]["gpu_util"]
    max_gpu = max(max((int(row.get("index", 0)) for row in per_gpu), default=-1),
                  *(int(sample.get("index", 0)) for sample in valid), 7)
    low_gpu = _lagging(per_gpu)
    rows = [{"gpu": gpu, "low": gpu in low_gpu, "cells": [{"bucket": bucket, **values.get((gpu, bucket), {})}
                                                         for bucket in range(len(offsets))]} for gpu in range(max_gpu + 1)]
    return {"offsets": offsets, "rows": rows, "low_gpu": sorted(low_gpu)}


def _phase_row(summary, offsets):
    bands = summary.get("telemetry_summary", {}).get("phase_bands") or []
    if not bands:
        return ""
    cells = []
    for offset in offsets:
        phase = next((band["phase"] for band in bands if band["start"] <= offset <= band["end"]), None)
        label = PHASE_LABELS.get(phase, "—") if phase else "—"
        cells.append(f'<td class="phase-cell tone-{PHASE_TONES.get(phase, "none")}" data-phase="{_text(phase or "none")}" '
                     f'title="{_text(label)}"></td>')
    return f'<tr class="phase-row"><th scope="row">Phase</th>{"".join(cells)}</tr>'


def _heatmap(summary):
    payload = _heatmap_payload(summary)
    offsets = payload["offsets"]
    columns = "".join(f'<th scope="col" data-time-header="{index}">{offset:,.0f}</th>' if index % 8 == 0 else
                      f'<th scope="col" data-time-header="{index}"></th>' for index, offset in enumerate(offsets))
    body = []
    for row in payload["rows"]:
        marker = ' <span class="low-flag" title="Low utilization">Low</span>' if row["low"] else ""
        cells = []
        for cell in row["cells"]:
            attributes = [f'data-gpu-row="{row["gpu"]}"', f'data-time-bucket="{cell["bucket"]}"']
            for metric in ("gpu_util", "memory_used_pct", "power", "wait_idle_ratio"):
                if cell.get(metric) is not None:
                    attributes.append(f'data-{metric}="{cell[metric]:.6f}"')
            value = cell.get("gpu_util")
            fill = "" if value is None else f' style="--cell-value:{max(0.0, min(100.0, value)):.3f}"'
            title = "unavailable" if value is None else f"{value:.1f}%"
            cells.append(f'<td class="heat-cell" {" ".join(attributes)}{fill} title="{title}"></td>')
        body.append(f'<tr><th scope="row">GPU {row["gpu"]}{marker}</th>{"".join(cells)}</tr>')
    telemetry = summary.get("telemetry_summary", {}).get("per_gpu", [])
    lagging = _lagging(telemetry)
    table = [[f'GPU {row.get("index", "?")}' + (" (low utilization)" if int(row.get("index", 0)) in lagging else ""),
              f'{row.get("gpu_util_mean_pct", 0):,.1f}%', f'{row.get("memory_used_peak_mib", 0):,.0f} MiB',
              f'{row.get("power_mean_w", 0):,.1f} W', f'{row.get("sm_clock_mean_mhz", 0):,.0f} MHz'] for row in telemetry]
    scope = ("steady updates" if summary.get("telemetry_summary", {}).get("per_gpu_scope") == "steady" else "the whole profile")
    body_html = (
        f'<div class="heatmap-tools"><label class="metric-switch">Show <select id="heatmap-metric">'
        f'<option value="gpu_util">GPU utilization</option><option value="memory_used_pct">Memory used %</option>'
        f'<option value="power">Power</option><option value="wait_idle_ratio">Idle ratio</option></select></label>'
        f'<div class="heatmap-legend" aria-label="Heatmap color scale"><span>Low</span><i></i><span>High</span></div></div>'
        f'<div class="heatmap-scroll"><table class="heatmap-grid"><thead>{_phase_row(summary, offsets)}'
        f'<tr><th scope="col">Seconds</th>{columns}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
        + (_legend([(label, PHASE_TONES[phase]) for phase, label in PHASE_LABELS.items() if phase != "core"])
           if summary.get("telemetry_summary", {}).get("phase_bands") else "")
        + _table_view(["GPU", "Mean utilization", "Peak memory", "Mean power", "Mean SM clock"], table, numeric=(1, 2, 3, 4),
                      title=f"Per GPU, over {scope}"))
    return _section("gpu-heatmap", "GPU telemetry over time",
                    "Sampled by nvidia-smi on every GPU for the whole profile; the strip on top marks the phase.",
                    body_html, extra_class="heatmap-panel")


# Page -------------------------------------------------------------------------------------------------------------

LIGHT = ("--page:#eef1f5;--surface:#ffffff;--raise:#f5f7fa;--ink:#0f1722;--ink-2:#465263;--ink-3:#6f7b8b;--line:#e2e7ee;"
         "--use:#2a78d6;--use-2:#86b6ef;--loss:#eb6834;--loss-2:#f19a72;--loss-3:#b8461a;--idle:#b5bdc8;--warn:#fab219;"
         "--track:#e9eef5;--heat-0:#e6eef9;--heat-1:#184f95;--on-strong:#ffffff;--on-soft:#0f1722")
DARK = ("color-scheme:dark;--page:#0c1218;--surface:#141b24;--raise:#1a2330;--ink:#edf1f6;--ink-2:#b2bdca;--ink-3:#7e8a99;"
        "--line:#243040;--use:#3987e5;--use-2:#184f95;--loss:#d95926;--loss-2:#7a3a1f;--loss-3:#f19a72;--idle:#4a5667;"
        "--warn:#fab219;--track:#1d2733;--heat-0:#1a2431;--heat-1:#5598e7;--on-strong:#ffffff;--on-soft:#edf1f6")

STYLE = (":root{color-scheme:light;" + LIGHT + ";--font:'Avenir Next','Segoe UI Variable Text','Segoe UI',system-ui,"
         "-apple-system,sans-serif;--mono:'SF Mono','Cascadia Code',ui-monospace,Menlo,monospace}"
         "@media (prefers-color-scheme:dark){:root:where(:not([data-theme='light'])){" + DARK + "}}"
         ":root[data-theme='dark']{" + DARK + "}" + """
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--font);-webkit-font-smoothing:antialiased}
a{color:var(--use)} code,.mono{font-family:var(--mono);font-size:.8em} .mono{word-break:break-all}
.shell{max-width:1160px;margin:0 auto;padding:24px 28px 64px}
.topbar{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;align-items:center;gap:16px;
  margin:0 -28px 26px;padding:12px 28px;background:color-mix(in oklab,var(--page) 88%,transparent);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line);font-size:.86rem}
.topbar nav{display:flex;gap:18px;flex-wrap:wrap} .topbar a{color:var(--ink-2);text-decoration:none} .topbar a:hover{color:var(--ink)}
.topbar a.back{color:var(--ink);font-weight:600}
.theme{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:999px;padding:5px 14px;font:inherit;cursor:pointer}
:focus-visible{outline:2px solid var(--use);outline-offset:2px}
.masthead{margin:8px 0 22px} .masthead h1{font-size:2.3rem;line-height:1.1;font-weight:650;letter-spacing:-.02em;margin:0 0 10px}
.masthead .headline{font-size:1.15rem;color:var(--ink-2);margin:0 0 16px;max-width:68ch}
.facts{display:flex;flex-wrap:wrap;gap:4px 26px;margin:0;font-size:.85rem} .facts div{display:flex;gap:8px}
.facts dt{color:var(--ink-3)} .facts dd{margin:0;color:var(--ink-2)}
.overview,.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px}
.overview{padding:22px 24px 14px;margin-bottom:18px} .panel{padding:22px 24px 18px;margin:0 0 18px}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:20px}
.tile{padding:0 18px;border-left:1px solid var(--line);display:flex;flex-direction:column;gap:1px;min-width:0}
.tile:first-child{border-left:0;padding-left:0} .tile-label{font-size:.8rem;color:var(--ink-3)}
.tile-value{font-size:1.8rem;font-weight:650;letter-spacing:-.015em;white-space:nowrap} .tile-note{font-size:.78rem;color:var(--ink-3)}
.tiles.small{margin-bottom:6px} .tiles.small .tile-value{font-size:1.3rem}
.hero-bar{margin:0} .hero-bar figcaption{font-size:.86rem;color:var(--ink-2);margin-bottom:8px}
.stack{display:flex;gap:2px;width:100%;border-radius:6px;overflow:hidden} .stack-lg{height:44px} .stack-md{height:26px} .stack-sm{height:18px}
.seg{min-width:2px;display:flex;align-items:center;padding-left:8px;overflow:hidden;cursor:default}
.seg:hover{filter:brightness(1.1)} .seg-label{font-size:.8rem;font-weight:600;color:var(--on-strong);white-space:nowrap}
.stack-sm .seg-label{font-size:.7rem}
.seg.tone-use-2 .seg-label,.seg.tone-loss-2 .seg-label,.seg.tone-idle .seg-label{color:var(--on-soft)}
.tone-use{background:var(--use);fill:var(--use)} .tone-use-2{background:var(--use-2);fill:var(--use-2)}
.tone-loss{background:var(--loss);fill:var(--loss)} .tone-loss-2{background:var(--loss-2);fill:var(--loss-2)}
.tone-loss-3{background:var(--loss-3);fill:var(--loss-3)} .tone-idle{background:var(--idle);fill:var(--idle)} .tone-none{background:transparent}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;list-style:none;padding:0;margin:10px 0 2px;font-size:.8rem;color:var(--ink-2)}
.legend li{display:flex;align-items:center;gap:7px} .key{display:inline-block;width:11px;height:11px;border-radius:3px;flex:none}
.panel header{margin-bottom:14px} .panel h2{font-size:1.14rem;font-weight:650;margin:0 0 4px;letter-spacing:-.005em}
.panel h3{font-size:.9rem;font-weight:600;color:var(--ink-2);margin:20px 0 10px} .lede{color:var(--ink-2);font-size:.88rem;margin:0;max-width:82ch}
.table-scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.84rem;margin-top:10px} th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--ink-3);font-weight:500} .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.overview td:first-child{white-space:nowrap} .overview td:first-child .key{margin-right:8px;vertical-align:-1px}
.table-view{margin-top:12px;font-size:.84rem} .table-view>summary{color:var(--ink-2)}
details>summary{list-style:none;cursor:pointer} details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"";display:inline-block;width:6px;height:6px;border-right:1.5px solid var(--ink-3);
  border-bottom:1.5px solid var(--ink-3);transform:rotate(-45deg);margin:0 10px 2px 2px;transition:transform .15s}
details[open]>summary::before{transform:rotate(45deg)}
.findings{list-style:none;padding:0;margin:0;display:grid;gap:8px}
.finding{display:grid;grid-template-columns:74px 1fr;gap:12px;align-items:start;padding:12px 14px;border-radius:10px;background:var(--raise)}
.finding .badge{justify-self:start;font-size:.72rem;font-weight:600;padding:2px 9px;border-radius:999px;background:var(--track);color:var(--ink-2)}
.finding.warning .badge{background:var(--warn);color:#111a24} .finding b{font-weight:600} .finding p{margin:2px 0 0;color:var(--ink-2);font-size:.86rem}
.alert{margin-bottom:18px}
.meters{display:flex;gap:40px;flex-wrap:wrap;margin:4px 0 4px} .meter{margin:0;display:flex;align-items:center;gap:16px}
.meter svg{width:168px;height:92px;flex:none} .meter-track{fill:none;stroke:var(--track);stroke-width:15;stroke-linecap:round}
.meter-fill{fill:none;stroke:var(--use);stroke-width:15;stroke-linecap:round} .meter-value{font-size:30px;font-weight:650;fill:var(--ink);text-anchor:middle}
.meter figcaption{display:flex;flex-direction:column;font-size:.84rem;color:var(--ink-2);max-width:26ch} .meter figcaption b{color:var(--ink);font-size:.95rem}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:32px}
.bars{display:grid;gap:7px}
.bar-row{display:grid;grid-template-columns:minmax(120px,1fr) minmax(0,2fr) 76px minmax(0,1fr);align-items:center;gap:12px;font-size:.84rem;border-radius:4px}
.split .bar-row{grid-template-columns:minmax(100px,1.1fr) minmax(0,1.6fr) 64px} .split .bar-note{display:none}
.bar-label{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap} .bar-value{text-align:right;font-variant-numeric:tabular-nums}
.bar-note{color:var(--ink-3);font-size:.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{display:flex;gap:2px;height:14px;background:var(--track);border-radius:0 4px 4px 0;overflow:hidden}
.bar{display:block;height:100%;min-width:0} .bar:last-child{border-radius:0 4px 4px 0}
.bar-row:hover .bar-label,.bar-row:focus-visible .bar-label{color:var(--ink)}
.rank-grid{display:grid;gap:4px;font-size:.8rem} .grid-row{display:grid;grid-template-columns:150px minmax(0,1fr) 70px;gap:12px;align-items:center}
.grid-label{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap} .grid-note{color:var(--ink-3);font-variant-numeric:tabular-nums;text-align:right}
.grid-cells{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:2px} .grid-row.head .grid-cells span{text-align:center;color:var(--ink-3);font-size:.72rem}
.cell{height:16px;border-radius:3px;background:color-mix(in oklab,var(--heat-1) calc(var(--v) * 100%),var(--heat-0))} .cell.empty{background:transparent;outline:1px solid var(--line);outline-offset:-1px}
.layers .grid-row{grid-template-columns:190px minmax(0,1fr) 0}
.gpu-row{display:grid;grid-template-columns:64px minmax(0,1fr);gap:12px;align-items:center;margin-bottom:6px;font-size:.84rem}
.timeline{display:block;width:100%;height:auto} .timeline .grid{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke}
.timeline .area.tone-use{opacity:.9} .timeline .area.tone-loss{opacity:.8}
.timeline text{font-size:12px;fill:var(--ink-3)} .crosshair{stroke:var(--ink);stroke-width:1;opacity:0;vector-effect:non-scaling-stroke}
.window{margin:4px 0 14px} .window>summary{font-weight:600;margin-bottom:8px;font-size:.92rem}
.window>summary span{font-weight:400;color:var(--ink-3);margin-left:10px;font-size:.84rem}
.readout{font-size:.8rem;color:var(--ink-2);min-height:1.3em;margin:0 0 0 6.6%;font-variant-numeric:tabular-nums}
.tree{font-size:.83rem;overflow-x:auto}
.tree-head,.tree summary,.tree .leaf{display:grid;grid-template-columns:minmax(180px,2.4fr) minmax(60px,1fr) 76px 56px 78px;gap:10px;align-items:center;padding:5px 0;border-bottom:1px solid var(--line)}
.tree-head{color:var(--ink-3);font-size:.76rem;padding-left:18px} .tree .leaf{padding-left:18px}
.tree details>details,.tree details>.leaf{margin-left:18px}
.tree .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tree [data-depth="2"] .name,.tree [data-depth="3"] .name{font-family:var(--mono);font-size:.76rem}
.tree [data-depth="0"]>summary .name{font-weight:600} .tree .bar-track{height:8px} .muted{color:var(--ink-3)}
.big-number{font-size:2.5rem;font-weight:650;margin:0 0 12px;letter-spacing:-.02em} .big-number span{font-size:.95rem;font-weight:400;color:var(--ink-2);margin-left:12px;letter-spacing:0}
.heatmap-tools{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:10px;font-size:.84rem;color:var(--ink-2)}
.metric-switch select{margin-left:6px;padding:4px 8px;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font:inherit}
.heatmap-legend{display:flex;align-items:center;gap:8px} .heatmap-legend i{width:120px;height:10px;border-radius:5px;background:linear-gradient(90deg,var(--heat-0),var(--heat-1))}
.heatmap-scroll{overflow-x:auto} .heatmap-grid{border-collapse:separate;border-spacing:2px;min-width:max-content;margin:0}
.heatmap-grid th,.heatmap-grid td{border:0;padding:0} .heatmap-grid thead th{font-size:.7rem;color:var(--ink-3);font-weight:400;text-align:left;height:16px}
.heatmap-grid tbody th{font-size:.78rem;color:var(--ink-2);font-weight:400;padding-right:10px;white-space:nowrap}
.heat-cell{width:11px;height:20px;border-radius:2px;background:color-mix(in oklab,var(--heat-1) calc(var(--cell-value,0) * 1%),var(--heat-0))}
.heat-cell:not([style]){background:transparent;outline:1px solid var(--line);outline-offset:-1px} .phase-cell{height:6px;border-radius:2px}
.low-flag{font-size:.68rem;padding:0 6px;border-radius:999px;background:var(--warn);color:#111a24;margin-left:4px}
.plain{padding-left:18px;margin:8px 0;color:var(--ink-2)}
.compare{display:grid;gap:12px} .compare-row{display:grid;grid-template-columns:180px minmax(0,1fr) 70px;gap:14px;align-items:center;font-size:.86rem}
.compare-row a{font-weight:600;text-decoration:none;color:var(--ink)} .compare-row a:hover{color:var(--use)}
.compare-row .num{color:var(--ink-2)} .index-table td:first-child a{font-weight:600;text-decoration:none}
.footer{color:var(--ink-3);font-size:.78rem;margin-top:28px}
@media (max-width:860px){.shell{padding:18px 16px 48px}.topbar{margin:0 -16px 20px;padding:10px 16px}.topbar nav{display:none}
  .tiles{grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 0}.tile:nth-child(3){border-left:0;padding-left:0}
  .split{grid-template-columns:minmax(0,1fr)}.bar-row{grid-template-columns:100px minmax(0,1fr) 64px}.bar-note{display:none}
  .grid-row{grid-template-columns:96px minmax(0,1fr) 56px}.compare-row{grid-template-columns:minmax(0,1fr);gap:4px}
  .finding{grid-template-columns:1fr}.masthead h1{font-size:1.8rem}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media print{.topbar{position:static}.theme{display:none}.panel,.overview{break-inside:avoid}}
""")

SCRIPT = """
(() => {
  const root = document.documentElement, button = document.querySelector('.theme');
  let stored = null; try { stored = localStorage.getItem('nt-profile-theme'); } catch (error) {}
  if (stored) root.dataset.theme = stored;
  const dark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const label = () => { if (button) button.textContent = dark() ? 'Light theme' : 'Dark theme'; };
  if (button) button.addEventListener('click', () => { root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem('nt-profile-theme', root.dataset.theme); } catch (error) {} label(); });
  label();
  const select = document.getElementById('heatmap-metric');
  if (select) {
    const cells = [...document.querySelectorAll('#gpu-heatmap .heat-cell')];
    const update = () => {
      const metric = select.value, name = select.selectedOptions[0].textContent;
      const values = cells.map(cell => Number(cell.dataset[metric])).filter(Number.isFinite);
      const low = values.length ? Math.min(...values) : 0, high = values.length ? Math.max(...values) : 1;
      cells.forEach(cell => { const value = Number(cell.dataset[metric]);
        if (!Number.isFinite(value)) { cell.style.removeProperty('--cell-value'); cell.title = 'unavailable'; return; }
        cell.style.setProperty('--cell-value', String(high === low ? 50 : (value - low) / (high - low) * 100));
        cell.title = value.toFixed(1) + ' \\u00b7 ' + name; });
    };
    select.addEventListener('change', update); update();
  }
  const format = seconds => seconds >= 1 ? seconds.toFixed(3) + ' s' : (seconds * 1e3).toFixed(1) + ' ms';
  document.querySelectorAll('.window').forEach(block => {
    const svg = block.querySelector('svg.timeline'), line = svg && svg.querySelector('.crosshair'), out = block.querySelector('.readout');
    if (!svg) return;
    const start = Number(block.dataset.start), seconds = Number(block.dataset.seconds);
    svg.addEventListener('pointermove', event => {
      const box = svg.getBoundingClientRect(), x = (event.clientX - box.left) / box.width * 1078 - 72;
      if (x < 0 || x > 1000) { line.style.opacity = 0; return; }
      line.setAttribute('x1', x); line.setAttribute('x2', x); line.style.opacity = 0.6;
      const near = [...svg.querySelectorAll('rect')].filter(rect => { const left = Number(rect.getAttribute('x'));
        return left <= x && x <= left + Number(rect.getAttribute('width')); });
      out.textContent = format(start + x / 1000 * seconds) + near.map(rect => '  \\u00b7  ' + rect.querySelector('title').textContent).join('');
    });
    svg.addEventListener('pointerleave', () => { line.style.opacity = 0; out.textContent = ''; });
  });
})();
"""

NAV = (("findings", "Findings"), ("mfu", "MFU"), ("stages", "Stages"), ("timeline", "Timeline"),
       ("communication", "Communication"), ("components", "Components"), ("kernel-tree", "Kernels"), ("gpu-heatmap", "Telemetry"))


def _page(title, body, nav="", back=None):
    links = (f'<a class="back" href="{_text(back)}">All profiles</a>' if back else "") + nav
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_text(title)}</title><style>{STYLE}</style></head><body><div class="shell">'
            f'<div class="topbar"><nav>{links}</nav><button class="theme" type="button">Dark theme</button></div><main>{body}</main>'
            f'<p class="footer">Static research report generated from summary.json. Open a PyTorch trace in Perfetto or an Nsight '
            f'report in Nsight Systems to see every kernel.</p></div><script>{SCRIPT}</script></body></html>\n')


def render_report(summary: dict, back: str | None = None) -> str:
    """One profile as a self-contained HTML page."""
    trace = summary.get("trace_summary")
    low = _heatmap_payload(summary)["low_gpu"]
    if any(finding.get("id") == "lagging-gpu" for finding in summary.get("findings") or []):
        low = []
    scope = "steady" if summary.get("telemetry_summary", {}).get("per_gpu_scope") == "steady" else "captured"
    warning = (f'<div class="finding warning alert"><span class="badge">Warning</span><div><b>Low utilization</b><p>GPU '
               f'{", GPU ".join(map(str, low))} averaged under half the median GPU utilization during the {scope} interval.</p></div></div>'
               if low else "")
    nav = "".join(f'<a href="#{identifier}">{label}</a>' for identifier, label in NAV)
    body = (f'<header class="masthead"><h1>{_text(_title(summary))}</h1><p class="headline">{_text(_headline(summary))}</p>'
            f'{_identity(summary)}</header>{warning}{_overview(summary)}{_findings(summary)}{_mfu(summary)}{_projection(summary)}'
            f'{_stages(summary)}{_timeline(trace)}{_activity(trace)}{_communication(trace)}{_scheduling(trace)}'
            f'{_components(trace)}{_kernels(summary)}{_memory(summary)}{_wall(summary)}{_heatmap(summary)}')
    return _page(f"{_title(summary)} profile", body, nav=nav, back=back)


def write_report(summary_path: Path, output_path: Path, back: str | None = None) -> None:
    """Render summary JSON to an HTML file."""
    Path(output_path).write_text(render_report(json.loads(Path(summary_path).read_text()), back=back))


# Index ------------------------------------------------------------------------------------------------------------

def render_index(entries: list[tuple[str, dict]]) -> str:
    """Compare profiles on one page; ``entries`` are ``(link to the profile page, summary)``."""
    rows, bars = [], []
    longest = max(((summary.get("mfu_summary") or {}).get("traced_window_s") or 0 for _, summary in entries), default=0) or 1.0
    for link, summary in entries:
        steady = (summary.get("metrics_summary") or {}).get("steady") or {}
        mfu = summary.get("mfu_summary") or {}
        projection = summary.get("projection") or {}
        shape = summary.get("shape") or {}
        name = _title(summary)
        attention = (summary.get("software") or {}).get("native_attention")
        detail = ", ".join(part for part in (f'mb{shape["micro_batch"]} × {shape["accumulation"]}' if shape else "",
                                             f"{attention} attention" if attention else "") if part)
        rows.append([f'<a href="{_text(link)}">{_text(name)}</a><br><span class="muted">{_text(detail)}</span>',
                     _seconds(steady.get("median_step_seconds")), _compact(steady.get("median_tokens_per_second") or 0),
                     _percent(mfu.get("mfu")),
                     f'{projection["estimated_hours"]:,.0f} h' if projection.get("estimated_hours") else "—"])
        waterfall = mfu.get("waterfall") or []
        if waterfall:
            window = mfu.get("traced_window_s") or sum(max(row["seconds"], 0) for row in waterfall)
            parts = [(row["part"], WATERFALL_PARTS.get(row["part"], row["part"]), max(row["seconds"], 0),
                      WATERFALL_TONES.get(row["part"], "loss")) for row in waterfall]
            bars.append(f'<div class="compare-row"><a href="{_text(link)}">{_text(name)}<br><span class="muted">{_text(detail)}</span></a>'
                        f'<div style="width:{100 * window / longest:.2f}%">{_segments(parts, label_min=0.12)}</div>'
                        f'<span class="num">{_seconds(window)}</span></div>')
    legend = _legend([(WATERFALL_PARTS[part], tone) for part, tone in WATERFALL_TONES.items() if part != "matrix kernels below peak"])
    body = (f'<header class="masthead"><h1>Training profiles</h1><p class="headline">{len(entries)} profiles, each at the update '
            f'shape it ran. The bars share one time scale, so their lengths compare update times.</p></header>'
            + (f'<section class="panel" id="compare"><header><h2>Where each update goes</h2><p class="lede">The traced update on '
               f'rank 0, split into model FLOPs at peak (blue) and the time lost beyond them.</p></header>'
               f'<div class="compare">{"".join(bars)}</div>{legend}</section>' if bars else "")
            + f'<section class="panel" id="profiles"><header><h2>Profiles</h2></header><div class="index-table table-scroll">'
            f'{_table(["Profile", "Update", "Tokens/s", "MFU", "Formal run"], rows, numeric=(1, 2, 3, 4))}</div></section>')
    return _page("Training profiles", body)


def write_site(summaries: list[Path], output: Path) -> list[Path]:
    """Write one page per summary under ``output`` and an index linking them; returns the pages written."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries, pages = [], []
    for summary_path in summaries:
        summary = json.loads(Path(summary_path).read_text())
        slug = Path(summary_path).resolve().parent.name or _title(summary)
        page = output / f"{slug}.html"
        page.write_text(render_report(summary, back="index.html"))
        entries.append((page.name, summary))
        pages.append(page)
    (output / "index.html").write_text(render_index(entries))
    return [output / "index.html", *pages]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Render profile summaries as one page each plus a comparison index.")
    parser.add_argument("summaries", type=Path, nargs="+", help="summary.json files of profile directories")
    parser.add_argument("--output", type=Path, required=True, help="directory for index.html and the profile pages")
    args = parser.parse_args(argv)
    for page in write_site(args.summaries, args.output):
        print(page)


if __name__ == "__main__":
    main()
