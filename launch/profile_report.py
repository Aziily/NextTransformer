"""Render a self-contained human-readable HTML report from profiler JSON."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from launch.profile_analysis import lagging_gpus


def _number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _memory_gib(value: int | float) -> str:
    return f"{value / (1024 ** 3):,.2f} GiB"


def _lagging(rows: list[dict]) -> set[int]:
    return set(lagging_gpus({int(row.get("index", 0)): float(row["gpu_util_mean_pct"])
                             for row in rows if isinstance(row.get("gpu_util_mean_pct"), (int, float))}))


def _telemetry_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">No GPU telemetry samples fell inside the profile interval.</p>'
    body = []
    lagging = _lagging(rows)
    for row in rows:
        warning = ' class="low"' if int(row.get("index", 0)) in lagging else ""
        body.append(
            f"<tr{warning}><td>GPU {row.get('index', '?')}</td>"
            f"<td>{_number(row.get('gpu_util_mean_pct', 0), 1)}%</td>"
            f"<td>{_number(row.get('memory_used_peak_mib', 0), 0)} MiB</td>"
            f"<td>{_number(row.get('power_mean_w', 0), 1)} W</td>"
            f"<td>{_number(row.get('sm_clock_mean_mhz', 0), 0)} MHz</td></tr>"
        )
    return '<table><thead><tr><th>GPU</th><th>Mean utilization</th><th>Peak memory</th><th>Mean power</th><th>Mean SM clock</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table>'


def _fallback_samples(per_gpu: list[dict]) -> list[dict]:
    """Make one honest time bucket for summaries produced before heatmap v2."""
    return [
        {
            "time_offset_sec": 0.0,
            "index": row.get("index", 0),
            "uuid": row.get("uuid", ""),
            "gpu_util": row.get("gpu_util_mean_pct"),
            "memory_used_pct": row.get("memory_used_peak_pct"),
            "power": row.get("power_mean_w"),
        }
        for row in per_gpu
    ]


def _heatmap_payload(summary: dict) -> dict:
    telemetry = summary.get("telemetry_summary", {})
    per_gpu = telemetry.get("per_gpu", [])
    samples = telemetry.get("samples") or _fallback_samples(per_gpu)
    valid = [sample for sample in samples if isinstance(sample.get("time_offset_sec"), (int, float))]
    offsets = sorted({round(float(sample["time_offset_sec"]), 3) for sample in valid}) or [0.0]
    if len(offsets) > 80:
        start, end = offsets[0], offsets[-1]
        width = (end - start) / 80 or 1.0
        offsets = [round(start + (index + 0.5) * width, 3) for index in range(80)]
    bucket_for = lambda offset: min(range(len(offsets)), key=lambda index: abs(offsets[index] - offset))
    values: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for sample in valid:
        gpu = int(sample.get("index", 0))
        grouped[(gpu, bucket_for(round(float(sample["time_offset_sec"]), 3)))].append(sample)
    for key, rows in grouped.items():
        for metric in ("gpu_util", "memory_used_pct", "power"):
            numbers = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
            if numbers:
                values[key][metric] = sum(numbers) / len(numbers)
        if "gpu_util" in values[key]:
            values[key]["wait_idle_ratio"] = 100.0 - values[key]["gpu_util"]
    max_gpu = max((int(row.get("index", 0)) for row in per_gpu), default=-1)
    max_gpu = max(max_gpu, *(int(sample.get("index", 0)) for sample in valid), 7)
    rows = []
    low_gpu = _lagging(per_gpu)
    for gpu in range(max_gpu + 1):
        cells = [{"bucket": bucket, **values.get((gpu, bucket), {})} for bucket in range(len(offsets))]
        rows.append({"gpu": gpu, "low": gpu in low_gpu, "cells": cells})
    return {"offsets": offsets, "rows": rows, "low_gpu": sorted(low_gpu)}


def _heatmap_html(summary: dict) -> str:
    payload = _heatmap_payload(summary)
    offsets = payload["offsets"]
    columns = ''.join(
        f'<th scope="col" data-time-header="{index}">{_number(offset, 1)}s</th>'
        for index, offset in enumerate(offsets)
    )
    body = []
    for row in payload["rows"]:
        marker = '<span class="low-flag" title="Low utilization">⚠</span>' if row["low"] else ''
        cells = []
        for cell in row["cells"]:
            attributes = [
                f'data-gpu-row="{row["gpu"]}"',
                f'data-time-bucket="{cell["bucket"]}"',
            ]
            for metric in ("gpu_util", "memory_used_pct", "power", "wait_idle_ratio"):
                value = cell.get(metric)
                if value is not None:
                    attributes.append(f'data-{metric}="{value:.6f}"')
            value = cell.get("gpu_util")
            fill = "" if value is None else f' style="--cell-value:{max(0.0, min(100.0, value)):.3f}"'
            label = "unavailable" if value is None else f'{value:.1f}%'
            cells.append(f'<td class="heat-cell" {" ".join(attributes)}{fill} title="{label}"></td>')
        body.append(f'<tr><th scope="row">GPU {row["gpu"]} {marker}</th>{"".join(cells)}</tr>')
    data = html.escape(json.dumps(payload, separators=(",", ":")), quote=False)
    return f"""
<section id="gpu-heatmap" class="panel heatmap-panel" aria-labelledby="heatmap-title">
<div class="section-heading"><div><h2 id="heatmap-title">GPU × time heatmap</h2><p class="muted">Telemetry buckets cover the captured profile interval. Missing samples remain unavailable.</p></div>
<label class="metric-switch">Metric <select id="heatmap-metric"><option value="gpu_util">GPU utilization</option><option value="memory_used_pct">Memory used %</option><option value="power">Power</option><option value="wait_idle_ratio">Wait / idle ratio</option></select></label></div>
<div class="heatmap-scroll"><table class="heatmap-grid"><thead>{_phase_row(summary, offsets)}<tr><th scope="col">GPU</th>{columns}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<div class="heatmap-legend" aria-label="Heatmap color scale"><span>Low</span><i></i><i></i><i></i><i></i><i></i><span>High</span><span class="muted">Cells show the selected metric; ⚠ marks low utilization.</span></div>
<script type="application/json" id="heatmap-data">{data}</script>
<script>
(() => {{
  const select = document.getElementById('heatmap-metric');
  const cells = [...document.querySelectorAll('#gpu-heatmap .heat-cell')];
  const update = () => {{
    const metric = select.value;
    const values = cells.map(cell => Number(cell.dataset[metric])).filter(Number.isFinite);
    const low = values.length ? Math.min(...values) : 0;
    const high = values.length ? Math.max(...values) : 1;
    cells.forEach(cell => {{
      const value = Number(cell.dataset[metric]);
      if (!Number.isFinite(value)) {{ cell.style.setProperty('--cell-value', ''); cell.title = 'unavailable'; return; }}
      const normalized = high === low ? 0.5 : (value - low) / (high - low);
      cell.style.setProperty('--cell-value', String(normalized * 100));
      cell.title = `${{value.toFixed(1)}} ${{metric}}`;
    }});
  }};
  select.addEventListener('change', update); update();
}})();
</script>
</section>"""


PHASE_LABELS = {"startup": "Startup", "warmup": "Warmup", "steady": "Steady", "traced": "Traced",
                "validation": "Validation", "core": "CORE", "checkpoint": "Checkpoint"}
STAGE_LABELS = {"step_setup": "Step setup", "data": "Data", "forward": "Forward", "backward": "Backward",
                "grad_norm": "Gradient norm", "wait": "Host wait", "optimizer.reduce": "Optimizer reduce",
                "optimizer.compute": "Optimizer compute", "optimizer.gather": "Optimizer gather",
                "step_tail": "Step tail", "unattributed": "Unattributed"}


def _text(value) -> str:
    return html.escape(str(value))


def _seconds(value: float) -> str:
    if value >= 1:
        return f"{value:,.2f} s"
    if value >= 1e-3:
        return f"{value * 1e3:,.1f} ms"
    return f"{value * 1e6:,.0f} µs"


def _phase_row(summary: dict, offsets: list[float]) -> str:
    """Phase of each heatmap column, from rank 0's stage timeline."""
    bands = summary.get("telemetry_summary", {}).get("phase_bands") or []
    if not bands:
        return ""
    cells = []
    for offset in offsets:
        phase = next((band["phase"] for band in bands if band["start"] <= offset <= band["end"]), None)
        label = PHASE_LABELS.get(phase, "—") if phase else "—"
        cells.append(f'<td class="phase-cell" data-phase="{_text(phase or "none")}" title="{_text(label)}"></td>')
    return f'<tr class="phase-row"><th scope="row">Phase</th>{"".join(cells)}</tr>'


def _findings_html(summary: dict) -> str:
    findings = summary.get("findings") or []
    if not findings:
        return ""
    items = "".join(
        f'<li class="finding {_text(finding["severity"])}" data-finding="{_text(finding["id"])}">'
        f'<b>{_text(finding["title"])}</b><span>{_text(finding["detail"])}</span></li>'
        for finding in findings
    )
    return f'<section id="findings" class="panel"><h2>What to look at first</h2><ul class="findings">{items}</ul></section>'


def _shape_html(summary: dict) -> str:
    shape = summary.get("shape")
    if not shape:
        return ""
    flops = shape.get("model_flops_per_gpu_s")
    rate = f" · {flops / 1e12:,.1f} model TFLOP/s per GPU (audited theory FLOPs)" if flops else ""
    return (f'<p class="muted">{shape["world"]} ranks × micro-batch {shape["micro_batch"]} × '
            f'{shape["accumulation"]} accumulation = {shape["tokens_per_update"]:,} tokens per update{rate}.</p>')


def _projection_html(summary: dict) -> str:
    projection = summary.get("projection")
    if not projection or not projection.get("representative") or "estimated_hours" not in projection:
        return ""
    rows = [("Training updates", f'{projection["updates"]:,}', projection["training_hours"])]
    if "validation_hours" in projection:
        rows.append(("Validation events", f'{projection["validation_events"]}', projection["validation_hours"]))
    if "checkpoint_hours" in projection:
        rows.append(("Rolling checkpoints", f'{projection["checkpoints"]} (every {projection["checkpoint_interval"]})'
                     if "checkpoint_interval" in projection else f'{projection["checkpoints"]}',
                     projection["checkpoint_hours"]))
    body = "".join(f"<tr><td>{label}</td><td>{count}</td><td>{hours:,.1f} h</td></tr>" for label, count, hours in rows)
    total = projection["estimated_hours"]
    missing = "CORE-22 evaluation is not measured here and comes on top."
    if "validation_hours" not in projection and "validation_events" in projection:
        missing = (f'Validation ({projection["validation_events"]} events) and CORE-22 are not included: this profile '
                   f'validated only {projection.get("validation_measured_tokens", 0):,} tokens. Rerun with '
                   f'<code>--validation-tokens</code> set to the formal size to measure validation.')
    return (f'<section id="projection" class="panel"><h2>Formal run estimate</h2>'
            f'<p><span class="value">{total:,.1f} h</span> ≈ {total / 24:,.1f} days on this machine, from the median '
            f'steady update. {missing}</p>'
            f'<table><thead><tr><th>Part</th><th>Count</th><th>Time</th></tr></thead><tbody>{body}</tbody></table></section>')


def _stages_html(summary: dict) -> str:
    stages = summary.get("stage_summary") or {}
    rows = stages.get("stages") or []
    if not rows:
        return ('<section id="stages" class="panel"><h2>Where an update spends its time</h2><p class="muted">Stage '
                'timings are unavailable: this profile predates the stage timer or the trainer exited early.</p></section>')
    timing = "CUDA events on each rank's stream" if stages.get("timing") == "cuda_events" else "host clocks"
    step = stages["step_seconds"]
    bar = "".join(f'<span class="seg" data-stage="{_text(row["stage"])}" style="width:{100 * row["share"]:.3f}%" '
                  f'title="{_text(STAGE_LABELS.get(row["stage"], row["stage"]))}: {_seconds(row["seconds"])}"></span>'
                  for row in rows)
    body = []
    for row in rows:
        spread = row["rank_max"] - row["rank_min"]
        ranks = "".join(f"<li>rank {rank}: {_seconds(value)}</li>" for rank, value in row.get("per_rank", []))
        body.append(
            f'<details class="stage-row" data-stage="{_text(row["stage"])}"><summary>'
            f'<span class="swatch" data-stage="{_text(row["stage"])}"></span>'
            f'<b>{_text(STAGE_LABELS.get(row["stage"], row["stage"]))}</b>'
            f'<span>{_seconds(row["seconds"])}</span><span>{100 * row["share"]:.1f}%</span>'
            f'<span class="muted">host {_seconds(row["host_seconds"])}</span>'
            f'<span class="muted">rank spread {_seconds(spread)} (slowest rank {row["slowest_rank"]})</span>'
            f'</summary><ul class="ranks">{ranks}</ul></details>'
        )
    return (f'<section id="stages" class="panel"><h2>Where an update spends its time</h2>'
            f'<p class="muted">Mean of steady updates {", ".join(map(str, stages["steady_steps"]))} over all ranks, '
            f'timed with {timing}: {_seconds(step)} per update. Each stage runs from the trainer entering it to entering '
            f'the next; host wait is time the host blocks on the GPU. Expand a stage for per-rank times.</p>'
            f'<div class="waterfall">{bar}</div>{"".join(body)}</section>')


def _wall_html(summary: dict) -> str:
    rows = summary.get("wall_accounting") or []
    if not rows:
        return ""
    total = sum(row["seconds"] for row in rows) or 1.0
    bar = "".join(f'<span class="seg" data-phase="{_text(row["phase"])}" style="width:{100 * row["seconds"] / total:.3f}%" '
                  f'title="{_text(PHASE_LABELS.get(row["phase"], row["phase"]))}: {_seconds(row["seconds"])}"></span>'
                  for row in rows)
    legend = "".join(f'<li><span class="swatch" data-phase="{_text(row["phase"])}"></span>'
                     f'{_text(PHASE_LABELS.get(row["phase"], row["phase"]))} {_seconds(row["seconds"])} '
                     f'({100 * row["seconds"] / total:.0f}%)</li>' for row in rows)
    return (f'<section id="wall-time" class="panel"><h2>Wall time of this profile</h2><p class="muted">Rank 0 from '
            f'trainer start to exit: {_seconds(total)}. Only steady updates measure training throughput.</p>'
            f'<div class="waterfall">{bar}</div><ul class="legend">{legend}</ul></section>')


def _activity_html(trace: dict) -> str:
    rows = trace.get("per_gpu") or []
    if not rows:
        return ""
    body = []
    for row in rows:
        window = row["window_s"] or 1.0
        parts = (("compute", row["compute_s"] - row["overlap_s"]), ("overlap", row["overlap_s"]),
                 ("communication", row["exposed_communication_s"]), ("memory", row["memory_s"]), ("idle", row["idle_s"]))
        bar = "".join(f'<span class="seg" data-activity="{name}" style="width:{100 * max(value, 0) / window:.3f}%" '
                      f'title="{name}: {_seconds(max(value, 0))}"></span>' for name, value in parts)
        body.append(f'<tr><th scope="row">GPU {row["gpu"]}</th><td class="bar-cell"><div class="waterfall">{bar}</div></td>'
                    f'<td>{100 * row["compute_s"] / window:.0f}%</td><td>{100 * row["overlap_s"] / window:.0f}%</td>'
                    f'<td>{100 * row["exposed_communication_s"] / window:.0f}%</td>'
                    f'<td>{100 * row["idle_s"] / window:.0f}%</td></tr>')
    legend = "".join(f'<li><span class="swatch" data-activity="{name}"></span>{label}</li>' for name, label in (
        ("compute", "Compute only"), ("overlap", "Compute + communication"), ("communication", "Exposed communication"),
        ("memory", "Memory copies"), ("idle", "Idle")))
    return (f'<section id="gpu-activity" class="panel"><h2>GPU activity in traced updates</h2>'
            f'<p class="muted">Share of the traced window on each GPU by what its kernels were doing. Exposed communication '
            f'runs with no compute kernel alongside it.</p><ul class="legend">{legend}</ul><table class="activity">'
            f'<thead><tr><th>GPU</th><th>Timeline</th><th>Compute</th><th>Overlap</th><th>Exposed communication</th>'
            f'<th>Idle</th></tr></thead><tbody>{"".join(body)}</tbody></table></section>')


def _tree_rows(nodes: list[dict], total: float, depth: int) -> str:
    items = []
    for node in nodes:
        label = STAGE_LABELS.get(node["name"], node["name"]) if depth == 0 else node["name"]
        share = 100 * node["seconds"] / total if total else 0.0
        line = (f'<span class="name" title="{_text(label)}">{_text(label)}</span><span class="meter" style="--share:{share:.3f}%"></span>'
                f'<span>{_seconds(node["seconds"])}</span><span>{share:.1f}%</span>'
                f'<span class="muted">{node["count"]:,.0f} launches</span>')
        if node["children"]:
            items.append(f'<details data-depth="{depth}"><summary>{line}</summary>'
                         f'{_tree_rows(node["children"], total, depth + 1)}</details>')
        else:
            items.append(f'<div class="leaf" data-depth="{depth}">{line}</div>')
    return "".join(items)


def _trace_html(summary: dict) -> str:
    trace = summary.get("trace_summary")
    if not trace:
        return ('<section id="kernel-tree" class="panel"><h2>Operators and kernels</h2><p class="muted">No trace in '
                'this profile. Rerun with <code>--trace torch</code> for a stage → category → operator → kernel '
                'breakdown of rank 0, or <code>--trace nsys</code> for the kernels of every rank.</p></section>')
    total = sum(node["seconds"] for node in trace["tree"])
    source = ("PyTorch profiler, operators from the CPU op that launched each kernel" if trace["source"] == "torch"
              else "Nsight Systems, kernels of every rank")
    top = "".join(f'<tr><td>{_text(row["name"])}</td><td>{_text(row["category"])}</td><td>{row["count"]:,.0f}</td>'
                  f'<td>{_seconds(row["seconds"])}</td><td>{row["mean_us"]:,.1f} µs</td></tr>' for row in trace["top_kernels"])
    small = trace.get("small_kernels", {})
    return (f'{_activity_html(trace)}<section id="kernel-tree" class="panel"><h2>Operators and kernels</h2>'
            f'<p class="muted">GPU time per traced update and rank ({source}; {trace["steps"]} update(s), '
            f'{trace["ranks"]} rank(s)). Each kernel belongs to the stage that launched it, so an asynchronous collective '
            f'appears where it was issued (the optimizer issues its parameter all-gathers while computing updates). '
            f'Kernels running concurrently on different streams are counted separately, so the total can exceed the '
            f'update time. {100 * small.get("count_share", 0):.0f}% of kernels run under '
            f'10 µs.</p><div class="tree">{_tree_rows(trace["tree"], total, 0)}</div></section>'
            f'<section id="top-kernels" class="panel"><details><summary><h2>Top kernels</h2></summary><table><thead><tr>'
            f'<th>Kernel</th><th>Category</th><th>Launches</th><th>Time</th><th>Mean</th></tr></thead><tbody>{top}'
            f'</tbody></table></details></section>')


def render_report(summary: dict) -> str:
    """Return a self-contained HTML report suitable for local file viewing."""
    metrics = summary.get("metrics_summary", {})
    warmup = metrics.get("warmup", {})
    steady = metrics.get("steady", {})
    telemetry = summary.get("telemetry_summary", {}).get("per_gpu", [])
    payload = _heatmap_payload(summary)
    low = payload["low_gpu"]
    scope_steady = summary.get("telemetry_summary", {}).get("per_gpu_scope") == "steady"
    # The trace-based findings repeat the lagging-GPU warning; show it once.
    if any(finding.get("id") == "lagging-gpu" for finding in summary.get("findings") or []):
        low = []
    warning = (
        f"<div class=\"alert\"><strong>Low utilization:</strong> GPU {', GPU '.join(map(str, low))} averaged under half "
        f"the median GPU utilization during the {'steady' if scope_steady else 'captured'} interval.</div>"
        if low else ""
    )
    status = html.escape(str(summary.get("status", "unknown")))
    scope = ("Means and peaks over steady updates; the heatmap covers the whole profile."
             if summary.get("telemetry_summary", {}).get("per_gpu_scope") == "steady"
             else "Means and peaks over the whole captured profile, including startup, evaluation and checkpoints.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NextTransformer profile</title>
<style>
:root {{ color-scheme: light dark; --bg:#0d1117; --panel:#161b22; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --warn:#d29922; }}
* {{ box-sizing:border-box; }} body {{ margin:0; padding:2rem; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1200px; margin:auto; }} h1 {{ margin:0 0 .3rem; }} h2 {{ margin-top:2rem; }} .muted {{ color:var(--muted); }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1rem; }} .card,.panel {{ background:var(--panel); border:1px solid #30363d; border-radius:10px; padding:1rem; }} .value {{ display:block; color:var(--accent); font-size:1.45rem; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); }} th,td {{ padding:.65rem .75rem; border-bottom:1px solid #30363d; text-align:left; }} th {{ color:var(--muted); font-size:.85rem; }} tr.low {{ background:rgba(210,153,34,.16); }} .alert {{ margin:1rem 0; padding:.8rem 1rem; border-left:4px solid var(--warn); background:rgba(210,153,34,.12); }} code {{ color:var(--accent); }}
.section-heading {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }} .metric-switch select {{ margin-left:.4rem; padding:.35rem; background:var(--panel); color:var(--text); border:1px solid #30363d; }} .heatmap-scroll {{ overflow-x:auto; }} .heatmap-grid {{ border-spacing:2px; border-collapse:separate; min-width:max-content; }} .heatmap-grid th,.heatmap-grid td {{ border:0; padding:.3rem .4rem; }} .heatmap-grid thead th {{ position:sticky; top:0; background:var(--panel); }} .heatmap-grid tbody th {{ position:sticky; left:0; background:var(--panel); min-width:90px; }} .heat-cell {{ --cell-value:50; width:32px; height:27px; padding:0 !important; background:color-mix(in srgb, #2f81f7 calc(var(--cell-value) * 1%), #21262d); }} .heat-cell[style*="--cell-value:"] {{ background:linear-gradient(135deg, color-mix(in srgb, #2f81f7 calc(var(--cell-value) * 1%), #21262d), color-mix(in srgb, #f0883e calc(var(--cell-value) * 1%), #21262d)); }} .low-flag {{ color:var(--warn); margin-left:.2rem; }} .heatmap-legend {{ display:flex; align-items:center; gap:.3rem; margin-top:1rem; font-size:.8rem; }} .heatmap-legend i {{ width:22px; height:12px; background:linear-gradient(90deg,#21262d,#2f81f7,#f0883e); }} .heatmap-legend .muted {{ margin-left:.8rem; }} .panel {{ margin-top:1.5rem; }} .panel h2 {{ margin-top:0; }} .findings {{ list-style:none; padding:0; margin:0; display:grid; gap:.5rem; }} .finding {{ display:flex; flex-direction:column; gap:.2rem; padding:.6rem .8rem; border-left:4px solid var(--accent); background:rgba(88,166,255,.08); }} .finding.warning {{ border-color:var(--warn); background:rgba(210,153,34,.12); }} .finding span {{ color:var(--muted); }}
.waterfall {{ display:flex; height:22px; border-radius:6px; overflow:hidden; background:#21262d; margin:.6rem 0; }} .waterfall .seg {{ display:block; height:100%; min-width:1px; }} .swatch {{ display:inline-block; width:.8rem; height:.8rem; border-radius:3px; margin-right:.35rem; vertical-align:middle; }} .legend {{ list-style:none; padding:0; display:flex; flex-wrap:wrap; gap:.4rem 1rem; font-size:.85rem; }}
[data-stage="step_setup"],[data-stage="step_tail"] {{ background:#6e7681; }} [data-stage="data"] {{ background:#a371f7; }} [data-stage="forward"] {{ background:#2f81f7; }} [data-stage="backward"] {{ background:#1f6feb; }} [data-stage="grad_norm"] {{ background:#56d364; }} [data-stage="wait"],[data-stage="unattributed"] {{ background:#484f58; }} [data-stage^="optimizer."] {{ background:#f0883e; }} [data-stage="optimizer.compute"] {{ background:#db6d28; }} [data-stage="optimizer.gather"] {{ background:#ffa657; }}
[data-phase="startup"] {{ background:#484f58; }} [data-phase="warmup"] {{ background:#a371f7; }} [data-phase="steady"] {{ background:#2f81f7; }} [data-phase="traced"] {{ background:#79c0ff; }} [data-phase="validation"] {{ background:#56d364; }} [data-phase="core"] {{ background:#3fb950; }} [data-phase="checkpoint"] {{ background:#d29922; }} [data-phase="none"] {{ background:transparent; }} .phase-cell {{ height:8px; padding:0 !important; }}
[data-activity="compute"] {{ background:#2f81f7; }} [data-activity="overlap"] {{ background:#a371f7; }} [data-activity="communication"] {{ background:#f0883e; }} [data-activity="memory"] {{ background:#56d364; }} [data-activity="idle"] {{ background:#30363d; }} .activity td.bar-cell {{ width:50%; }} .activity .waterfall {{ margin:0; }}
details > summary {{ cursor:pointer; list-style:none; }} details > summary::-webkit-details-marker {{ display:none; }} details > summary::before {{ content:"▸"; color:var(--muted); display:inline-block; width:1rem; transition:transform .1s; }} details[open] > summary::before {{ transform:rotate(90deg); }} #top-kernels summary h2 {{ display:inline; }}
.stage-row > summary {{ display:grid; grid-template-columns:1rem 1rem minmax(9rem,1.3fr) 6rem 4rem 8rem 1fr; align-items:center; gap:.4rem; padding:.35rem 0; border-bottom:1px solid #21262d; }} .stage-row .ranks {{ columns:4 10rem; font-size:.85rem; color:var(--muted); margin:.3rem 0 .6rem 2.4rem; }}
.tree {{ overflow-x:auto; }} .tree summary,.tree .leaf {{ display:grid; grid-template-columns:1rem minmax(0,2.4fr) minmax(3rem,1fr) 5.5rem 4rem 8.5rem; align-items:center; gap:.5rem; padding:.25rem 0; border-bottom:1px solid #21262d; font-size:.9rem; }} .tree .leaf {{ grid-template-columns:minmax(0,2.4fr) minmax(3rem,1fr) 5.5rem 4rem 8.5rem; padding-left:1rem; }} .tree details > details,.tree details > .leaf {{ margin-left:1.2rem; }} .tree .name {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }} .tree [data-depth="0"] > summary .name,.tree [data-depth="1"] > summary .name {{ font-family:inherit; font-size:.9rem; font-weight:600; }} .meter {{ height:8px; border-radius:4px; background:linear-gradient(90deg,var(--accent) var(--share),#21262d var(--share)); }}
</style></head><body><main>
<h1>NextTransformer profile</h1><p class="muted">Status: <code>{status}</code>. Warmup is separated from steady-state measurements.</p>
{warning}
{_findings_html(summary)}
<h2>Steady state</h2>{_shape_html(summary)}<div class="grid">
<div class="card"><span class="muted">Median step</span><span class="value">{_number(steady.get('median_step_seconds', 0), 3)} s</span></div>
<div class="card"><span class="muted">Throughput</span><span class="value">{_number(steady.get('median_tokens_per_second', 0), 0)} tok/s</span></div>
<div class="card"><span class="muted">Optimizer time</span><span class="value">{_number(steady.get('optimizer_seconds_total', 0), 3)} s</span></div>
<div class="card"><span class="muted">Data time</span><span class="value">{_number(steady.get('data_seconds_total', 0), 3)} s</span></div>
<div class="card"><span class="muted">Peak memory</span><span class="value">{_memory_gib(steady.get('peak_memory_bytes', 0))}</span></div>
</div>
{_projection_html(summary)}
{_stages_html(summary)}
{_wall_html(summary)}
{_trace_html(summary)}
{_heatmap_html(summary)}
<h2>Warmup</h2><p>{warmup.get('steps', 0)} step(s): {', '.join(f'{v:,.3f}s' for v in warmup.get('step_seconds', [])) or 'none'}</p>
<h2>GPU telemetry</h2><p class="muted">{_text(scope)}</p>{_telemetry_table(telemetry)}
<p class="muted">A static research report generated from <code>summary.json</code>. Open a PyTorch trace in Perfetto or the Nsight report in Nsight Systems for full timelines.</p>
</main></body></html>
"""


def write_report(summary_path: Path, output_path: Path) -> None:
    """Render summary JSON to an HTML file."""
    output_path.write_text(render_report(json.loads(summary_path.read_text())))
