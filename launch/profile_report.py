"""Render a self-contained human-readable HTML report from profiler JSON."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path


def _number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _memory_gib(value: int | float) -> str:
    return f"{value / (1024 ** 3):,.2f} GiB"


def _telemetry_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">No GPU telemetry samples fell inside the profile interval.</p>'
    body = []
    for row in rows:
        warning = ' class="low"' if row.get("gpu_util_mean_pct", 0) < 20 else ""
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
    low_gpu = set()
    for row in per_gpu:
        if float(row.get("gpu_util_mean_pct", 100)) < 20:
            low_gpu.add(int(row.get("index", 0)))
    for gpu in range(max_gpu + 1):
        cells = []
        observed_util = []
        for bucket in range(len(offsets)):
            cell = values.get((gpu, bucket), {})
            if "gpu_util" in cell:
                observed_util.append(cell["gpu_util"])
            cells.append({"bucket": bucket, **cell})
        if observed_util and sum(observed_util) / len(observed_util) < 20:
            low_gpu.add(gpu)
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
<div class="heatmap-scroll"><table class="heatmap-grid"><thead><tr><th scope="col">GPU</th>{columns}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
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


def _timeline_html(summary: dict) -> str:
    stages = summary.get("timeline", {})
    labels = (
        ("data", "Data"), ("forward_backward", "Forward / backward"),
        ("gradient_clip", "Gradient clip"), ("optimizer_reduce", "Optimizer reduce"),
        ("optimizer_compute", "Optimizer compute"), ("optimizer_gather", "Optimizer gather"),
        ("sync", "Sync"),
    )
    items = []
    for key, label in labels:
        value = stages.get(key)
        text = "unavailable" if value is None else f"{_number(float(value), 3)} s"
        items.append(f'<li data-stage="{key}"><b>{label}</b><span>{text}</span></li>')
    return '<section class="panel timeline-panel"><h2>Training timeline</h2><p class="muted">Stage timings are shown only when emitted by the trainer; telemetry does not infer optimizer or NCCL phases.</p><ol class="timeline">' + ''.join(items) + '</ol></section>'


def render_report(summary: dict) -> str:
    """Return a self-contained HTML report suitable for local file viewing."""
    metrics = summary.get("metrics_summary", {})
    warmup = metrics.get("warmup", {})
    steady = metrics.get("steady", {})
    telemetry = summary.get("telemetry_summary", {}).get("per_gpu", [])
    payload = _heatmap_payload(summary)
    low = payload["low_gpu"]
    warning = (
        f"<div class=\"alert\"><strong>Low utilization:</strong> GPU {', GPU '.join(map(str, low))} averaged below 20% during the captured interval.</div>"
        if low else ""
    )
    status = html.escape(str(summary.get("status", "unknown")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NextTransformer profile</title>
<style>
:root {{ color-scheme: light dark; --bg:#0d1117; --panel:#161b22; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --warn:#d29922; }}
* {{ box-sizing:border-box; }} body {{ margin:0; padding:2rem; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1200px; margin:auto; }} h1 {{ margin:0 0 .3rem; }} h2 {{ margin-top:2rem; }} .muted {{ color:var(--muted); }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1rem; }} .card,.panel {{ background:var(--panel); border:1px solid #30363d; border-radius:10px; padding:1rem; }} .value {{ display:block; color:var(--accent); font-size:1.45rem; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); }} th,td {{ padding:.65rem .75rem; border-bottom:1px solid #30363d; text-align:left; }} th {{ color:var(--muted); font-size:.85rem; }} tr.low {{ background:rgba(210,153,34,.16); }} .alert {{ margin:1rem 0; padding:.8rem 1rem; border-left:4px solid var(--warn); background:rgba(210,153,34,.12); }} code {{ color:var(--accent); }}
.section-heading {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }} .metric-switch select {{ margin-left:.4rem; padding:.35rem; background:var(--panel); color:var(--text); border:1px solid #30363d; }} .heatmap-scroll {{ overflow-x:auto; }} .heatmap-grid {{ border-spacing:2px; border-collapse:separate; min-width:max-content; }} .heatmap-grid th,.heatmap-grid td {{ border:0; padding:.3rem .4rem; }} .heatmap-grid thead th {{ position:sticky; top:0; background:var(--panel); }} .heatmap-grid tbody th {{ position:sticky; left:0; background:var(--panel); min-width:90px; }} .heat-cell {{ --cell-value:50; width:32px; height:27px; padding:0 !important; background:color-mix(in srgb, #2f81f7 calc(var(--cell-value) * 1%), #21262d); }} .heat-cell[style*="--cell-value:"] {{ background:linear-gradient(135deg, color-mix(in srgb, #2f81f7 calc(var(--cell-value) * 1%), #21262d), color-mix(in srgb, #f0883e calc(var(--cell-value) * 1%), #21262d)); }} .low-flag {{ color:var(--warn); margin-left:.2rem; }} .heatmap-legend {{ display:flex; align-items:center; gap:.3rem; margin-top:1rem; font-size:.8rem; }} .heatmap-legend i {{ width:22px; height:12px; background:linear-gradient(90deg,#21262d,#2f81f7,#f0883e); }} .heatmap-legend .muted {{ margin-left:.8rem; }} .timeline {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:.5rem; padding:0; list-style:none; }} .timeline li {{ display:flex; flex-direction:column; gap:.15rem; padding:.65rem; border-left:3px solid #30363d; }} .timeline li span {{ color:var(--muted); font-size:.85rem; }}
</style></head><body><main>
<h1>NextTransformer profile</h1><p class="muted">Status: <code>{status}</code>. Warmup is separated from steady-state measurements.</p>
{warning}
<h2>Steady state</h2><div class="grid">
<div class="card"><span class="muted">Median step</span><span class="value">{_number(steady.get('median_step_seconds', 0), 3)} s</span></div>
<div class="card"><span class="muted">Throughput</span><span class="value">{_number(steady.get('median_tokens_per_second', 0), 0)} tok/s</span></div>
<div class="card"><span class="muted">Optimizer time</span><span class="value">{_number(steady.get('optimizer_seconds_total', 0), 3)} s</span></div>
<div class="card"><span class="muted">Data time</span><span class="value">{_number(steady.get('data_seconds_total', 0), 3)} s</span></div>
<div class="card"><span class="muted">Peak memory</span><span class="value">{_memory_gib(steady.get('peak_memory_bytes', 0))}</span></div>
</div>
{_heatmap_html(summary)}
{_timeline_html(summary)}
<h2>Warmup</h2><p>{warmup.get('steps', 0)} step(s): {', '.join(f'{v:,.3f}s' for v in warmup.get('step_seconds', [])) or 'none'}</p>
<h2>GPU telemetry</h2>{_telemetry_table(telemetry)}
<p class="muted">This is a static research report generated from <code>summary.json</code>. Open the optional Nsight report for kernel and communication timelines.</p>
</main></body></html>
"""


def write_report(summary_path: Path, output_path: Path) -> None:
    """Render summary JSON to an HTML file."""
    output_path.write_text(render_report(json.loads(summary_path.read_text())))
