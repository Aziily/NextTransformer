"""Render a self-contained human-readable HTML report from profiler JSON."""

from __future__ import annotations

import html
import json
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


def render_report(summary: dict) -> str:
    """Return a self-contained HTML report suitable for local file viewing."""
    metrics = summary.get("metrics_summary", {})
    warmup = metrics.get("warmup", {})
    steady = metrics.get("steady", {})
    telemetry = summary.get("telemetry_summary", {}).get("per_gpu", [])
    low = [row for row in telemetry if row.get("gpu_util_mean_pct", 0) < 20]
    warning = (
        f"<div class=\"alert\"><strong>Low utilization:</strong> {len(low)} GPU(s) averaged below 20% during the captured interval.</div>"
        if low else ""
    )
    status = html.escape(str(summary.get("status", "unknown")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NextTransformer profile</title>
<style>
:root {{ color-scheme: light dark; --bg:#0d1117; --panel:#161b22; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --warn:#d29922; }}
* {{ box-sizing:border-box; }} body {{ margin:0; padding:2rem; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1100px; margin:auto; }} h1 {{ margin:0 0 .3rem; }} h2 {{ margin-top:2rem; }} .muted {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1rem; }} .card {{ background:var(--panel); border:1px solid #30363d; border-radius:10px; padding:1rem; }} .value {{ display:block; color:var(--accent); font-size:1.45rem; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); }} th,td {{ padding:.65rem .75rem; border-bottom:1px solid #30363d; text-align:left; }} th {{ color:var(--muted); font-size:.85rem; }} tr.low {{ background:rgba(210,153,34,.16); }} .alert {{ margin:1rem 0; padding:.8rem 1rem; border-left:4px solid var(--warn); background:rgba(210,153,34,.12); }} code {{ color:var(--accent); }}
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
<h2>Warmup</h2><p>{warmup.get('steps', 0)} step(s): {', '.join(f'{v:,.3f}s' for v in warmup.get('step_seconds', [])) or 'none'}</p>
<h2>GPU telemetry</h2>{_telemetry_table(telemetry)}
<p class="muted">This is a static research report generated from <code>summary.json</code>. Open the optional Nsight report for kernel and communication timelines.</p>
</main></body></html>
"""


def write_report(summary_path: Path, output_path: Path) -> None:
    """Render summary JSON to an HTML file."""
    output_path.write_text(render_report(json.loads(summary_path.read_text())))
