import json
from pathlib import Path
import subprocess
import sys

import pytest

from launch.profile_report import render_report
from launch.profile_train import (_parse_args, build_command, read_metrics, summarize_metrics, summarize_telemetry,
                                  trace_window)


def test_summarize_metrics_separates_warmup_and_steady_steps():
    rows = [
        {"event": "validation", "step": 0, "seconds": 0.7},
        {"event": "train", "step": 1, "step_seconds": 19.0,
         "data_seconds": 0.2, "optimizer_seconds": 18.0,
         "tokens_per_second": 800.0, "peak_memory_bytes": 10_000},
        {"event": "train", "step": 2, "step_seconds": 1.4,
         "data_seconds": 0.001, "optimizer_seconds": 0.4,
         "tokens_per_second": 12_000, "peak_memory_bytes": 11_000},
        {"event": "train", "step": 3, "step_seconds": 1.3,
         "data_seconds": 0.001, "optimizer_seconds": 0.39,
         "tokens_per_second": 12_300, "peak_memory_bytes": 11_100},
    ]

    assert summarize_metrics(rows, warmup_steps=1) == {
        "warmup": {"steps": 1, "step_seconds": [19.0]},
        "steady": {
            "steps": 2,
            "step_seconds": [1.4, 1.3],
            "median_step_seconds": 1.35,
            "median_tokens_per_second": 12150.0,
            "data_seconds_total": 0.002,
            "optimizer_seconds_total": 0.79,
            "peak_memory_bytes": 11100,
        },
    }


def test_summarize_metrics_leaves_traced_updates_out_of_the_steady_window():
    rows = [{"event": "train", "step": step, "step_seconds": seconds, "data_seconds": 0.0,
             "optimizer_seconds": 0.1, "tokens_per_second": 1 / seconds, "peak_memory_bytes": 1}
            for step, seconds in ((1, 9.0), (2, 1.0), (3, 5.0), (4, 1.2))]

    result = summarize_metrics(rows, warmup_steps=1, traced_steps=[3])
    assert result["steady"]["step_seconds"] == [1.0, 1.2]
    assert result["traced"] == {"steps": 1, "step_seconds": [5.0]}


def test_script_imports_its_own_checkout(tmp_path: Path):
    """Run as a script from elsewhere, the launcher must not import an installed copy of the repository."""
    script = Path(__file__).resolve().parents[1] / "launch" / "profile_train.py"
    probe = (f"import runpy; runpy.run_path({str(script)!r}, run_name='profile_train'); "
             "import launch.profile_analysis, nexttransformer; print(nexttransformer.__file__)")
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith(str(script.parents[1]))


def test_read_metrics_reports_malformed_line_number(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event":"train"}\nnot-json\n')

    with pytest.raises(ValueError, match="invalid metrics JSON at line 2"):
        read_metrics(path)


def test_summarize_metrics_requires_a_steady_step():
    rows = [{"event": "train", "step": 1, "step_seconds": 1.0}]

    with pytest.raises(ValueError, match="at least 2 train events"):
        summarize_metrics(rows, warmup_steps=1)


def test_read_metrics_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('\n{"event":"train","step":1}\n\n')

    assert read_metrics(path) == [{"event": "train", "step": 1}]


def parse(tmp_path, *extra):
    return _parse_args([
        "--site-config", "site.local.json", "--config", "configs/baseline/dense_full.json",
        "--output", str(tmp_path), "--world-size", "8", "--micro-batch", "1",
        "--warmup-steps", "1", "--steady-steps", "2", *extra,
    ])


def test_build_command_is_bounded_research_only_and_checkpoints_once(tmp_path: Path):
    args = parse(tmp_path)
    command = build_command(args, tmp_path / "run")
    assert command[0] == sys.executable
    assert command[command.index("--smoke-steps") + 1] == "3"
    assert command[command.index("--rolling-every") + 1] == "4"
    assert command[command.index("-m", command.index("--nproc-per-node")) + 1] == "launch.profile_hooks"
    assert command[command.index("--profile-dir") + 1] == str(tmp_path / "stages")
    assert "--profile-trace" not in command
    assert "--smoke-accumulation" not in command
    assert "--formal" not in command
    assert "--smoke-val-tokens" not in command
    explicit = build_command(parse(tmp_path, "--smoke-accumulation", "8", "--validation-tokens", "41943040"),
                             tmp_path / "run")
    assert explicit[explicit.index("--smoke-accumulation") + 1] == "8"
    assert explicit[explicit.index("--smoke-val-tokens") + 1] == "41943040"


def test_build_command_traces_one_steady_window_with_torch_profiler(tmp_path: Path):
    args = parse(tmp_path, "--trace", "torch", "--trace-steps", "1")
    command = build_command(args, tmp_path / "run")
    assert command[0] == sys.executable
    assert command[command.index("--profile-trace") + 1] == "torch"
    first = command.index("--profile-trace-steps")
    assert command[first + 1:first + 3] == ["4", "4"]
    assert command[command.index("--profile-trace-ranks") + 1] == "0"
    assert command[command.index("--smoke-steps") + 1] == "4"


def test_build_command_wraps_nsys_with_a_capture_range(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("launch.profile_train.shutil.which", lambda name: f"/usr/bin/{name}")
    args = parse(tmp_path, "--trace", "nsys")
    command = build_command(args, tmp_path / "run")
    assert command[:2] == ["nsys", "profile"]
    assert "--capture-range=cudaProfilerApi" in command
    assert "--trace=cuda,nvtx" in command
    assert "--export=sqlite" in command
    assert not any("nccl" in item for item in command)
    assert command[command.index("--profile-trace") + 1] == "nsys"


def test_legacy_nsys_flag_selects_the_nsys_trace(tmp_path: Path):
    assert parse(tmp_path, "--nsys").trace == "nsys"


def test_trace_window_follows_the_steady_updates(tmp_path: Path):
    assert trace_window(parse(tmp_path)) is None
    assert trace_window(parse(tmp_path, "--trace", "torch", "--trace-steps", "2")) == (4, 5)


def test_parse_rejects_formal_and_oversized_traces(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse(tmp_path, "--formal")
    with pytest.raises(SystemExit):
        parse(tmp_path, "--trace", "torch", "--trace-steps", "0")


def test_render_report_is_static_and_human_readable():
    html = render_report({
        "status": "passed",
        "metrics_summary": {
            "warmup": {"steps": 1, "step_seconds": [6.2]},
            "steady": {
                "steps": 2, "median_step_seconds": 1.46,
                "median_tokens_per_second": 11184,
                "data_seconds_total": 0.002,
                "optimizer_seconds_total": 0.84,
                "peak_memory_bytes": 16793004032,
            },
        },
        "telemetry_summary": {"per_gpu": [
            {"index": 0, "uuid": "GPU-a", "gpu_util_mean_pct": 7.0,
             "memory_used_peak_mib": 18758, "power_mean_w": 90.0, "sm_clock_mean_mhz": 1183.0},
            {"index": 1, "uuid": "GPU-b", "gpu_util_mean_pct": 48.0,
             "memory_used_peak_mib": 18774, "power_mean_w": 111.0, "sm_clock_mean_mhz": 1285.0},
        ]},
    })
    assert "<!doctype html>" in html.lower()
    assert "11.2k tok/s" in html
    assert "GPU 0" in html
    assert "low utilization" in html.lower()
    assert "http" not in html


def test_render_report_contains_gpu_time_heatmap_and_metric_switch():
    html = render_report({
        "status": "passed",
        "metrics_summary": {
            "warmup": {"steps": 1, "step_seconds": [6.2]},
            "steady": {
                "steps": 2, "median_step_seconds": 1.46,
                "median_tokens_per_second": 11184,
                "data_seconds_total": 0.002,
                "optimizer_seconds_total": 0.84,
                "peak_memory_bytes": 16793004032,
            },
        },
        "telemetry_summary": {
            "per_gpu": [
                {"index": index, "uuid": f"GPU-{index}", "gpu_util_mean_pct": 10.0 if index == 0 else 48.0,
                 "memory_used_peak_mib": 18758, "memory_used_peak_pct": 73.2,
                 "power_mean_w": 111.0, "sm_clock_mean_mhz": 1285.0}
                for index in range(8)
            ],
            "samples": [
                {"time_offset_sec": 0.0, "index": 0, "uuid": "GPU-0", "gpu_util": 5.0,
                 "memory_used_pct": 72.0, "power": 90.0},
                {"time_offset_sec": 0.0, "index": 1, "uuid": "GPU-1", "gpu_util": 80.0,
                 "memory_used_pct": 73.0, "power": 190.0},
                {"time_offset_sec": 1.0, "index": 0, "uuid": "GPU-0", "gpu_util": 15.0,
                 "memory_used_pct": 74.0, "power": 100.0},
                {"time_offset_sec": 1.0, "index": 1, "uuid": "GPU-1", "gpu_util": 90.0,
                 "memory_used_pct": 75.0, "power": 200.0},
            ],
        },
    })

    assert 'id="gpu-heatmap"' in html
    assert html.count('<tr><th scope="row">GPU ') == 8
    assert html.count('data-time-bucket="') == 16
    assert 'value="gpu_util"' in html
    assert 'value="memory_used_pct"' in html
    assert 'value="power"' in html
    assert 'class="heatmap-legend"' in html
    assert "metric-switch" in html
    assert "Low utilization" in html
    assert "https://" not in html


def test_render_report_does_not_flag_uniformly_moderate_utilization():
    rows = [{"index": index, "uuid": f"GPU-{index}", "gpu_util_mean_pct": 18.0, "memory_used_peak_mib": 1,
             "power_mean_w": 1.0, "sm_clock_mean_mhz": 1.0} for index in range(8)]
    html = render_report({"status": "passed", "telemetry_summary": {"per_gpu": rows}})
    assert "Low utilization:" not in html


STAGE_SUMMARY = {"steady_steps": [2, 3], "timing": "cuda_events", "step_seconds": 1.0, "stages": [
    {"stage": "forward", "seconds": 0.3, "host_seconds": 0.1, "share": 0.3, "rank_min": 0.29, "rank_max": 0.31,
     "slowest_rank": 1, "per_rank": [[0, 0.29], [1, 0.31]]},
    {"stage": "backward", "seconds": 0.5, "host_seconds": 0.2, "share": 0.5, "rank_min": 0.5, "rank_max": 0.5,
     "slowest_rank": 0, "per_rank": [[0, 0.5], [1, 0.5]]},
    {"stage": "optimizer.reduce", "seconds": 0.2, "host_seconds": 0.2, "share": 0.2, "rank_min": 0.2,
     "rank_max": 0.2, "slowest_rank": 0, "per_rank": [[0, 0.2], [1, 0.2]]},
]}
TRACE_SUMMARY = {
    "source": "torch", "steps": 1, "ranks": 1,
    "per_gpu": [dict(gpu=0, window_s=1.0, compute_s=0.6, communication_s=0.5, overlap_s=0.2, memory_s=0.01,
                     idle_s=0.1, exposed_communication_s=0.3, kernels=100)],
    "tree": [{"name": "backward", "seconds": 0.5, "count": 10, "children": [
        {"name": "gemm", "seconds": 0.4, "count": 4, "children": [
            {"name": "AddmmBackward0 [8, 32]", "seconds": 0.4, "count": 4, "children": [
                {"name": "ampere_bf16_s16816gemm<script>", "seconds": 0.4, "count": 4, "children": []}]}]}]}],
    "top_kernels": [{"name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL", "category": "communication",
                     "count": 3, "seconds": 0.3, "mean_us": 100000.0}],
    "small_kernels": {"count_share": 0.6, "time_share": 0.04},
}


def full_summary():
    return {
        "status": "passed",
        "metrics_summary": {"warmup": {"steps": 1, "step_seconds": [9.0]},
                            "steady": {"steps": 2, "median_step_seconds": 1.0, "median_tokens_per_second": 16384,
                                       "data_seconds_total": 0.002, "optimizer_seconds_total": 0.4,
                                       "peak_memory_bytes": 1e9}},
        "shape": {"world": 8, "micro_batch": 1, "accumulation": 1, "tokens_per_update": 16384,
                  "model_flops_per_gpu_s": 6.6e12},
        "projection": {"representative": False, "accumulation": 1, "formal_accumulation": 64, "micro_batch": 1,
                       "world": 8, "updates": 46875},
        "findings": [{"id": "unrepresentative-update", "severity": "warning", "title": "Update shape differs",
                      "detail": "Profile with --smoke-accumulation 64 <now>."}],
        "wall_accounting": [{"phase": "startup", "seconds": 20.0}, {"phase": "steady", "seconds": 2.0},
                            {"phase": "checkpoint", "seconds": 18.0}],
        "stage_summary": STAGE_SUMMARY,
        "trace_summary": TRACE_SUMMARY,
        "telemetry_summary": {"per_gpu_scope": "steady", "per_gpu": [], "samples": [
            {"time_offset_sec": 0.0, "index": 0, "gpu_util": 50.0, "memory_used_pct": 50.0, "power": 100.0}],
            "phase_bands": [{"phase": "startup", "start": 0.0, "end": 20.0},
                            {"phase": "checkpoint", "start": 22.0, "end": 40.0}]},
    }


def test_render_report_shows_findings_stage_waterfall_and_drilldown_tree():
    html = render_report(full_summary())

    assert 'id="findings"' in html and "Update shape differs" in html
    assert "&lt;now&gt;" in html and "<now>" not in html
    assert 'id="stages"' in html and "Backward" in html and 'class="rank-grid"' in html
    assert "50.0%" in html
    assert 'id="wall-time"' in html and 'data-part="checkpoint"' in html
    assert 'id="gpu-activity"' in html and "Exposed communication" in html
    assert 'id="kernel-tree"' in html
    assert html.count("<details") >= 3
    assert "AddmmBackward0 [8, 32]" in html
    content = html.split("<body>")[1].rsplit("<script>", 1)[0]
    assert "ampere_bf16_s16816gemm&lt;script&gt;" in content and "<script>" not in content
    assert 'id="top-kernels"' in html and "ncclDevKernel_AllReduce_Sum_f32_RING_LL" in html
    assert "16,384 tokens per update" in html
    assert "native attention sdpa" in render_report(dict(full_summary(), software={"native_attention": "sdpa"}))
    assert "https://" not in html and "http://" not in html


def test_render_report_explains_missing_trace_and_stage_data():
    summary = full_summary()
    for key in ("stage_summary", "trace_summary", "wall_accounting", "findings", "shape", "projection"):
        summary.pop(key)
    html = render_report(summary)
    assert "--trace torch" in html
    assert 'id="kernel-tree"' in html


def test_render_report_shows_the_formal_projection_when_the_shape_matches():
    summary = full_summary()
    summary["projection"] = {"representative": True, "accumulation": 64, "formal_accumulation": 64, "micro_batch": 1,
                             "world": 8, "updates": 46875, "training_hours": 209.9, "validation_events": 189,
                             "validation_hours": 3.5, "checkpoints": 94, "checkpoint_hours": 0.5,
                             "estimated_hours": 213.9}
    html = render_report(summary)
    assert 'id="projection"' in html
    assert "213.9 h" in html and "8.9 days" in html
    assert "CORE-22" in html
    for key in ("validation_hours",):
        summary["projection"].pop(key)
    summary["projection"].update(estimated_hours=210.4, validation_measured_tokens=16384)
    html = render_report(summary)
    assert "--validation-tokens" in html and "16,384 tokens" in html


def write_samples(path: Path, rows):
    path.write_text("".join(json.dumps(dict(memory_total=24000, temperature=60, sm_clock=1700, **row)) + "\n"
                            for row in rows))


def test_summarize_telemetry_keeps_time_bucket_samples(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    write_samples(path, [
        {"time": 10.0, "index": 0, "uuid": "GPU-a", "memory_used": 12000, "gpu_util": 80, "power": 200},
        {"time": 11.0, "index": 0, "uuid": "GPU-a", "memory_used": 14000, "gpu_util": 60, "power": 180},
    ])

    result = summarize_telemetry(path, 10.0, 11.0)
    assert result["samples"] == [
        {"time_offset_sec": 0.0, "index": 0, "uuid": "GPU-a", "gpu_util": 80.0,
         "memory_used_pct": 50.0, "power": 200.0},
        {"time_offset_sec": 1.0, "index": 0, "uuid": "GPU-a", "gpu_util": 60.0,
         "memory_used_pct": 58.333333333333336, "power": 180.0},
    ]


def test_summarize_telemetry_groups_by_uuid(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    write_samples(path, [
        {"time": 10.0, "index": 0, "uuid": "GPU-a", "memory_used": 12000, "gpu_util": 80, "power": 200},
        {"time": 11.0, "index": 0, "uuid": "GPU-a", "memory_used": 14000, "gpu_util": 60, "power": 180},
        {"time": 11.0, "index": 1, "uuid": "GPU-b", "memory_used": 10000, "gpu_util": 50, "power": 150},
    ])

    result = summarize_telemetry(path, 10.0, 11.0)
    assert [row["uuid"] for row in result["per_gpu"]] == ["GPU-a", "GPU-b"]
    assert result["per_gpu"][0]["memory_used_peak_mib"] == 14000
    assert result["per_gpu"][0]["gpu_util_mean_pct"] == 70
    assert result["per_gpu_scope"] == "captured"


def test_summarize_telemetry_reports_per_gpu_over_the_steady_interval(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    write_samples(path, [
        {"time": 10.0, "index": 0, "uuid": "GPU-a", "memory_used": 1000, "gpu_util": 0, "power": 50},
        {"time": 20.0, "index": 0, "uuid": "GPU-a", "memory_used": 1000, "gpu_util": 90, "power": 300},
        {"time": 30.0, "index": 0, "uuid": "GPU-a", "memory_used": 1000, "gpu_util": 0, "power": 50},
    ])

    result = summarize_telemetry(path, 10.0, 30.0, steady=(15.0, 25.0))
    assert result["per_gpu_scope"] == "steady"
    assert result["per_gpu"][0]["gpu_util_mean_pct"] == 90
    assert result["steady_offsets"] == [5.0, 15.0]
    assert len(result["samples"]) == 3


MFU_SUMMARY = {
    "achieved_flops": 38.1e12, "mfu": 0.535, "peak_flops": 71.2e12, "clock_peak_flops": 78.7e12, "clock_mfu": 0.484,
    "phases": [{"phase": "forward", "seconds": 5.0, "flops": 2e14, "achieved_flops": 4e13, "mfu": 0.56},
               {"phase": "backward", "seconds": 10.5, "flops": 4e14, "achieved_flops": 3.8e13, "mfu": 0.53},
               {"phase": "optimizer", "seconds": 0.4, "flops": 3e12, "achieved_flops": 7.5e12, "mfu": 0.1},
               {"phase": "rest of the update", "seconds": 0.2, "flops": 0.0, "achieved_flops": None, "mfu": None}],
    "kernels": [{"kind": "gemm", "seconds": 8.6, "flops": 5.5e14, "tflops": 64.4, "efficiency": 0.9,
                 "clock_efficiency": 0.82},
                {"kind": "attention", "seconds": 1.35, "flops": 5.7e13, "tflops": 42.1, "efficiency": 0.59,
                 "clock_efficiency": 0.53}],
    "waterfall": [{"part": "ideal", "seconds": 8.6, "share": 0.535},
                  {"part": "gemm below peak", "seconds": 0.8, "share": 0.05},
                  {"part": "other compute kernels", "seconds": 4.6, "share": 0.29},
                  {"part": "exposed communication", "seconds": 1.0, "share": 0.06}],
    "traced_window_s": 16.07,
}
COMMUNICATION = [{"collective": "all_reduce", "dtype": "Float", "stage": "backward",
                  "kernel": "ncclDevKernel_AllReduce_Sum_f32_RING_LL", "calls": 71, "seconds": 0.57, "exposed_s": 0.44,
                  "bytes": 2.8e9, "ranks": 8, "algbw_gbps": 4.9, "busbw_gbps": 8.6}]
GEMM = [{"kind": "gemm", "stage": "forward", "op": "aten::mm", "shape": "2048×1536 · 1536×1536",
         "dtype": "c10::BFloat16", "calls": 5888, "seconds": 1.31, "flops": 5.7e13, "tflops": 43.3,
         "kernel": "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8"}]
COMPONENTS = [{"name": "attn", "forward_s": 1.5, "backward_s": 2.8, "flops": 1.4e14,
               "layers": [{"layer": 0, "forward_s": 0.07, "backward_s": 0.12}]}]
TIMELINE = [{"name": "update", "start_s": 0.0, "seconds": 16.07, "buckets": 4, "lanes": {
    "stage": [[0, 2, "forward", 1.0], [2, 4, "backward", 1.0]], "compute": [[0, 3, "gemm", 0.9]],
    "communication": [[3, 4, "all_reduce", 0.5]], "memory": []}}]


def rich_summary():
    summary = full_summary()
    summary["hardware"] = {"device": {"name": "Example GPU", "major": 8, "minor": 6, "sms": 82},
                           "dense_bf16_flops": 71.2e12, "observed_clock_flops": 78.7e12,
                           "basis": "sm86: 82 SMs × 512 FLOP/clock × 1695 MHz (specification boost clock)"}
    summary["mfu_summary"] = MFU_SUMMARY
    summary["memory_summary"] = {"per_rank": [{"rank": 0, "allocated_peak": 18 * 2**30, "reserved_peak": 20 * 2**30}],
                                 "total_bytes": 24 * 2**30, "allocated_peak": 18 * 2**30, "reserved_peak": 20 * 2**30,
                                 "headroom_bytes": 4 * 2**30, "reserved_share": 20 / 24}
    summary["trace_summary"] = dict(TRACE_SUMMARY, communication=COMMUNICATION, gemm=GEMM, components=COMPONENTS,
                                    timeline=TIMELINE, syncs=[{"api": "cudaStreamSynchronize", "source": "aten::to",
                                                               "stage": "backward", "calls": 64, "seconds": 4.58}],
                                    launch={"api_calls": 195255, "kernels": 195255, "memory_ops": 17118, "host_s": 2.8,
                                            "micro_batches": 64, "per_micro_batch": 3050.9},
                                    idle={"total_s": 0.39, "host_s": 0.21, "launch_s": 0.01, "dependency_s": 0.17,
                                          "gaps": 209665, "largest": [{"gpu": 0, "seconds": 0.024, "start_s": 7.2,
                                                                        "cause": "host", "stage": "data",
                                                                        "next": "Memcpy HtoD"}]})
    return summary


def test_render_report_shows_mfu_waterfall_and_phase_efficiency():
    html = render_report(rich_summary())
    assert 'id="mfu"' in html and "53.5%" in html and "38.1 TFLOP/s" in html
    assert "71.2 TFLOP/s" in html and "sm86: 82 SMs" in html
    assert 'data-part="other compute kernels"' in html and "4.60 s" in html
    assert "Forward" in html and "56.0%" in html
    assert "64.4 TFLOP/s" in html and "90.0%" in html


def test_render_report_shows_communication_scheduling_matrices_components_memory_and_timeline():
    html = render_report(rich_summary())
    assert 'id="communication"' in html and "8.6 GB/s" in html and "440.0 ms" in html
    assert 'id="scheduling"' in html and "3,051 kernels per micro-batch" in html and "aten::to" in html
    assert 'id="matrix-shapes"' in html and "2048×1536 · 1536×1536" in html and "43.3" in html
    assert 'id="components"' in html and "attn" in html and "layer 0" in html
    assert 'id="memory"' in html and "20.00 GiB" in html and "4.00 GiB" in html
    assert 'id="timeline"' in html and html.count('class="timeline"') == 1
    assert html.count('data-lane="stage"') == 2 and 'class="area tone-use"' not in html
    assert 'data-lane="communication" data-label="all_reduce"' in html


def test_render_report_timeline_draws_the_compute_mix():
    summary = rich_summary()
    summary["trace_summary"]["timeline"] = [dict(TIMELINE[0], mix=[[0.5, 0.25], [1.0, 0.0], [0.0, 0.0], [0.2, 0.7]])]
    html = render_report(summary)
    assert 'class="area tone-use"' in html and 'class="area tone-loss"' in html


def test_render_report_has_two_themes_a_hero_bar_and_table_views():
    html = render_report(rich_summary(), back="index.html")
    assert "prefers-color-scheme:dark" in html and ":root[data-theme='dark']" in html
    assert 'class="theme"' in html and 'href="index.html"' in html
    assert 'id="mfu-loss"' in html and 'class="stack stack-lg"' in html
    assert html.count('class="table-view"') >= 5
    assert 'class="meter"' in html
    assert "https://" not in html and "http://" not in html


def test_write_site_writes_one_page_per_profile_and_an_index(tmp_path: Path):
    from launch.profile_report import write_site
    paths = []
    for name, method in (("profile-a", "dense_full"), ("profile-b", "qsa_hybrid")):
        directory = tmp_path / name
        directory.mkdir()
        summary = rich_summary()
        summary["run"] = {"method": method, "config": f"configs/x/{method}.json", "name": name}
        (directory / "summary.json").write_text(json.dumps(summary))
        paths.append(directory / "summary.json")
    pages = write_site(paths, tmp_path / "site")
    assert [page.name for page in pages] == ["index.html", "profile-a.html", "profile-b.html"]
    index = pages[0].read_text()
    assert 'href="profile-a.html"' in index and "qsa_hybrid" in index and 'class="compare-row"' in index
    assert "<h1>qsa_hybrid</h1>" in pages[2].read_text()


def test_build_command_passes_sparse_profiling_overrides(tmp_path: Path):
    command = build_command(parse(tmp_path, "--sparse-backend", "kernel", "--sparse-stage", "joint",
                                  "--flex-options", '{"fwd_num_stages": 2}'), tmp_path / "run")
    assert command[command.index("--sparse-backend") + 1] == "kernel"
    assert command[command.index("--profile-sparse-stage") + 1] == "joint"
    assert json.loads(command[command.index("--profile-flex-options") + 1]) == {"fwd_num_stages": 2}
    assert command.index("--profile-flex-options") < command.index("--site-config")


GANTT = {"step": 2, "seconds": 4.1, "rows": [
    {"rank": 0, "segments": [[0.0, 0.1, "step_setup"], [0.1, 1.0, "forward"], [1.1, 2.0, "backward"], [3.1, 0.5, "optimizer.gather"]]},
    {"rank": 1, "segments": [[0.5, 0.1, "step_setup"], [0.6, 1.0, "forward"], [1.6, 2.0, "backward"], [3.6, 0.5, "optimizer.gather"]]}]}
PIPELINE = [{"name": "update", "start_s": 0.0, "seconds": 16.0, "buckets": 4, "lead": [0.002, None, 5e-6, 0.0004],
             "lanes": [{"name": "Host thread", "runs": [[0, 2, "launch", 0.8], [3, 4, "sync", 1.0]]},
                       {"name": "Stream 7", "runs": [[0, 3, "gemm", 0.9], [3, 4, "elementwise", 0.4]]},
                       {"name": "Stream 23", "runs": [[3, 4, "all_reduce", 0.6]]}]}]


def test_render_report_draws_a_rank_gantt_and_a_stream_pipeline():
    summary = rich_summary()
    summary["rank_gantt"] = GANTT
    summary["trace_summary"]["pipeline"] = PIPELINE
    html = render_report(summary)
    assert 'id="gantt"' in html and html.count('class="gantt-row"') == 2
    assert 'data-stage="backward"' in html and "rank 1" in html
    assert 'id="pipeline"' in html and "Host thread" in html and "Stream 23" in html
    assert 'class="lead"' in html and "2.0 ms" in html
    assert 'href="#gantt"' in html and 'href="#pipeline"' in html
