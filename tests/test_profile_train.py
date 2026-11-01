import json
from pathlib import Path
import sys

import pytest

from launch.profile_train import _parse_args, build_command, read_metrics, summarize_metrics, summarize_telemetry


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


def test_build_command_is_bounded_and_research_only(tmp_path: Path, monkeypatch):
    args = _parse_args([
        "--site-config", "site.local.json", "--config", "configs/baseline/dense_full.json",
        "--output", str(tmp_path), "--world-size", "8", "--micro-batch", "1",
        "--warmup-steps", "1", "--steady-steps", "2",
    ])
    command = build_command(args, tmp_path / "run")
    assert "--smoke-steps" in command and command[command.index("--smoke-steps") + 1] == "3"
    assert "--formal" not in command
    assert command[0] == sys.executable

    monkeypatch.setattr("launch.profile_train.shutil.which", lambda name: "/usr/bin/nsys" if name == "nsys" else "/usr/bin/nvidia-smi")
    args.nsys = True
    nsys_command = build_command(args, tmp_path / "run")
    assert "--trace=cuda,nvtx,osrt" in nsys_command
    assert not any("nccl" in item for item in nsys_command)


def test_summarize_telemetry_groups_by_uuid(tmp_path: Path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        json.dumps({"time": 10.0, "index": 0, "uuid": "GPU-a", "memory_total": 24000,
                    "memory_used": 12000, "gpu_util": 80, "power": 200, "temperature": 60, "sm_clock": 1700}) + "\n"
        + json.dumps({"time": 11.0, "index": 0, "uuid": "GPU-a", "memory_total": 24000,
                      "memory_used": 14000, "gpu_util": 60, "power": 180, "temperature": 61, "sm_clock": 1600}) + "\n"
        + json.dumps({"time": 11.0, "index": 1, "uuid": "GPU-b", "memory_total": 24000,
                      "memory_used": 10000, "gpu_util": 50, "power": 150, "temperature": 55, "sm_clock": 1500}) + "\n"
    )

    result = summarize_telemetry(path, 10.0, 11.0)
    assert [row["uuid"] for row in result["per_gpu"]] == ["GPU-a", "GPU-b"]
    assert result["per_gpu"][0]["memory_used_peak_mib"] == 14000
    assert result["per_gpu"][0]["gpu_util_mean_pct"] == 70
