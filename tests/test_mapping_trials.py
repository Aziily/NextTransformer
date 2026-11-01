from pathlib import Path

from launch.mapping_trials import build_command, mapping_order, summarize_steady


def test_mapping_presets_keep_all_devices_once():
    islands = mapping_order("islands", 8)
    interleaved = mapping_order("interleaved", 8)

    assert islands == (0, 1, 2, 3, 4, 5, 6, 7)
    assert interleaved == (0, 4, 1, 5, 2, 6, 3, 7)
    assert sorted(islands) == sorted(interleaved) == list(range(8))


def test_build_command_is_bounded_and_uses_mapping(tmp_path: Path):
    command = build_command(
        repo=tmp_path,
        site_config=Path("site.local.json"),
        config=Path("configs/baseline/dense_full.json"),
        output=tmp_path / "run",
        world_size=8,
        micro_batch=1,
        smoke_steps=3,
        smoke_accumulation=1,
    )

    assert command[:4] == ["python", "-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node" in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("--smoke-steps") + 1] == "3"
    assert "--formal" not in command
    assert str(tmp_path / "run") in command


def test_summarize_steady_uses_interpolated_percentiles_and_true_median():
    rows = [
        {"event": "train", "step_seconds": 10.0, "tokens_per_second": 100.0},
        {"event": "train", "step_seconds": 1.0, "tokens_per_second": 200.0},
        {"event": "train", "step_seconds": 2.0, "tokens_per_second": 300.0},
        {"event": "train", "step_seconds": 3.0, "tokens_per_second": 400.0},
        {"event": "train", "step_seconds": 4.0, "tokens_per_second": 500.0},
    ]

    result = summarize_steady(rows, warmup_steps=1)
    assert result["steady_steps"] == 4
    assert result["steady_step_seconds"] == [1.0, 2.0, 3.0, 4.0]
    assert result["median_step_seconds"] == 2.5
    assert result["p95_step_seconds"] == 3.85
    assert result["median_tokens_per_second"] == 350.0
    assert result["steady_complete"] is True


def test_summarize_steady_handles_missing_samples():
    assert summarize_steady([], warmup_steps=1) == {
        "steady_steps": 0,
        "steady_complete": False,
        "steady_step_seconds": [],
        "steady_tokens_per_second": [],
        "median_step_seconds": None,
        "p95_step_seconds": None,
        "stdev_step_seconds": None,
        "median_tokens_per_second": None,
        "mean_tokens_per_second": None,
    }
