from pathlib import Path

from launch.mapping_trials import build_command, mapping_order


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
