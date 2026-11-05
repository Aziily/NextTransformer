import json
import runpy
import sys

import pytest

from nexttransformer_gym.common import GymError
from nexttransformer_gym.history import research_context

from test_gym import ROOT, run_job, context as make_context


@pytest.fixture(name="context")
def research_session(tmp_path):
    return make_context.__wrapped__(tmp_path)


def test_published_history_exposes_results_and_native_architecture(context):
    references, history = research_context(ROOT)
    service, _, call, _ = context
    service.catalog, service.history = references, history
    page = call("list_experiments", limit=2)
    assert len(page["items"]) == 2 and page["next_cursor"]
    second = call("list_experiments", cursor=page["next_cursor"], limit=2)
    assert {x["id"] for x in page["items"]}.isdisjoint(x["id"] for x in second["items"])
    past = call("read_experiment", experiment_id="moba_hybrid")
    assert past["architecture"]["matches_recorded_model_hash"] is True
    assert past["architecture"]["model"]["method"] == "moba_hybrid"
    expected = json.loads((ROOT / "results/runs/moba_hybrid.json").read_text())
    assert past["result"] == expected
    historical_only = call("read_experiment", experiment_id="gqa")
    assert historical_only["architecture"] is None
    assert "three key/value heads" in historical_only["method_card"]["description"]
    source = "source:nexttransformer/models/sparse.py"
    assert source in past["source_reference_ids"]
    assert (
        call("read_reference", resource_id=source)["content"]
        == (ROOT / "nexttransformer/models/sparse.py").read_text()
    )
    with pytest.raises(GymError):
        call("read_reference", resource_id="../../.git/config")
    ids = []
    cursor = ""
    while True:
        page = call("list_components", cursor=cursor, limit=2)
        ids.extend(x["id"] for x in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert ids == sorted(references)


def test_session_history_and_actual_execution_node(context):
    service, _, call, candidate = context
    resources = call("get_resources")
    assert resources["server_count"] == 1
    node = resources["servers"][0]
    assert node["cpu_capacity"] > 0 and node["memory_limit_bytes"] > 0
    result = run_job(context, target_update=1)
    jobs = call("list_experiments", scope="session")["items"]
    detail = call("read_experiment", scope="session", experiment_id=jobs[0]["id"])
    assert detail["result"] == result
    assert detail["server_id"] == node["id"]
    assert detail["candidate"]["id"] == candidate["id"]
    assert "plugin/research_model/__init__.py" in detail["files"]
    other = service.create_session(10**12, "other")
    with pytest.raises(GymError):
        service.call(
            other["agent_token"],
            "read_experiment",
            {"scope": "session", "experiment_id": jobs[0]["id"]},
        )
    # A different session's uncertain work still occupies the single global slot.
    with service.store.transaction() as db:
        service.store.put(db, other["session_id"], "job", {"state": "reconciliation"})
    assert call("get_resources")["allocatable_slots"] == 0


@pytest.mark.parametrize("example", ["claude-code", "deepseek-v4.1-flash"])
def test_examples_use_native_claude_and_preserve_total_budget(monkeypatch, example):
    from harbor.agents.installed.claude_code import ClaudeCodeOptions

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    monkeypatch.setenv("GYM_RESEARCH_MODEL", "gateway-model-id")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trial.py",
            "--config",
            str(ROOT / f"gym/examples/{example}.toml"),
            "--dry-run",
        ],
    )
    args = runner["arguments"]()
    assert args.agent == "claude-code"
    assert args.research_flops_limit == 10**12
    native = runner["agent_arguments"](args)
    assert native[:2] == ["--agent", "claude-code"]
    options = dict(
        native[i + 1].split("=", 1)
        for i, item in enumerate(native)
        if item == "--agent-kwarg"
    )
    parsed = ClaudeCodeOptions.model_validate(options)
    assert parsed.max_turns == 64 and parsed.max_budget_usd == "5.00"
    if example.startswith("deepseek"):
        assert args.model == "gateway-model-id"


def test_launcher_rejects_unbounded_budget_before_docker(monkeypatch):
    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trial.py",
            "--config",
            str(ROOT / "gym/examples/claude-code.toml"),
            "--research-flops-limit",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        runner["arguments"]()


def test_launcher_freezes_gpu_uuid_and_mounts_policy_only_in_gym(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    import yaml

    from nexttransformer_gym.gpu import configuration

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    gpu_uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    preflight = runner["check_gpu_runtime"]
    monkeypatch.setitem(
        preflight.__globals__,
        "discover",
        lambda: {"status": "ok", "gpus": [{"index": 3, "uuid": gpu_uuid}]},
    )
    probes = []

    def probe(command, **kwargs):
        probes.append(command)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"cuda": "12.8", "gpu_ids": [gpu_uuid]})
        )

    monkeypatch.setattr(subprocess, "run", probe)
    args = SimpleNamespace(
        execution_policy=configuration({"gpu_ids": [3]}), service_image="cuda-service"
    )
    preflight(args)
    assert args.execution_policy["gpu_ids"] == [gpu_uuid]
    assert len(probes) == 1 and probes[0][:2] == ["docker", "run"]
    command = probes[0]
    assert command[command.index("--gpus") + 1] == f'"device={gpu_uuid}"'
    assert "--runtime" not in command
    assert "torch.version.cuda" in command[-1] and "torch.cuda" not in command[-1]

    task = tmp_path / "task"
    shutil.copytree(ROOT / "gym/tasks/architecture-research", task)
    compose_path = task / "environment/docker-compose.yaml"
    original = yaml.safe_load(compose_path.read_text())
    runner["configure_execution"](task, args.execution_policy, tmp_path, "fixture")
    compose = yaml.safe_load(compose_path.read_text())
    assert compose["services"]["main"] == original["services"]["main"]
    gym = compose["services"]["gym"]
    devices = gym["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {"driver": "nvidia", "device_ids": [gpu_uuid], "capabilities": ["gpu"]}
    ]
    mount = gym["volumes"][-1]
    snapshot = tmp_path / "fixture.execution.json"
    assert mount == {
        "type": "bind",
        "source": str(snapshot.resolve()),
        "target": "/run/gym-execution/config.json",
        "read_only": True,
        "bind": {"create_host_path": False},
    }
    assert gym["environment"]["GYM_EXECUTION_CONFIG"] == mount["target"]
    assert json.loads(snapshot.read_text()) == args.execution_policy


@pytest.mark.parametrize("case", ["unavailable", "missing", "aliased"])
def test_launcher_rejects_unresolved_or_duplicate_physical_gpus(monkeypatch, case):
    from nexttransformer_gym.gpu import configuration

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    resolve = runner["resolve_gpu_policy"]
    gpu_uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    observation = {"status": "ok", "gpus": [{"index": 3, "uuid": gpu_uuid}]}
    selectors = [3, gpu_uuid] if case == "aliased" else [7]
    if case == "unavailable":
        observation = {"status": "unavailable", "gpus": []}
    monkeypatch.setitem(resolve.__globals__, "discover", lambda: observation)
    with pytest.raises(SystemExit):
        resolve(configuration({"gpu_ids": selectors}))


@pytest.mark.parametrize(
    "case", ["device_request_failed", "cpu_image", "wrong_gpu", "malformed"]
)
def test_launcher_rejects_failed_gpu_device_request_probe(monkeypatch, case):
    import subprocess
    from types import SimpleNamespace

    from nexttransformer_gym.gpu import configuration

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    preflight = runner["check_gpu_runtime"]
    gpu_uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    monkeypatch.setitem(
        preflight.__globals__,
        "discover",
        lambda: {"status": "ok", "gpus": [{"index": 3, "uuid": gpu_uuid}]},
    )
    result = {"cuda": None if case == "cpu_image" else "12.8", "gpu_ids": [gpu_uuid]}
    if case == "wrong_gpu":
        result["gpu_ids"] = ["GPU-87654321-1234-1234-1234-123456789abc"]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1 if case == "device_request_failed" else 0,
            stdout="invalid" if case == "malformed" else json.dumps(result),
        ),
    )
    args = SimpleNamespace(
        execution_policy=configuration({"gpu_ids": [3]}), service_image="cuda-service"
    )
    with pytest.raises(SystemExit):
        preflight(args)
