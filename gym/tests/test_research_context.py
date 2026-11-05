import json
import runpy
import sys

import pytest
from nexttransformer_gym.common import GymError
from nexttransformer_gym.history import research_context
from test_gym import ROOT, run_job
from test_gym import context as make_context


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


@pytest.mark.parametrize("usd_limit", [None, "1.25"])
def test_launcher_optional_api_budget(monkeypatch, usd_limit):
    from harbor.agents.installed.claude_code import ClaudeCodeOptions

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    argv = [
        "run_trial.py",
        "--model",
        "test-model",
        "--research-flops-limit",
        "1000000000000",
        "--client-image",
        "nt-gym-client:local",
        "--verifier-image",
        "nt-gym-verifier:local",
    ]
    if usd_limit is not None:
        argv += ["--max-budget-usd", usd_limit]
    monkeypatch.setattr(sys, "argv", argv)
    args = runner["arguments"]()
    native = runner["agent_arguments"](args)
    options = dict(
        native[i + 1].split("=", 1)
        for i, item in enumerate(native)
        if item == "--agent-kwarg"
    )
    assert args.research_flops_limit == 10**12
    assert ("max_budget_usd" in options) == (usd_limit is not None)
    assert ClaudeCodeOptions.model_validate(options).max_budget_usd == usd_limit


@pytest.mark.parametrize("usd_limit", ["0", "-1", "nan", "inf", "invalid", ""])
def test_launcher_rejects_invalid_api_budget(monkeypatch, usd_limit):
    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trial.py",
            "--config",
            str(ROOT / "gym/examples/claude-code.toml"),
            f"--max-budget-usd={usd_limit}",
        ],
    )
    with pytest.raises(SystemExit):
        runner["arguments"]()


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


def test_launcher_resolves_authorized_gpu_to_stable_uuid(monkeypatch):
    from nexttransformer_gym.gpu import configuration

    runner = runpy.run_path(str(ROOT / "gym/run_trial.py"))
    resolve = runner["resolve_gpu_policy"]
    gpu_uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    monkeypatch.setitem(
        resolve.__globals__,
        "discover",
        lambda: {"status": "ok", "gpus": [{"index": 3, "uuid": gpu_uuid}]},
    )
    assert resolve(configuration({"gpu_ids": [3]}))["gpu_ids"] == [gpu_uuid]


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
