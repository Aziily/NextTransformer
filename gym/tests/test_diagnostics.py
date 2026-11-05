"""Named diagnostics execute and charge only their required numerical work."""

import base64
import copy
import io
from contextlib import contextmanager
from pathlib import Path
import sys
import zipfile

import pytest

from nexttransformer_gym import gpu, isolation
from nexttransformer_gym.backend import NativeBackend, Scheduler
from nexttransformer_gym.common import GymError

from test_gym import ROOT, candidate_bytes, context as make_context


@pytest.fixture(name="context")
def diagnostic_session(tmp_path):
    return make_context.__wrapped__(tmp_path)


def complete(service, call, job):
    assert Scheduler(service).work_once()
    state = call("get_job", job_id=job["id"])
    assert state["state"] == "succeeded", state
    return call("get_result", job_id=job["id"])


def test_named_diagnostics_have_distinct_exact_bounds_and_cannot_admit(context):
    service, credentials, call, candidate = context
    parameters = {"candidate_id": candidate["id"], "profile_id": "final-fixture"}
    report = call("inspect_cost", **parameters)
    init, forward, backward, optimizer, prediction = (
        int(report[key])
        for key in ("initialization", "forward", "backward", "optimizer", "prediction")
    )
    expected = {
        "construction": init,
        "cost_coverage": init,
        "numerical": init + forward + backward + optimizer,
        "causal": init + 2 * prediction,
        "checkpoint": 2 * init + forward + backward + optimizer + 2 * prediction,
    }
    for check, amount in expected.items():
        before = int(call("get_budget")["spent"])
        job = call(
            "debug_candidate",
            **parameters,
            check=check,
            idempotency_key="debug-" + check,
        )
        assert job["operation"] == "debug"
        assert int(call("get_budget")["reserved"]) == amount
        result = complete(service, call, job)
        assert result["check"] == check
        assert result["checks"] == {check: True}
        assert result["admitted"] is False
        assert "checkpoint_id" not in result
        budget = call("get_budget")
        assert int(budget["spent"]) - before == amount
        assert budget["reserved"] == "0"
    with service.store.transaction() as db:
        assert service.store.all(db, credentials["session_id"], "report") == []
        assert service.store.all(db, credentials["session_id"], "checkpoint") == []
    with pytest.raises(GymError) as denied:
        call("finalize", **parameters, idempotency_key="debug-cannot-admit")
    assert denied.value.code == "missing_evidence"
    admission = complete(
        service,
        call,
        call("validate_candidate", **parameters, idempotency_key="full-validation"),
    )
    assert admission["admitted"] is True
    assert set(admission["checks"]) == set(expected)
    assert (
        call("finalize", **parameters, idempotency_key="full-admission")["candidate_id"]
        == candidate["id"]
    )


def test_causal_diagnostic_does_not_run_unrequested_training_checks(context):
    service, _, call, _ = context
    with zipfile.ZipFile(io.BytesIO(candidate_bytes())) as original:
        files = {name: original.read(name) for name in original.namelist()}
    name = "plugin/research_model/model.py"
    files[name] = files[name].replace(
        b'self.config.option("output_scale")',
        b'self.config.option("output_scale") * (float("nan") if self.training else 1.0)',
    )
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(blob.getvalue()).decode(),
        idempotency_key="nonfinite-training",
    )
    parameters = {"candidate_id": candidate["id"], "profile_id": "research-fixture"}
    causal = complete(
        service,
        call,
        call(
            "debug_candidate",
            **parameters,
            check="causal",
            idempotency_key="causal-only",
        ),
    )
    assert causal["checks"] == {"causal": True}
    numerical = complete(
        service,
        call,
        call(
            "debug_candidate",
            **parameters,
            check="numerical",
            idempotency_key="numerical-only",
        ),
    )
    assert numerical["checks"] == {"numerical": False}
    assert numerical["admitted"] is False
    assert call("get_budget")["reserved"] == "0"


@pytest.mark.parametrize("operation", ["validate_candidate", "debug_candidate"])
def test_official_final_diagnostics_use_only_the_bounded_gpu_pool(
    context, monkeypatch, operation
):
    service, _, call, candidate = context
    selected = "GPU-00000000-0000-0000-0000-000000000001"
    other = "GPU-00000000-0000-0000-0000-000000000002"
    profiles = copy.deepcopy(service.backend.profiles)
    final = profiles["final-fixture"]
    final.update(
        official_plan={"operator_owned": "unchanged-plan"},
        cuda_visible_devices=[str(i) for i in range(8)],
        device="cuda:7",
        memory_limit_bytes=1024**3,
    )
    original = copy.deepcopy(final)
    policy = {"gpu_ids": [selected], "gpu_memory_limit_bytes": 2 * 1024**3}
    backend = NativeBackend(
        profiles, service.backend.state_dir, ROOT, execution_config=policy
    )
    service.backend = backend
    monkeypatch.setattr(
        gpu,
        "discover",
        lambda: {"status": "ok", "gpus": [{"id": selected, "index": 3}]},
    )
    assert backend.validate_placement(final, selected) == selected
    with pytest.raises(GymError):
        backend.validate_placement(final, other)
    with pytest.raises(GymError):
        backend.validate_placement(final, selected, "another-node")
    public = next(
        item for item in backend.public_profiles() if item["id"] == "final-fixture"
    )
    assert public["supported_device_counts"] == [1]
    assert public["supported_precisions"] == ["float32"]
    assert public["usable_memory_bytes_per_device"] == 1024**3
    assert public["operator_final_training"]["device_count"] == 8
    parameters = {"candidate_id": candidate["id"], "profile_id": "final-fixture"}
    with pytest.raises(GymError) as denied:
        call("quote_experiment", **parameters)
    assert denied.value.code == "denied_access"
    with pytest.raises(GymError):
        call("launch_final_run", **parameters)
    job = call(
        operation,
        **parameters,
        **({"check": "construction"} if operation == "debug_candidate" else {}),
        idempotency_key="bounded-final",
    )
    recorded = {}

    @contextmanager
    def reserve(allocation, requested):
        recorded["allocation"] = allocation
        recorded["requested"] = requested
        yield {
            "id": selected,
            "memory_limit_bytes": allocation["gpu_memory_limit_bytes"],
        }

    def execute(_scheduler, _sid, assigned, quote, _candidate):
        recorded["job"] = assigned
        recorded["quote"] = quote

    monkeypatch.setattr(gpu, "reserve", reserve)
    monkeypatch.setattr(Scheduler, "execute", execute)
    backend.worker_runtime["cuda_version"] = "fixture-cuda"
    assert Scheduler(service).work_once()
    assert recorded["allocation"]["gpu_ids"] == [selected]
    assert recorded["allocation"]["gpu_memory_limit_bytes"] == 1024**3
    assert recorded["quote"]["operation"] in {"validate", "debug"}
    assert recorded["job"]["id"] == job["id"]
    assert recorded["job"]["assigned_gpu"]["id"] == selected
    assert backend.bounded_profile(final)["device"] == "cuda:0"
    assert backend.profiles["final-fixture"] == original
    assert profiles["final-fixture"] == original
    backend.execution_config["gpu_ids"] = []
    with pytest.raises(GymError) as unallocated:
        backend.validate_placement(final)
    assert unallocated.value.code == "resource_unavailable"


def test_bounded_quote_rejects_model_above_operator_memory_limit(context):
    service, _, call, candidate = context
    profiles = copy.deepcopy(service.backend.profiles)
    profiles["final-fixture"].update(
        official_plan={"operator_owned": "plan"}, cuda_visible_devices=list(range(8))
    )
    service.backend = NativeBackend(
        profiles,
        service.backend.state_dir,
        ROOT,
        execution_config={"gpu_ids": [0], "gpu_memory_limit_bytes": 1},
    )
    with pytest.raises(GymError) as error:
        call(
            "validate_candidate",
            candidate_id=candidate["id"],
            profile_id="final-fixture",
            idempotency_key="too-small-allocation",
        )
    assert error.value.code == "resource_unavailable"
    assert call("get_budget")["reserved"] == "0"


@pytest.mark.parametrize("mode", ["execute", "inspect"])
def test_real_data_token_bytes_outside_tokenizer_are_mounted_only_for_execution(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(isolation.shutil, "which", lambda _name: "/usr/bin/bwrap")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    table = tmp_path / "metadata" / "token_bytes.pt"
    table.parent.mkdir()
    table.write_bytes(b"fixture-token-table")
    train, validation = tmp_path / "train.parquet", tmp_path / "validation.parquet"
    train.touch()
    validation.touch()
    data = {
        "kind": "climbmix",
        "train_paths": [str(train)],
        "dev_path": str(validation),
        "tokenizer": str(tokenizer),
        "token_bytes": str(table),
    }
    command = isolation.command(
        sys.executable, ROOT, {"mode": mode, "profile": {"device": "cpu", "data": data}}
    )
    mounts = [
        command[index + 1 : index + 3]
        for index, item in enumerate(command)
        if item == "--ro-bind"
    ]
    expected = [str(table.resolve()), str(table.resolve())]
    assert (expected in mounts) == (mode == "execute")
    assert all(
        command[index + 1] != str(table)
        for index, item in enumerate(command)
        if item == "--bind"
    )
    assert not Path(table).is_relative_to(tokenizer)
