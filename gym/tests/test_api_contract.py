"""HTTP input validation and the complete public research lifecycle."""

import base64
import fcntl

import anyio
import pytest

from nexttransformer_gym.api import OPERATION_SCHEMAS
from nexttransformer_gym.backend import Scheduler
from nexttransformer_gym.common import GymError
from nexttransformer_gym.final import reconcile
from test_gym import candidate_bytes
from test_mcp import connection, endpoint as make_endpoint, payload


@pytest.fixture(name="endpoint")
def api_endpoint(tmp_path, monkeypatch):
    yield from make_endpoint.__wrapped__(tmp_path, monkeypatch)


def test_http_and_mcp_reject_malformed_parameters_before_dispatch(endpoint):
    _, _, candidate, http, credential = endpoint
    invalid = [
        ("get_candidate", {"candidate_id": []}),
        ("get_candidate", {"candidate_id": ""}),
        ("get_resources", {"resource_profile_id": []}),
        ("read_reference", {"resource_id": {}}),
        ("list_candidates", {"limit": True}),
        ("list_experiments", {"scope": ["published"]}),
        ("read_events", {"cursor": -1}),
        ("submit_candidate", {"package_base64": [], "idempotency_key": "bad"}),
        (
            "quote_experiment",
            {
                "candidate_id": candidate["id"],
                "profile_id": "research-fixture",
                "gpu_id": 0,
            },
        ),
    ]

    async def check():
        async with connection(credential) as mcp:
            for operation, params in invalid:
                with pytest.raises(GymError) as error:
                    http.call(operation, **params)
                assert error.value.code == "invalid_request"
                result = await mcp.call_tool(operation, params)
                assert result.isError
                assert payload(result) == error.value.wire()

    anyio.run(check)


def test_every_public_operation_in_http_research_lifecycle(endpoint):
    service, _, _, http, _ = endpoint
    used = set()

    def call(operation, **params):
        used.add(operation)
        return http.call(operation, **params)

    def finish(job):
        assert Scheduler(service).work_once()
        assert call("get_job", job_id=job["id"])["state"] == "succeeded"
        return call("get_result", job_id=job["id"])

    service.catalog = {"example-reference": "Native model architecture reference"}
    assert call("describe_space")["candidate_contract"]["model_config_schema"]
    assert call("get_resources")["server_count"] == 1
    reference = call("list_components")["items"][0]
    assert call("read_reference", resource_id=reference["id"])["content"]
    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(candidate_bytes()).decode(),
        idempotency_key="lifecycle-candidate",
    )
    assert call("get_candidate", candidate_id=candidate["id"]) == candidate
    assert candidate in call("list_candidates")["items"]
    research = {"candidate_id": candidate["id"], "profile_id": "research-fixture"}
    assert call("inspect_cost", **research)["parameters"] > 0
    calibration = call(
        "calibrate_candidate",
        **research,
        grid=[{"layers": 2, "mlp_width": 128}],
        idempotency_key="lifecycle-calibration",
    )
    assert call("get_result", job_id=calibration["id"])["analysis"] == "static_only"
    quote = call("quote_experiment", **research, target_update=1)
    cancelled = call(
        "run_experiment", quote_id=quote["id"], idempotency_key="lifecycle-cancel"
    )
    assert (
        call("cancel_job", job_id=cancelled["id"], idempotency_key="cancel")["state"]
        == "cancelled"
    )
    quote = call("quote_experiment", **research, target_update=1)
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="lifecycle-run")
    result = finish(job)
    assert result["step"] == 1 and result["checkpoint_id"]
    assert call("list_experiments", scope="session")["items"]
    assert (
        call("read_experiment", scope="session", experiment_id=job["id"])["result"]
        == result
    )
    diagnostic = finish(
        call(
            "debug_candidate",
            **research,
            check="construction",
            idempotency_key="lifecycle-debug",
        )
    )
    assert diagnostic["checks"] == {"construction": True}
    final = {"candidate_id": candidate["id"], "profile_id": "final-fixture"}
    assert finish(
        call("validate_candidate", **final, idempotency_key="lifecycle-validation")
    )["admitted"]
    submission = call("finalize", **final, idempotency_key="lifecycle-finalize")
    assert (
        call("get_submission", submission_id=submission["id"])["final_run_state"]
        == "not_scheduled"
    )
    assert call("read_events")["items"]
    budget = call("get_budget")
    assert int(budget["spent"]) > 0 and budget["reserved"] == "0"
    assert used == set(OPERATION_SCHEMAS)


@pytest.mark.parametrize("state", ["running", "cancelling", "reconciliation"])
def test_operator_recovers_persisted_crash_without_releasing_live_worker(
    endpoint, state
):
    service, credentials, candidate, http, _ = endpoint
    quote = http.quote_experiment(
        candidate_id=candidate["id"], profile_id="research-fixture"
    )
    job = http.run_experiment(quote_id=quote["id"], idempotency_key="crash")
    owner = credentials["session_id"]
    with service.store.transaction() as db:
        job.update(state=state, lease="old-controller")
        service.store.update(db, owner, "job", job)
    with pytest.raises(GymError):
        reconcile(service, owner, job["id"], worker_confirmed_stopped=False)
    with (service.backend.state_dir / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(GymError, match="scheduler still owns"):
            reconcile(service, owner, job["id"], worker_confirmed_stopped=True)
    settled = reconcile(service, owner, job["id"], worker_confirmed_stopped=True)
    assert settled["state"] == "failed"
    assert (
        reconcile(service, owner, job["id"], worker_confirmed_stopped=True) == settled
    )
    assert http.get_budget()["spent"] == quote["bound"]
    assert http.get_budget()["reserved"] == "0"
