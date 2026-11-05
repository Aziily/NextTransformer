"""The official MCP client must see the exact authenticated HTTP contract."""

import json
import os
import sys
import threading
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from nexttransformer_gym.client import GymClient
from nexttransformer_gym.common import GymError
from nexttransformer_gym.server import atomic_json, bind

from test_gym import context as make_context
from test_gym import rewrite_package


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    from nexttransformer_gym import gpu

    monkeypatch.setattr(gpu, "discover", lambda: {"status": "ok", "gpus": []})
    service, credentials, _, candidate = make_context.__wrapped__(tmp_path)
    server = bind(service, tmp_path / "listener")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    credential = tmp_path / "credential.json"
    atomic_json(credential, {"endpoint": endpoint, "token": credentials["agent_token"]})
    try:
        yield (
            service,
            credentials,
            candidate,
            GymClient(endpoint, credentials["agent_token"]),
            credential,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        server.service_lock.close()


@asynccontextmanager
async def connection(credential):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nexttransformer_gym.mcp_server", "--credential", str(credential)],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def payload(result):
    return json.loads(next(item.text for item in result.content if item.type == "text"))


def test_official_stdio_client_discovers_http_schemas_and_budget(endpoint):
    _, _, _, http, credential = endpoint

    async def check():
        async with connection(credential) as mcp:
            tools = {tool.name: tool for tool in (await mcp.list_tools()).tools}
            specs = http.describe_space()["operation_schemas"]
            assert tools.keys() == specs.keys()
            for name, spec in specs.items():
                assert tools[name].inputSchema == spec["inputSchema"]
                assert tools[name].description == spec["description"]
                assert tools[name].annotations.readOnlyHint == spec["readOnly"]
            for name in ("describe_space", "get_resources", "get_budget"):
                result = await mcp.call_tool(name, {})
                assert not result.isError
                actual, expected = payload(result), http.call(name)
                if name == "get_resources":
                    assert actual.pop("observed_at") <= expected.pop("observed_at")
                assert actual == expected

    anyio.run(check)


def test_http_new_model_config_and_static_alternatives(endpoint):
    from jsonschema import Draft202012Validator

    service, credentials, original, http, _ = endpoint
    space = http.describe_space()
    contract = space["candidate_contract"]
    schema = contract["model_config_schema"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    architecture = {
        "layers": 4,
        "mlp_width": 128,
        "window_pattern": "L",
        "options": {"activation": "silu", "output_scale": 1.0},
    }
    validator.validate(architecture)
    assert not validator.is_valid({"layers": True})
    assert not validator.is_valid({"optimizer": "different"})
    assert not validator.is_valid({"options": {"activation": {"expr": "x"}}})
    fixed = space["profiles"][0]["model_config_constraints"]["fixed"]
    assert fixed["width"] == 32 and fixed["scale"] == "debug"

    candidate = http.submit_candidate(
        rewrite_package({"architecture.json": json.dumps(architecture)}), "new-config"
    )
    assert candidate["architecture"] == architecture
    assert candidate["hash"] != original["hash"]
    assert http.get_candidate(candidate_id=original["id"]) == original
    # A config change preserves every plugin source byte.
    with service.store.transaction() as db:
        packages = [
            service.store.get(db, credentials["session_id"], "package", c["id"])[
                "files"
            ]
            for c in (original, candidate)
        ]
    for name in packages[0]:
        if name.startswith("plugin/"):
            assert packages[0][name] == packages[1][name]
    costs = [
        http.inspect_cost(candidate_id=c["id"], profile_id="research-fixture")
        for c in (original, candidate)
    ]
    assert costs[1]["parameters"] > costs[0]["parameters"]

    replacement = {"layers": 2, "mlp_width": 128}
    job = http.calibrate_candidate(
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        grid=[replacement],
        idempotency_key="replace-config",
    )
    result = http.get_result(job_id=job["id"])
    assert result["analysis"] == "static_only"
    derived = result["derived"][0]
    # This is a full replacement, including omission of the parent's options.
    assert derived["candidate"]["architecture"] == replacement
    assert derived["candidate"]["parent_ids"] == [candidate["id"]]
    assert derived["cost"]["parameters"] == costs[0]["parameters"]
    assert http.get_budget()["spent"] == "0"


def test_mcp_preserves_http_scope_and_rejects_execution_overrides(endpoint):
    service, _, candidate, http, credential = endpoint
    other = service.create_session(10**12, "other-mcp-session")
    with service.store.transaction() as db:
        private = service.store.put(db, other["session_id"], "job", {"state": "queued"})
    requests = [
        ("get_job", {"job_id": private["id"]}),
        (
            "run_experiment",
            {
                "quote_id": "unused",
                "idempotency_key": "injection",
                "command": "echo denied",
            },
        ),
        (
            "quote_experiment",
            {
                "candidate_id": candidate["id"],
                "profile_id": "research-fixture",
                "env": {"CUDA_VISIBLE_DEVICES": "all"},
            },
        ),
        ("launch_final_run", {}),
    ]
    original_budget = http.get_budget()

    async def check():
        async with connection(credential) as mcp:
            for operation, arguments in requests:
                with pytest.raises(GymError) as error:
                    http.call(operation, **arguments)
                result = await mcp.call_tool(operation, arguments)
                assert result.isError
                assert payload(result) == error.value.wire()
            assert payload(await mcp.call_tool("get_budget", {})) == original_budget

    anyio.run(check)


def test_mcp_and_http_share_one_budget_and_reject_over_budget_work(endpoint):
    service, credentials, candidate, http, credential = endpoint
    with service.store.transaction() as db:
        account = service.store.get(
            db, credentials["session_id"], "account", credentials["session_id"]
        )
        account["limit"] = "1"
        service.store.update(db, credentials["session_id"], "account", account)
    quote = http.quote_experiment(
        candidate_id=candidate["id"], profile_id="research-fixture", target_update=1
    )
    assert not quote["affordable"]
    arguments = {"quote_id": quote["id"], "idempotency_key": "over-budget"}

    async def check():
        async with connection(credential) as mcp:
            with pytest.raises(GymError) as error:
                http.run_experiment(**arguments)
            assert error.value.code == "insufficient_budget"
            denied = await mcp.call_tool("run_experiment", arguments)
            assert denied.isError and payload(denied) == error.value.wire()
            budget = payload(await mcp.call_tool("get_budget", {}))
            assert budget == http.get_budget()
            assert budget["spent"] == budget["reserved"] == "0"
            assert budget["available"] == "1"

    anyio.run(check)
