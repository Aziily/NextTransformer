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
            for name in ("get_resources", "get_budget"):
                result = await mcp.call_tool(name, {})
                assert not result.isError
                actual, expected = payload(result), http.call(name)
                if name == "get_resources":
                    assert actual.pop("observed_at") <= expected.pop("observed_at")
                assert actual == expected

    anyio.run(check)


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
