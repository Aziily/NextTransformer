"""Run native Harbor Claude Code against a deterministic local model stub.

This tests real CLI/tool/container/API execution, not model research quality.
No external model API is called. All numerical work uses bounded CPU fixtures.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent


def model_server(state, token):
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / "listener.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = state / "listener.json"
    port = json.loads(config.read_text())["port"] if config.exists() else 0
    script = (ROOT / "tasks/architecture-research/solution/research.py").read_text()
    calls = {
        "messages": 0,
        "research_commands": 0,
        "mcp_resource_calls": 0,
        "mcp_resource_success": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.headers.get("x-api-key") != token:
                self.send_error(403)
                return
            self.connection.settimeout(30)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8 * 1024 * 1024:
                self.send_error(413)
                return
            data = json.loads(self.rfile.read(size))
            if self.path.split("?")[0].endswith("/count_tokens"):
                self.reply({"input_tokens": 128})
                return
            if self.path.split("?")[0] != "/v1/messages":
                self.send_error(404)
                return
            calls["messages"] += 1
            tools = {t["name"] for t in data.get("tools", [])}
            results = {
                block["tool_use_id"]: block
                for message in data.get("messages", [])
                if isinstance(message.get("content"), list)
                for block in message["content"]
                if isinstance(block, dict) and block.get("type") == "tool_result"
            }
            completed = "toolu_gym_smoke" in results
            resource_result = results.get("toolu_gym_resources")
            if resource_result is not None:
                calls["mcp_resource_success"] = not resource_result.get(
                    "is_error", False
                )
            resource_tool = next(
                (name for name in tools if name == "mcp__gym__get_resources"), None
            )
            if resource_tool and resource_result is None:
                block = {
                    "type": "tool_use",
                    "id": "toolu_gym_resources",
                    "name": resource_tool,
                    "input": {},
                }
                stop = "tool_use"
                calls["mcp_resource_calls"] += 1
            elif (
                not resource_tool
                and resource_result is None
                and "ToolSearch" in tools
                and "toolu_gym_search" not in results
            ):
                block = {
                    "type": "tool_use",
                    "id": "toolu_gym_search",
                    "name": "ToolSearch",
                    "input": {"query": "+gym get_resources", "max_results": 1},
                }
                stop = "tool_use"
            elif "Bash" in tools and calls["mcp_resource_success"] and not completed:
                block = {
                    "type": "tool_use",
                    "id": "toolu_gym_smoke",
                    "name": "Bash",
                    "input": {
                        "command": "python - <<'GYM_SMOKE'\n" + script + "\nGYM_SMOKE",
                        "description": "Run the bounded Gym integration workflow",
                        "timeout": 120000,
                    },
                }
                stop = "tool_use"
                calls["research_commands"] += 1
            else:
                block, stop = (
                    {"type": "text", "text": "Bounded Gym smoke complete."},
                    "end_turn",
                )
            message = {
                "id": "msg_gym_smoke",
                "type": "message",
                "role": "assistant",
                "model": data.get("model", "smoke"),
                "content": [block],
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 128, "output_tokens": 128},
            }
            if not data.get("stream"):
                self.reply(message)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.event(
                "message_start",
                {"message": {**message, "content": [], "stop_reason": None}},
            )
            if block["type"] == "tool_use":
                start = {**block, "input": {}}
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                }
            else:
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            self.event("content_block_start", {"index": 0, "content_block": start})
            self.event("content_block_delta", {"index": 0, "delta": delta})
            self.event("content_block_stop", {"index": 0})
            self.event(
                "message_delta",
                {
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": 128},
                },
            )
            self.event("message_stop", {})

        def reply(self, value):
            payload = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def event(self, kind, value):
            self.wfile.write(
                (
                    "event: "
                    + kind
                    + "\ndata: "
                    + json.dumps({"type": kind, **value})
                    + "\n\n"
                ).encode()
            )
            self.wfile.flush()

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        server.daemon_threads = True
        temporary = config.with_suffix(".tmp")
        temporary.write_text(json.dumps({"port": server.server_port}))
        temporary.replace(config)
    except BaseException:
        lock.close()
        raise
    return server, lock, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, default=ROOT.parent / ".agent-work/claude-gym-smoke"
    )
    parser.add_argument("--client-image", default="nt-gym-client:local")
    parser.add_argument("--verifier-image", default="nt-gym-verifier:local")
    args = parser.parse_args()
    token = secrets.token_urlsafe(32)
    server, lock, calls = model_server(args.state, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    jobs = args.state / "jobs"
    before = set(jobs.glob("*.binding.json"))
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "CLAUDE_", "AWS_"))
    }
    environment.update(
        ANTHROPIC_API_KEY=token,
        ANTHROPIC_BASE_URL=f"http://host.docker.internal:{server.server_port}",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_trial.py"),
                "--agent",
                "claude-code",
                "--model",
                "claude-sonnet-4-5",
                "--max-turns",
                "6",
                "--max-budget-usd",
                "1",
                "--timeout-sec",
                "300",
                "--research-flops-limit",
                "1000000000000",
                "--jobs-dir",
                str(jobs),
                "--client-image",
                args.client_image,
                "--verifier-image",
                args.verifier_image,
            ],
            env=environment,
            check=True,
        )
        bindings = set(jobs.glob("*.binding.json")) - before
        assert len(bindings) == 1, "Expected one fresh trial binding"
        binding = json.loads(bindings.pop().read_text())
        result_files = list((jobs / binding["GYM_TRIAL_ID"]).glob("*/result.json"))
        assert len(result_files) == 1, "Expected exactly one trial result"
        result = json.loads(result_files[0].read_text())
        assert not result.get("exception_info"), result.get("exception_info")
        assert result["verifier_result"]["rewards"]["reward"] == 1, (
            "Submission verification failed"
        )
        assert calls["research_commands"] >= 1, (
            "Native Claude Code never requested the research tool"
        )
        assert calls["mcp_resource_calls"] >= 1 and calls["mcp_resource_success"], (
            "Native Claude Code did not complete a Gym MCP resource call"
        )
        summary = {
            "trial_id": binding["GYM_TRIAL_ID"],
            "agent": "claude-code",
            "model_backend": "local_stub",
            "external_model_calls": 0,
            "reward": 1,
            **calls,
        }
        (args.state / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        lock.close()


if __name__ == "__main__":
    main()
