"""Mediated researcher runtime: JSON actions only, no participant shell or eval."""

import io
import json
import time
import zipfile
from pathlib import Path
from .client import GymClient
from .common import GymError, canonical, require

SYSTEM = """You are an offline architecture researcher. Return exactly one JSON action.
Actions: {"action":"call","operation":<Gym operation>,"params":{...}},
{"action":"submit","files":{<package-relative filename>:<source text>},"key":<idempotency key>},
{"action":"note","text":<research note>}, or {"action":"finish","submission_id":<server ID>}.
No shell, Python execution, network URLs, dependencies, weights or self-reported scores are accepted.
All numerical work must use the Gym. Use wait_result with job_id for pending jobs.
Before finishing, validate against the selected final profile and finalize one candidate.
"""


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical(value))
    temporary.replace(path)


def dispatch(client, binding, action, workspace):
    require(isinstance(action, dict), "invalid_request", "Action must be an object.")
    kind = action.get("action")
    if kind == "call":
        require(
            set(action) == {"action", "operation", "params"},
            "invalid_request",
            "Invalid call action.",
        )
        operation, params = action["operation"], action["params"]
        require(
            isinstance(operation, str) and isinstance(params, dict),
            "invalid_request",
            "Invalid operation.",
        )
        if operation == "wait_result":
            require(
                set(params) == {"job_id"}, "invalid_request", "Only job_id is accepted."
            )
            return client.wait_result(params["job_id"]), False
        return client.call(operation, **params), False
    if kind == "submit":
        require(
            set(action) == {"action", "files", "key"}
            and isinstance(action["files"], dict),
            "invalid_request",
            "Invalid source action.",
        )
        files = action["files"]
        require(
            len(files) <= 32
            and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
            and sum(len(k.encode()) + len(v.encode()) for k, v in files.items())
            <= 1 << 20,
            "invalid_artifact",
            "Source package exceeds bounds.",
        )
        # No extraction or filesystem paths: source goes directly to the service.
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for name, source in files.items():
                archive.writestr(name, source)
        return client.submit_candidate(out.getvalue(), action["key"]), False
    if kind == "note":
        require(
            set(action) == {"action", "text"}
            and isinstance(action["text"], str)
            and len(action["text"]) <= 65536,
            "invalid_request",
            "Bounded text required.",
        )
        (workspace / "notes.md").write_text(action["text"])
        return {"saved": True}, False
    if kind == "finish":
        require(
            set(action) == {"action", "submission_id"},
            "invalid_request",
            "Final receipt requires a submission ID.",
        )
        submission = client.get_submission(submission_id=action["submission_id"])
        receipt = {
            "session_id": binding["session_id"],
            "submission_id": submission["id"],
            "candidate_hash": submission["candidate_hash"],
        }
        atomic(workspace / "submission.json", receipt)
        return receipt, True
    raise GymError("denied_access", "Unknown action; researcher execution is mediated.")


def main():
    workspace = Path("/workspace")
    transport = workspace / ".model"
    transport.mkdir(exist_ok=True)
    binding = json.loads(Path("/run/gym-auth/credential.json").read_text())
    client = GymClient(binding["endpoint"], binding["token"])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (workspace / "gym-instruction.txt").read_text()},
    ]
    for step in range(256):
        atomic(
            workspace / "agent-state.json",
            {"phase": "waiting_for_model", "step": step, "time": time.time()},
        )
        atomic(transport / "request.json", {"id": step, "messages": messages})
        deadline = time.monotonic() + 180
        while True:
            response_path = transport / "response.json"
            if response_path.exists():
                response = json.loads(response_path.read_text())
                if response["id"] == step:
                    break
            require(
                time.monotonic() < deadline,
                "infrastructure_failure",
                "Inference broker timed out.",
            )
            time.sleep(0.1)
        content = response["content"]
        require(
            isinstance(content, str) and len(content) <= 1 << 20,
            "invalid_request",
            "Model response exceeds bounds.",
        )
        atomic(
            workspace / "agent-state.json",
            {"phase": "executing_action", "step": step, "time": time.time()},
        )
        try:
            result, done = dispatch(client, binding, json.loads(content), workspace)
        except (GymError, ValueError, KeyError, TypeError) as error:
            result, done = (
                {
                    "error": error.code
                    if isinstance(error, GymError)
                    else "invalid_action"
                },
                False,
            )
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": canonical(result)},
        ]
        require(
            len(canonical(messages)) <= 4 << 20,
            "resource_unavailable",
            "Research conversation limit reached.",
        )
        if done:
            atomic(
                workspace / "agent-state.json",
                {"phase": "submitted", "step": step, "time": time.time()},
            )
            return
    raise RuntimeError("Research action limit reached without submission")


if __name__ == "__main__":
    main()
