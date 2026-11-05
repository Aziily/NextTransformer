"""Launch native Harbor containers backed by a host-owned Gym service.

Images must already exist locally. No implicit training authorization or builds.
"""

import argparse
import json
import os
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path

import tomllib
from nexttransformer_gym.common import GymError
from nexttransformer_gym.gpu import configuration, discover


def arguments():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    first, _ = bootstrap.parse_known_args()
    config = tomllib.loads(first.config.read_text()) if first.config else {}
    agent, budget, runtime, images = (
        config.get(k, {}) for k in ("agent", "budget", "runtime", "images")
    )
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument(
        "--research-flops-limit", type=int, default=budget.get("research_flops_limit")
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=runtime.get("jobs_dir", ".agent-work/harbor-gym"),
    )
    for role in ("client", "verifier"):
        parser.add_argument(f"--{role}-image", default=images.get(role))
    parser.add_argument(
        "--agent",
        choices=("claude-code", "oracle"),
        default=agent.get("name", "claude-code"),
    )
    parser.add_argument(
        "--model",
        default=agent.get("model")
        or os.environ.get(agent.get("model_env", "GYM_RESEARCH_MODEL")),
    )
    parser.add_argument("--agent-version", default=agent.get("version", "2.1.312"))
    parser.add_argument("--max-turns", type=int, default=agent.get("max_turns", 64))
    parser.add_argument(
        "--max-budget-usd",
        default=agent.get("max_budget_usd"),
        help="Optional Claude Code API cost limit in USD",
    )
    parser.add_argument(
        "--timeout-sec", type=int, default=runtime.get("timeout_sec", 1800)
    )
    parser.add_argument("--server-id", default=runtime.get("server_id"))
    parser.add_argument("--host-bind", default=runtime.get("host_bind"))
    parser.add_argument("--worker-python", default=runtime.get("worker_python"))
    parser.add_argument("--profiles", type=Path, default=runtime.get("profiles"))
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=runtime.get("execution_config"),
        help="Operator-owned GPU execution policy used only by the Gym controller",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.research_flops_limit or args.research_flops_limit <= 0:
        parser.error("A positive finite research FLOPs allocation is required")
    if args.max_turns <= 0 or args.timeout_sec <= 0:
        parser.error("Agent turn and time limits must be positive")
    if args.max_budget_usd is not None:
        try:
            amount = Decimal(str(args.max_budget_usd))
            if not amount.is_finite() or amount <= 0:
                raise ValueError
        except (InvalidOperation, ValueError):
            parser.error("Agent USD limit must be finite and positive")
    if args.agent == "claude-code" and not args.model:
        parser.error("Set --model or the configured model_env for Claude Code")
    roles = ("client", "verifier")
    if any(not getattr(args, role + "_image") for role in roles):
        parser.error("Prebuilt images are required for: " + ", ".join(roles))
    try:
        args.execution_policy = (
            configuration(json.loads(args.execution_config.read_text()))
            if args.execution_config
            else None
        )
    except (OSError, ValueError, TypeError, GymError) as error:
        parser.error(f"Invalid execution configuration: {error}")
    return args


def agent_arguments(args):
    command = ["--agent", args.agent]
    if args.agent == "claude-code":
        command += ["--model", args.model]
        for name, value in (
            ("version", args.agent_version),
            ("max_turns", args.max_turns),
            ("max_budget_usd", args.max_budget_usd),
        ):
            if value is not None:
                command += ["--agent-kwarg", f"{name}={value}"]
    return command


def image_id(name):
    return subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", name],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def resolve_gpu_policy(policy):
    """Bind host selectors to stable UUIDs before container device reindexing."""
    observation = discover()
    if observation["status"] != "ok":
        raise SystemExit("Cannot resolve authorized GPUs: host discovery unavailable.")
    cards = observation["gpus"]
    identifiers = []
    for selector in policy["gpu_ids"]:
        matches = [
            card
            for card in cards
            if (selector.isdecimal() and card["index"] == int(selector))
            or card["uuid"] == selector
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"Authorized GPU {selector} is not uniquely present on the host."
            )
        identifiers.append(matches[0]["uuid"])
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("Multiple GPU selectors resolve to the same physical device.")
    return configuration({**policy, "gpu_ids": identifiers})


def main():
    args = arguments()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "agent_arguments": agent_arguments(args),
                    "research_flops_limit": str(args.research_flops_limit),
                    "timeout_sec": args.timeout_sec,
                    "execution_backend": "local_subprocess",
                    "gym_backend": "host",
                    "worker_python": args.worker_python,
                    "execution_config": (
                        str(args.execution_config) if args.execution_config else None
                    ),
                    "execution_policy": args.execution_policy,
                    "mcp_transport": "stdio",
                },
                indent=2,
            )
        )
        return
    from nexttransformer_gym.host_trial import run

    if args.execution_policy and args.execution_policy["gpu_ids"]:
        args.execution_policy = resolve_gpu_policy(args.execution_policy)
    return run(args, image_id)


if __name__ == "__main__":
    main()
