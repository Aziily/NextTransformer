"""Launch one native Harbor trial with explicit allocation and isolated volumes.

Images must already exist locally. No implicit training authorization or builds.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import uuid
import tomllib
import hashlib
import socket
from decimal import Decimal, InvalidOperation

import yaml

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
    for role in ("client", "service", "verifier"):
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
    parser.add_argument("--max-budget-usd", default=agent.get("max_budget_usd", "5.00"))
    parser.add_argument(
        "--timeout-sec", type=int, default=runtime.get("timeout_sec", 1800)
    )
    parser.add_argument("--server-id", default=runtime.get("server_id"))
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=runtime.get("execution_config"),
        help="Operator-owned GPU execution policy; mounted read-only only in Gym",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.research_flops_limit or args.research_flops_limit <= 0:
        parser.error("A positive finite research FLOPs allocation is required")
    if args.max_turns <= 0 or args.timeout_sec <= 0:
        parser.error("Agent turn and time limits must be positive")
    try:
        amount = Decimal(str(args.max_budget_usd))
        if not amount.is_finite() or amount <= 0:
            raise ValueError
    except (InvalidOperation, ValueError):
        parser.error("Agent USD limit must be finite and positive")
    if args.agent == "claude-code" and not args.model:
        parser.error("Set --model or the configured model_env for Claude Code")
    if any(
        not getattr(args, role + "_image") for role in ("client", "service", "verifier")
    ):
        parser.error("All three prebuilt image names are required")
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
            command += ["--agent-kwarg", f"{name}={value}"]
    return command


def image_id(name):
    return subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", name],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def configure_execution(materialized, policy, jobs_dir, trial):
    """Freeze an operator policy and expose only its authorized devices to Gym."""
    if policy is None:
        return
    snapshot = (jobs_dir / (trial + ".execution.json")).resolve()
    snapshot.write_text(json.dumps(policy, indent=2) + "\n")
    compose_path = materialized / "environment/docker-compose.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    service = compose["services"]["gym"]
    service["volumes"].append(
        {
            "type": "bind",
            "source": str(snapshot),
            "target": "/run/gym-execution/config.json",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    )
    service["environment"]["GYM_EXECUTION_CONFIG"] = "/run/gym-execution/config.json"
    if policy["gpu_ids"]:
        device = {
            "driver": "nvidia",
            "device_ids": [str(gpu) for gpu in policy["gpu_ids"]],
            "capabilities": ["gpu"],
        }
        service["deploy"] = {"resources": {"reservations": {"devices": [device]}}}
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))


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


def check_gpu_runtime(args):
    if not args.execution_policy or not args.execution_policy["gpu_ids"]:
        return
    runtimes = json.loads(
        subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if "nvidia" not in runtimes:
        raise SystemExit(
            "GPU trials require the NVIDIA Container Toolkit Docker runtime."
        )
    args.execution_policy = resolve_gpu_policy(args.execution_policy)
    probe = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--entrypoint",
            "python",
            args.service_image,
            "-c",
            "import torch; assert torch.version.cuda, 'CUDA-enabled PyTorch is required'",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        raise SystemExit(
            "GPU trials require a prepared service image with CUDA-enabled PyTorch."
        )


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
    check_gpu_runtime(args)
    task = Path(__file__).resolve().parent / "tasks/architecture-research"
    trial = "ntgym-" + uuid.uuid4().hex
    environment = dict(
        os.environ,
        GYM_TRIAL_ID=trial,
        GYM_RESEARCH_FLOPS_LIMIT=str(args.research_flops_limit),
        GYM_STATE_VOLUME=trial + "-state",
        GYM_VERIFIER_VOLUME=trial + "-verifier",
        GYM_CLIENT_IMAGE=image_id(args.client_image),
        GYM_SERVICE_IMAGE=image_id(args.service_image),
        GYM_VERIFIER_IMAGE=image_id(args.verifier_image),
        GYM_SERVER_ID=args.server_id
        or "local-" + hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12],
    )
    # Keep explicit cohort tags as well as immutable IDs: replacing a build tag
    # must not leave the retained final runtime eligible for dangling-image GC.
    for role in ("CLIENT", "SERVICE", "VERIFIER"):
        subprocess.run(
            [
                "docker",
                "image",
                "tag",
                environment[f"GYM_{role}_IMAGE"],
                f"nexttransformer-gym/{trial}:{role.lower()}",
            ],
            check=True,
        )
    for name in (environment["GYM_STATE_VOLUME"], environment["GYM_VERIFIER_VOLUME"]):
        subprocess.run(
            [
                "docker",
                "volume",
                "create",
                "--label",
                "nexttransformer.gym.trial=" + trial,
                name,
            ],
            check=True,
            capture_output=True,
        )
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    # Durable operator receipt: no secrets; state is retained even on a failed trial.
    (args.jobs_dir / (trial + ".binding.json")).write_text(
        json.dumps(
            {
                k: environment[k]
                for k in (
                    "GYM_TRIAL_ID",
                    "GYM_RESEARCH_FLOPS_LIMIT",
                    "GYM_STATE_VOLUME",
                    "GYM_VERIFIER_VOLUME",
                    "GYM_CLIENT_IMAGE",
                    "GYM_SERVICE_IMAGE",
                    "GYM_VERIFIER_IMAGE",
                    "GYM_SERVER_ID",
                )
            },
            indent=2,
        )
    )
    materialized = args.jobs_dir / (trial + "-task")
    shutil.copytree(task, materialized)
    configure_execution(materialized, args.execution_policy, args.jobs_dir, trial)
    manifest = materialized / "task.toml"
    source = manifest.read_text()
    for variable in ("GYM_CLIENT_IMAGE", "GYM_VERIFIER_IMAGE"):
        source = source.replace("${" + variable + "}", environment[variable])
    source = source.replace(
        "timeout_sec = 1800.0", f"timeout_sec = {float(args.timeout_sec)}"
    )
    manifest.write_text(source)
    task = materialized
    harbor = Path(sys.executable).parent / "harbor"
    result = subprocess.run(
        [
            str(harbor),
            "run",
            "--path",
            str(task),
            *agent_arguments(args),
            "--jobs-dir",
            str(args.jobs_dir.resolve()),
            "--job-name",
            trial,
            "--n-concurrent",
            "1",
            "--n-attempts",
            "1",
            "--max-retries",
            "0",
            "--yes",
        ],
        env=environment,
    )
    result_files = list((args.jobs_dir / trial).glob("*/result.json"))
    failed = not result_files or any(
        json.loads(path.read_text()).get("exception_info") for path in result_files
    )
    raise SystemExit(result.returncode or (1 if failed else 0))


if __name__ == "__main__":
    main()
