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


def image_id(name):
    return subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", name],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-flops-limit", type=int, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--client-image", required=True)
    parser.add_argument("--service-image", required=True)
    parser.add_argument("--verifier-image", required=True)
    parser.add_argument("--agent", default="oracle")
    args = parser.parse_args()
    if args.research_flops_limit <= 0:
        parser.error("A positive finite research FLOPs allocation is required")
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
                )
            },
            indent=2,
        )
    )
    materialized = args.jobs_dir / (trial + "-task")
    shutil.copytree(task, materialized)
    manifest = materialized / "task.toml"
    source = manifest.read_text()
    for variable in ("GYM_CLIENT_IMAGE", "GYM_VERIFIER_IMAGE"):
        source = source.replace("${" + variable + "}", environment[variable])
    manifest.write_text(source)
    task = materialized
    harbor = Path(sys.executable).parent / "harbor"
    result = subprocess.run(
        [
            str(harbor),
            "run",
            "--path",
            str(task),
            "--agent",
            args.agent,
            "--jobs-dir",
            str(args.jobs_dir.resolve()),
            "--job-name",
            trial,
            "--n-concurrent",
            "1",
            "--n-attempts",
            "1",
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
