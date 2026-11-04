"""Prepare an isolated official snapshot without changing the source repository.

This performs static calibration only. It never starts training. Deployment
operators must provide their own separately authorized runtime/resources.
"""

import json
import subprocess
from pathlib import Path
from .common import file_hash, require
from .lowering import plugin_source


def prepare_snapshot(service, session_id, candidate_id, destination, resources):
    destination = Path(destination).resolve()
    require(
        not destination.exists(),
        "conflict",
        "Official snapshot destination must be new.",
    )
    require(
        not destination.is_relative_to(service.backend.platform_root),
        "denied_access",
        "Prepare official snapshots outside the source repository.",
    )
    with service.store.transaction() as db:
        candidate = service.store.get(db, session_id, "candidate", candidate_id)
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--",
            str(service.backend.platform_root),
            str(destination),
        ],
        check=True,
    )
    method = "gym_" + candidate["hash"][:24]
    plugin = destination / "nexttransformer" / "methods" / method
    plugin.mkdir()
    (plugin / "__init__.py").write_text(plugin_source(candidate["expression"]))
    (plugin / "method.json").write_text(
        json.dumps(
            {
                "name": method,
                "family": "others",
                "title": method,
                "options": {},
                "shell": {
                    "layers": candidate["architecture"]["layers"],
                    "mlp_width": 1536 * candidate["architecture"]["mlp_ratio"],
                    "window_pattern": "L",
                },
            }
        )
    )
    env = service.backend.environment()
    env["PYTHONPATH"] = str(destination)
    config = f"configs/others/{method}.json"
    # All dynamic source is produced from validated integers and a fixed method identity.
    bootstrap = """import sys
from pathlib import Path
from nexttransformer.models.config import candidate_config
from nexttransformer.protocol.spec import write_config
method, output, candidate_hash = sys.argv[1:]
write_config(Path(output), candidate_config(method), {
    "status": "gym_frozen_selection", "candidate_hash": candidate_hash,
})
"""
    subprocess.run(
        [service.backend.python, "-c", bootstrap, method, config, candidate["hash"]],
        cwd=destination,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            service.backend.python,
            "-m",
            "nexttransformer.protocol.calibrate",
            "--methods",
            method,
            "--reuse-configs",
            "configs",
            "--cache",
            ".agent-work/gym-calibration",
            "--output",
            "configs",
            "--results",
            "evidence/costs/gym-final",
        ],
        cwd=destination,
        env=env,
        check=True,
        timeout=600,
    )
    # Admission checks the unchanged selected architecture; calibration may not pick a replacement.
    admission = """import sys
from nexttransformer.protocol.admission import admit_config
config, proof = admit_config(sys.argv[1])
assert config.layers == int(sys.argv[2]) and config.mlp_width == int(sys.argv[3])
"""
    subprocess.run(
        [
            service.backend.python,
            "-c",
            admission,
            config,
            str(candidate["architecture"]["layers"]),
            str(1536 * candidate["architecture"]["mlp_ratio"]),
        ],
        cwd=destination,
        env=env,
        check=True,
        timeout=60,
    )
    # Commit only this disposable staging checkout, never the user's repository.
    subprocess.run(
        [
            "git",
            "add",
            "nexttransformer/methods/" + method,
            config,
            "evidence/costs/gym-final",
        ],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Gym Platform",
            "-c",
            "user.email=gym@example.invalid",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Freeze admitted Gym architecture",
        ],
        cwd=destination,
        check=True,
    )
    files = [
        p
        for folder in (
            "nexttransformer",
            "nanochat",
            "configs",
            "evidence/costs/gym-final",
        )
        for p in (destination / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json"}
    ]
    profile = {
        "width": 1536,
        "heads": 12,
        "sequence_len": 2048,
        "vocab_size": 32768,
        "batch_size": 1,
        "device": "cpu",
        "formal_dimensions": True,
        "timeout_sec": 600,
    }
    initialization = int(service.backend.inspect(candidate, profile)["initialization"])
    plan = {
        "candidate_hash": candidate["hash"],
        "snapshot": str(destination),
        "config": config,
        "resources": str(Path(resources).resolve()),
        "method": method,
        "initialization_flops": 2 * initialization,
        "manifest": {
            p.relative_to(destination).as_posix(): file_hash(p) for p in files
        },
    }
    (destination / "gym-final-plan.json").write_text(json.dumps(plan, indent=2))
    return plan
