"""Host-only controller actions; research credentials cannot call these remotely."""

import argparse
import json
from pathlib import Path
from .backend import NativeBackend
from .service import Service
from .server import atomic_json, serve
from .history import research_context


def load(args):
    profiles = json.loads(args.profiles.read_text())
    backend = NativeBackend(
        profiles, args.state, args.platform, server_id=getattr(args, "server_id", None)
    )
    sources, history = research_context(args.platform)
    references = {
        "native-plugin": "Use native method.json and ModelConfig architecture fields. Define mixer, ffn, block or a ReferenceLM model subclass. Local Python modules are supported.",
        "final-separation": "Finalization freezes one architecture. Verification does not launch training or award a performance score.",
    }
    for key, relative in {
        "reference-model": "nexttransformer/models/lm.py",
        "plugin-contract": "nexttransformer/models/plugins.py",
        "model-shell": "nexttransformer/models/backbone.py",
    }.items():
        source = (args.platform / relative).read_text()
        if len(source.encode()) <= 65536:
            references[key] = source
    return Service(
        args.state / "gym.sqlite3", backend, {**sources, **references}, history
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--platform", type=Path, required=True)
    parser.add_argument(
        "--server-id", help="Operator-assigned local execution node identity"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--container-port", type=int)
    server.add_argument("--host", default="127.0.0.1")
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--flops-limit", type=int, required=True)
    bootstrap.add_argument("--trial-id", required=True)
    bootstrap.add_argument("--agent-credential", type=Path, required=True)
    bootstrap.add_argument("--verifier-credential", type=Path, required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--credential", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("launch-final")
    final.add_argument("--session-id", required=True)
    final.add_argument("--submission-id", required=True)
    final.add_argument("--allocation", type=int, required=True)
    final.add_argument("--idempotency-key", required=True)
    prepare = commands.add_parser("prepare-final")
    prepare.add_argument("--session-id", required=True)
    prepare.add_argument("--candidate-id", required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--resources", type=Path, required=True)
    commands.add_parser("work-once")
    status = commands.add_parser("get-final")
    status.add_argument("--session-id", required=True)
    status.add_argument("--run-id", required=True)
    recovery = commands.add_parser("reconcile")
    recovery.add_argument("--session-id", required=True)
    recovery.add_argument("--job-id", required=True)
    recovery.add_argument(
        "--worker-confirmed-stopped", action="store_true", required=True
    )
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    service = load(args)
    if args.command == "bootstrap":
        binding = args.state / "binding.json"
        if binding.exists():
            old = json.loads(binding.read_text())
            if old["trial_id"] != args.trial_id or old["flops_limit"] != str(
                args.flops_limit
            ):
                parser.error("Existing trial allocation cannot be reset or rebound")
            return
        if args.agent_credential.exists() or args.verifier_credential.exists():
            parser.error("Credential destination already exists")
        session = service.create_session(args.flops_limit, args.trial_id)
        for path, role in (
            (args.agent_credential, "agent"),
            (args.verifier_credential, "verifier"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                path,
                {
                    "session_id": session["session_id"],
                    "token": session[role + "_token"],
                    "endpoint": "http://gym:8080",
                },
            )
            path.chmod(0o444)
        atomic_json(
            binding,
            {
                "session_id": session["session_id"],
                "trial_id": args.trial_id,
                "flops_limit": str(args.flops_limit),
            },
        )
    elif args.command == "snapshot":
        binding = json.loads(args.credential.read_text())
        with service.store.transaction() as db:
            sid = binding["session_id"]
            snapshot = {
                "session": service.session(db, sid),
                "budget": service.store.budget(db, sid),
                "jobs": service.store.all(db, sid, "job"),
                "submissions": service.store.all(db, sid, "submission"),
            }
        atomic_json(args.output, snapshot)
    elif args.command == "work-once":
        from .backend import Scheduler

        print(json.dumps({"dispatched": Scheduler(service).work_once()}))
    elif args.command == "get-final":
        from .final import get_final_run

        print(json.dumps(get_final_run(service, args.session_id, args.run_id)))
    elif args.command == "reconcile":
        from .final import reconcile

        print(
            json.dumps(
                reconcile(
                    service,
                    args.session_id,
                    args.job_id,
                    worker_confirmed_stopped=args.worker_confirmed_stopped,
                )
            )
        )
    elif args.command == "prepare-final":
        from .prepare import prepare_snapshot

        prepare_snapshot(
            service,
            args.session_id,
            args.candidate_id,
            args.destination,
            args.resources,
        )
        print(
            json.dumps(
                {
                    "plan": str(args.destination / "gym-final-plan.json"),
                    "training_started": False,
                }
            )
        )
    elif args.command == "launch-final":
        from .final import launch_final_run

        print(
            json.dumps(
                launch_final_run(
                    service,
                    args.session_id,
                    args.submission_id,
                    args.allocation,
                    args.idempotency_key,
                )
            )
        )
    else:
        serve(service, args.state, args.host, args.container_port)


if __name__ == "__main__":
    main()
