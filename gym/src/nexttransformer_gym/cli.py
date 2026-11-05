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
        profiles,
        args.state,
        args.platform,
        server_id=getattr(args, "server_id", None),
        python=getattr(args, "worker_python", None),
        execution_config=json.loads(args.execution_config.read_text())
        if getattr(args, "execution_config", None)
        else None,
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
    parser.add_argument(
        "--execution-config", type=Path, help="Operator GPU allocation JSON"
    )
    parser.add_argument(
        "--worker-python",
        type=Path,
        help="Numerical worker Python with pinned dependencies",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
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
    if args.command == "snapshot":
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
        serve(service, args.state, args.host)


if __name__ == "__main__":
    main()
