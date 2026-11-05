"""Host-owned Gym lifecycle around an unmodified native Harbor job."""

import asyncio
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import tomllib
from types import SimpleNamespace
import uuid

import toml
import yaml

from .backend import Scheduler
from .cli import load
from .evidence import snapshot
from .server import atomic_json, bind


class HostGym:
    def __init__(
        self,
        service,
        directory,
        allocation,
        trial_id,
        bind_host,
        advertise_host="host.docker.internal",
    ):
        self.service = service
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        for name in ("agent-auth", "verifier-auth", "export", "state"):
            (self.directory / name).mkdir(exist_ok=True)
        self.server = bind(service, self.directory / "state", host=bind_host)
        # Drain authenticated requests before cancelling work at agent end.
        self.server.daemon_threads = False
        endpoint = f"http://{advertise_host}:{self.server.server_port}"
        binding = self.directory / "binding.json"
        try:
            if binding.exists():
                self.session = json.loads(binding.read_text())
                if (
                    self.session["trial_id"] != trial_id
                    or self.session["allocation"] != allocation
                ):
                    raise ValueError(
                        "Existing host session cannot be rebound or refunded"
                    )
                self.agent_credential = json.loads(
                    (self.directory / "agent-auth/credential.json").read_text()
                )
                self.verifier_credential = json.loads(
                    (self.directory / "verifier-auth/credential.json").read_text()
                )
                if self.agent_credential["endpoint"] != endpoint:
                    raise ValueError("Existing host service endpoint cannot be changed")
            else:
                created = service.create_session(allocation, trial_id)
                self.session = {
                    "session_id": created["session_id"],
                    "trial_id": trial_id,
                    "allocation": allocation,
                }
                for role in ("agent", "verifier"):
                    credential = {
                        "session_id": created["session_id"],
                        "token": created[role + "_token"],
                        "endpoint": endpoint,
                    }
                    setattr(self, role + "_credential", credential)
                    path = self.directory / (role + "-auth/credential.json")
                    atomic_json(path, credential)
                    # Parent is private on the host; only this role's directory
                    # is mounted into its container, where UIDs differ.
                    path.chmod(0o444)
                atomic_json(binding, self.session)
        except BaseException:
            self.server.server_close()
            self.server.service_lock.close()
            raise
        self.stop = threading.Event()
        self.scheduler = Scheduler(service)
        self.http_thread = None
        self.worker_thread = None
        self.failure = None

    def start(self):
        def schedule():
            try:
                while not self.stop.is_set():
                    if not self.scheduler.work_once():
                        self.stop.wait(0.2)
            except BaseException as error:
                self.failure = error
                self.stop.set()

        self.http_thread = threading.Thread(target=self.server.serve_forever)
        self.worker_thread = threading.Thread(target=schedule)
        self.http_thread.start()
        self.worker_thread.start()

    def stop_research(self):
        if self.http_thread is not None and self.http_thread.is_alive():
            self.server.shutdown()
            self.http_thread.join()
        self.server.server_close()
        self.stop.set()
        # Use the store transaction directly: finalization may already have
        # closed agent mutations, but operator cleanup must still settle jobs.
        sid = self.session["session_id"]
        with self.service.store.transaction() as db:
            for job in self.service.store.all(db, sid, "job"):
                if job["state"] == "queued":
                    job["state"] = "cancelled"
                    self.service.store.release(db, sid, job["id"])
                elif job["state"] in {"running", "cancelling"}:
                    job.update(state="cancelling", cancel_requested=True)
                self.service.store.update(db, sid, "job", job)
        # Scheduler enforces each worker deadline and kills/reaps its process
        # group on failure. Never exit while a research worker is still owned.
        if self.worker_thread is not None:
            self.worker_thread.join()

    def export_evidence(self):
        value = snapshot(self.service, self.verifier_credential)
        path = self.directory / "export/evidence.json"
        atomic_json(path, value)
        path.chmod(0o444)
        return value

    def close(self):
        try:
            self.stop_research()
        finally:
            self.server.server_close()
            self.server.service_lock.close()


def _mount(source, target):
    return {
        "type": "bind",
        "source": str(Path(source).resolve()),
        "target": target,
        "read_only": True,
        "bind": {"create_host_path": False},
    }


def materialize_task(
    source, destination, host_dir, client_image, verifier_image, timeout_sec
):
    source, destination, host_dir = Path(source), Path(destination), Path(host_dir)
    shutil.copytree(source, destination)
    manifest = tomllib.loads((destination / "task.toml").read_text())
    manifest["agent"]["timeout_sec"] = float(timeout_sec)
    manifest["environment"]["docker_image"] = client_image
    manifest["verifier"]["environment"]["docker_image"] = verifier_image
    (destination / "task.toml").write_text(toml.dumps(manifest))
    compose_path = destination / "environment/docker-compose.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    main = compose["services"]["main"]
    main["image"] = client_image
    main["volumes"] = [
        _mount(host_dir / "agent-auth", "/run/gym-auth"),
        _mount(host_dir / "export", "/export"),
    ]
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    verifier_path = destination / "tests/docker-compose.yaml"
    verifier = yaml.safe_load(verifier_path.read_text())["services"]["main"]
    verifier["image"] = verifier_image
    verifier["volumes"] = [_mount(host_dir / "verifier-auth", "/run/verifier-auth")]
    verifier_path.write_text(
        yaml.safe_dump({"services": {"main": verifier}}, sort_keys=False)
    )
    return destination


def docker_gateway():
    result = subprocess.run(
        ["docker", "network", "inspect", "bridge", "--format", "{{json .IPAM.Config}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    entries = json.loads(result.stdout)
    return next(
        entry["Gateway"] for entry in entries if ":" not in entry.get("Gateway", ":")
    )


async def run_job(args, trial_id, task, host):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    options = (
        {}
        if args.agent == "oracle"
        else {
            "version": args.agent_version,
            "max_turns": args.max_turns,
            "max_budget_usd": args.max_budget_usd,
        }
    )
    config = JobConfig.model_validate(
        {
            "job_name": trial_id,
            "jobs_dir": args.jobs_dir.resolve(),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "tasks": [{"path": task.resolve()}],
            "agents": [
                {"name": args.agent, "model_name": args.model, "kwargs": options}
            ],
        }
    )
    job = await Job.create(config)

    async def on_agent_ended(_event):
        await asyncio.to_thread(host.stop_research)
        host.export_evidence()

    job.on_agent_ended(on_agent_ended)
    return await job.run()


def run(args, image_id):
    root = Path(__file__).resolve().parents[3]
    # Running ``python gym/run_trial.py`` puts gym/, not the repository root,
    # on sys.path. Native plugin metadata is read by the host controller too.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    trial_id = "ntgym-" + uuid.uuid4().hex
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    directory = (args.jobs_dir / (trial_id + ".host")).resolve()
    directory.mkdir(mode=0o700)
    state = directory / "state"
    state.mkdir(mode=0o700)
    policy_path = directory / "execution.json"
    atomic_json(policy_path, args.execution_policy or {})
    service = load(
        SimpleNamespace(
            profiles=args.profiles or root / "gym/examples/profiles.fixture.json",
            state=state,
            platform=root,
            server_id=args.server_id,
            execution_config=policy_path,
            worker_python=args.worker_python,
        )
    )
    if (
        service.backend.execution_config["gpu_ids"]
        and not service.backend.worker_runtime["cuda_version"]
    ):
        raise SystemExit(
            "GPU research requires a CUDA-enabled worker interpreter; "
            "configure --worker-python before starting the Harbor agent."
        )
    host = HostGym(
        service,
        directory,
        args.research_flops_limit,
        trial_id,
        args.host_bind or docker_gateway(),
    )
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    try:
        images = {
            role: image_id(getattr(args, role + "_image"))
            for role in ("client", "verifier")
        }
        for role, image in images.items():
            subprocess.run(
                [
                    "docker",
                    "image",
                    "tag",
                    image,
                    f"nexttransformer-gym/{trial_id}:{role}",
                ],
                check=True,
            )
        task = materialize_task(
            root / "gym/tasks/architecture-research",
            args.jobs_dir / (trial_id + "-task"),
            directory,
            images["client"],
            images["verifier"],
            args.timeout_sec,
        )
        atomic_json(
            args.jobs_dir / (trial_id + ".binding.json"),
            {
                "GYM_TRIAL_ID": trial_id,
                "GYM_RESEARCH_FLOPS_LIMIT": str(args.research_flops_limit),
                "GYM_HOST_STATE": str(state),
                "GYM_HOST_DIRECTORY": str(directory),
                "GYM_CLIENT_IMAGE": images["client"],
                "GYM_VERIFIER_IMAGE": images["verifier"],
                "GYM_SERVER_ID": service.backend.server_id,
                "GYM_ENDPOINT": host.agent_credential["endpoint"],
            },
        )
        host.start()
        signal.signal(signal.SIGTERM, terminate)
        asyncio.run(run_job(args, trial_id, task, host))
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        host.close()
    results = list((args.jobs_dir / trial_id).glob("*/result.json"))
    failed = (
        host.failure
        or not results
        or any(json.loads(path.read_text()).get("exception_info") for path in results)
    )
    raise SystemExit(1 if failed else 0)
