"""Pinned worker subprocesses and a fenced, crash-conservative scheduler."""

import fcntl
import hashlib
import json
import importlib.metadata
import os
import queue
import threading
import subprocess
import sys
import time
import tempfile
import shutil
from pathlib import Path
from .common import GymError, canonical, identity, new_id, require, file_hash


class NativeBackend:
    def __init__(self, profiles, state_dir, platform_root, python=None, server_id=None):
        from .resources import local_server_id

        self.server_id = server_id or local_server_id()
        self.profiles = profiles
        self.state_dir, self.platform_root = (
            Path(state_dir).resolve(),
            Path(platform_root).resolve(),
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        for pid, profile in profiles.items():
            require(
                profile["id"] == pid and profile["kind"] in {"research", "final"},
                "invalid_request",
                "Invalid profile manifest.",
            )
            require(
                all(
                    type(profile[k]) is int and profile[k] > 0
                    for k in (
                        "width",
                        "heads",
                        "sequence_len",
                        "vocab_size",
                        "batch_size",
                        "max_updates",
                        "dev_batches",
                        "timeout_sec",
                    )
                ),
                "invalid_request",
                "Profile bounds must be explicit positive integers.",
            )
            require(
                profile["max_updates"] <= 46875,
                "invalid_request",
                "Schedule horizon cannot be extended.",
            )
        source_files = [
            p
            for root in (
                self.platform_root / "nexttransformer",
                self.platform_root / "nanochat",
                Path(__file__).parent,
            )
            for p in root.rglob("*.py")
        ]
        self.revision = identity(
            {
                str(p.relative_to(self.platform_root))
                if p.is_relative_to(self.platform_root)
                else "gym/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source_files)
            }
        )
        dependencies = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "pyarrow",
                "tiktoken",
                "rustbpe",
                "filelock",
                "fsspec",
                "sympy",
            )
        }
        self.revision = identity(
            {
                "source": self.revision,
                "dependencies": dependencies,
                "python": list(sys.version_info[:3]),
                "compute_dtype": "float32",
            }
        )
        self.cache = {}

    def public_profiles(self):
        for profile in self.profiles.values():
            devices = profile.get("cuda_visible_devices", [])
            public = {
                k: profile[k]
                for k in (
                    "id",
                    "kind",
                    "width",
                    "heads",
                    "sequence_len",
                    "vocab_size",
                    "max_updates",
                    "resource_profile_id",
                )
            }
            public.update(
                accelerator_class=profile.get("accelerator_class", "cpu"),
                usable_memory_bytes_per_device=profile.get("memory_limit_bytes"),
                supported_device_counts=[len(devices)] if devices else [1],
                supported_precisions=["bfloat16"]
                if profile.get("official_plan")
                else ["float32"],
                runtime_manifest_id=self.revision,
                checkpoint_quota_bytes=profile.get("checkpoint_quota_bytes"),
            )
            yield public

    def environment(self, profile=None, official=False):
        # Explicit environment prevents ambient plugin paths, credentials and proxies.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).parent.parent), str(self.platform_root)]
            ),
            "NANOCHAT_DTYPE": "float32",
            "TORCHDYNAMO_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "USER": "gym-worker",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(self.state_dir),
            "XDG_CACHE_HOME": str(self.state_dir / "cache"),
            "CUDA_VISIBLE_DEVICES": "",
        }
        if (
            not official
            and profile
            and str(profile.get("device", "cpu")).startswith("cuda")
        ):
            devices = profile.get("cuda_visible_devices", [])
            require(
                isinstance(devices, list) and len(devices) == 1,
                "resource_unavailable",
                "Research GPU execution requires one explicitly allocated device.",
            )
            env["CUDA_VISIBLE_DEVICES"] = str(devices[0])
        if official:
            require(
                profile is not None and profile.get("cuda_visible_devices"),
                "resource_unavailable",
                "Official GPU allocation must be explicit.",
            )
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(x) for x in profile["cuda_visible_devices"]
            )
            env["NANOCHAT_DTYPE"] = "bfloat16"
            env["TORCHDYNAMO_DISABLE"] = "0"
        return env

    def inspect(self, candidate, profile):
        key = identity([candidate["hash"], profile, self.revision])
        if key not in self.cache:
            request = {"mode": "inspect", "candidate": candidate, "profile": profile}
            try:
                result = subprocess.run(
                    self.worker_command(request),
                    input=canonical(request) + "\n",
                    text=True,
                    capture_output=True,
                    env=self.environment(),
                    cwd=self.state_dir,
                    timeout=profile["timeout_sec"],
                    check=True,
                    **self.worker_identity(request),
                )
                events = [
                    json.loads(line[len("GYM_EVENT ") :])
                    for line in result.stdout.splitlines()
                    if line.startswith("GYM_EVENT ")
                ]
                self.cache[key] = next(
                    e["value"] for e in events if e["kind"] == "inspection"
                )
            except (subprocess.SubprocessError, ValueError, StopIteration):
                raise GymError(
                    "incomplete_coverage",
                    "Trusted shape-only cost audit failed; inspect controller diagnostics.",
                ) from None
        return self.cache[key]

    def worker_command(self, request):
        if request.get("quote", {}).get("operation") == "official":
            # Official source is separately reviewed/admitted in an operator-owned snapshot.
            return [self.python, "-m", "nexttransformer_gym.worker"]
        from .isolation import command

        return command(self.python, self.platform_root, request)

    @staticmethod
    def worker_identity(request):
        # Docker controllers may be root for volume bootstrap. Namespace creation
        # runs as an unprivileged UID; the worker receives no controller capabilities.
        if (
            os.geteuid() == 0
            and request.get("quote", {}).get("operation") != "official"
        ):
            return {"user": 65534, "group": 65534, "extra_groups": []}
        return {}

    @staticmethod
    def bound(report, operation, updates, profile):
        init, fwd, bwd, opt = (
            int(report[k])
            for k in ("initialization", "forward", "backward", "optimizer")
        )
        if operation == "validate":
            return 2 * init + fwd + bwd + opt + 3 * int(report["prediction"])
        return (
            init
            + updates * (fwd + bwd + opt)
            + profile["dev_batches"] * int(report["evaluation"])
        )

    def checkpoint_path(self, checkpoint_id):
        require(
            isinstance(checkpoint_id, str)
            and len(checkpoint_id) == 32
            and all(c in "0123456789abcdef" for c in checkpoint_id),
            "invalid_request",
            "Invalid checkpoint identity.",
        )
        return self.state_dir / "checkpoints" / (checkpoint_id + ".pt")


class Scheduler:
    def __init__(self, service):
        self.service, self.backend, self.store = service, service.backend, service.store

    def recover(self):
        """Called only under the resource lock, after the old controller is gone."""
        with self.store.transaction() as db:
            rows = db.execute(
                "SELECT owner,value FROM objects WHERE kind='job'"
            ).fetchall()
            for sid, raw in rows:
                job = json.loads(raw)
                if job["state"] in {"running", "cancelling"}:
                    job.update(state="reconciliation", error_code="worker_lost")
                    self.store.update(db, sid, "job", job)

    def work_once(self):
        with (self.backend.state_dir / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            with self.store.transaction() as db:
                rows = db.execute(
                    "SELECT owner,value FROM objects WHERE kind='job' ORDER BY rowid"
                ).fetchall()
                selected = next(
                    (
                        (sid, json.loads(raw))
                        for sid, raw in rows
                        if json.loads(raw)["state"] == "queued"
                    ),
                    None,
                )
                if selected is None:
                    return False
                sid, job = selected
                # Uncertain work is a resource lease, not a free slot.
                if any(
                    json.loads(raw)["state"]
                    in {"running", "cancelling", "reconciliation"}
                    for _, raw in rows
                ):
                    return False
                job.update(
                    state="running", lease=new_id(), server_id=self.backend.server_id
                )
                self.store.update(db, sid, "job", job)
                quote = self.store.get(db, sid, "quote", job["quote_id"])
                if quote["checkpoint_id"]:
                    checkpoint = self.store.get(
                        db, sid, "checkpoint", quote["checkpoint_id"]
                    )
                    try:
                        valid_checkpoint = (
                            file_hash(self.backend.checkpoint_path(checkpoint["id"]))
                            == checkpoint["sha256"]
                        )
                    except OSError:
                        valid_checkpoint = False
                    if not valid_checkpoint:
                        job.update(
                            state="failed",
                            lease=None,
                            error_code="incompatible_checkpoint",
                        )
                        self.store.release(db, sid, job["id"])
                        self.store.update(db, sid, "job", job)
                        return True
                candidate = self.store.get(db, sid, "candidate", quote["candidate_id"])
                candidate["files"] = self.store.get(
                    db, sid, "package", candidate["id"]
                )["files"]
            try:
                self.execute(sid, job, quote, candidate)
            except Exception:
                with self.store.transaction() as db:
                    current = self.store.get(db, sid, "job", job["id"])
                    current.update(
                        state="reconciliation", error_code="infrastructure_failure"
                    )
                    self.store.update(db, sid, "job", current)
            return True

    def execute(self, sid, job, quote, candidate):
        profile = self.backend.profiles[quote["profile_id"]]
        output_id = new_id()
        workspace = tempfile.TemporaryDirectory(
            prefix="worker-", dir=self.backend.state_dir
        )
        output = Path(workspace.name) / "checkpoint.pt"
        if os.geteuid() == 0:
            self.backend.state_dir.chmod(self.backend.state_dir.stat().st_mode | 0o111)
            Path(workspace.name).chmod(0o777)
        request = {
            "mode": "execute",
            "candidate": candidate,
            "profile": profile,
            "quote": quote,
            "output": str(output),
            "platform_root": str(self.backend.platform_root),
            "final_output_root": str(self.backend.state_dir / "final-runs"),
            "checkpoint": str(self.backend.checkpoint_path(quote["checkpoint_id"]))
            if quote["checkpoint_id"]
            else None,
        }
        process = subprocess.Popen(
            self.backend.worker_command(request),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.backend.environment(profile, quote["operation"] == "official"),
            cwd=self.backend.state_dir,
            start_new_session=True,
            **self.backend.worker_identity(request),
        )
        pending, result, cancelled, failure = None, None, False, None
        # Drain stderr to a bounded private log without blocking stdout.
        messages = queue.Queue()

        def drain(stream, channel):
            for line in stream:
                messages.put((channel, line))
            messages.put((channel, None))

        readers = [
            threading.Thread(target=drain, args=(stream, channel), daemon=True)
            for stream, channel in (
                (process.stdout, "stdout"),
                (process.stderr, "stderr"),
            )
        ]
        for reader in readers:
            reader.start()
        diagnostics = []

        def send(value):
            process.stdin.write(canonical(value) + "\n")
            process.stdin.flush()

        send(request)
        deadline = time.monotonic() + profile["timeout_sec"]
        try:
            closed = set()
            while len(closed) < 2:
                require(
                    time.monotonic() < deadline,
                    "infrastructure_failure",
                    "Worker deadline exceeded.",
                )
                try:
                    channel, line = messages.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line is None:
                    closed.add(channel)
                    continue
                if channel == "stderr":
                    if sum(map(len, diagnostics)) < 65536:
                        diagnostics.append(line[:4096])
                    continue
                if not line.startswith("GYM_EVENT "):
                    continue
                event = json.loads(line[len("GYM_EVENT ") :])
                with self.store.transaction() as db:
                    current = self.store.get(db, sid, "job", job["id"])
                    require(
                        current["lease"] == job["lease"]
                        and current["state"] in {"running", "cancelling"},
                        "stale_lease",
                        "Worker no longer owns this job.",
                    )
                    kind = event["kind"]
                    if kind == "intent":
                        require(
                            pending is None,
                            "integrity_violation",
                            "Overlapping work intents.",
                        )
                        remaining = self.store.get(db, sid, "reservation", job["id"])[
                            "remaining"
                        ]
                        require(
                            0 <= int(event["amount"]) <= int(remaining),
                            "insufficient_budget",
                            "Operation exceeds reservation.",
                        )
                        if current["cancel_requested"]:
                            send({"cancel": True})
                        else:
                            pending = event
                            current["inflight"] = {
                                "label": event["label"],
                                "amount": event["amount"],
                            }
                            self.store.update(db, sid, "job", current)
                            # Acknowledgement follows durable intent commit below.
                    elif kind == "charge":
                        require(
                            pending is not None
                            and event["label"] == pending["label"]
                            and event["amount"] == pending["amount"],
                            "integrity_violation",
                            "Charge does not match in-flight operation.",
                        )
                        self.store.charge(
                            db,
                            sid,
                            job["id"],
                            identity([job["id"], event["label"]]),
                            int(event["amount"]),
                        )
                        pending = None
                        current.pop("inflight", None)
                        self.store.update(db, sid, "job", current)
                    elif kind == "progress":
                        current["progress"] = event["step"]
                        self.store.update(db, sid, "job", current)
                    elif kind == "result":
                        result = event["value"]
                    elif kind == "cancelled":
                        cancelled = True
                    elif kind == "failed":
                        require(
                            pending is None,
                            "integrity_violation",
                            "Failure has unsettled work.",
                        )
                        failure = event["error_code"]
                    else:
                        raise GymError("integrity_violation", "Unknown worker event.")
                if kind == "charge" or kind == "intent" and pending is not None:
                    send({"ack": event["label"]})
            returncode = process.wait(timeout=5)
            require(
                returncode == 0
                and pending is None
                and (result is not None or cancelled or failure is not None),
                "infrastructure_failure",
                "Worker ended without reconciled completion.",
            )
            with self.store.transaction() as db:
                current = self.store.get(db, sid, "job", job["id"])
                require(
                    current["lease"] == job["lease"], "stale_lease", "Lost job lease."
                )
                if not cancelled and failure is None:
                    result.update(id=job["id"], cost=self.store.budget(db, sid))
                    if quote["operation"] == "validate":
                        self.store.put(db, sid, "report", {**result, "id": new_id()})
                    elif quote["operation"] != "official":
                        require(
                            output.is_file()
                            and not output.is_symlink()
                            and output.stat().st_nlink == 1
                            and output.stat().st_size
                            <= job.get(
                                "checkpoint_reservation_bytes",
                                profile["checkpoint_quota_bytes"],
                            ),
                            "infrastructure_failure",
                            "Missing complete checkpoint.",
                        )
                        retained = self.backend.checkpoint_path(output_id)
                        retained.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(output, retained)
                        self.store.put(
                            db,
                            sid,
                            "checkpoint",
                            {
                                "id": output_id,
                                "complete": True,
                                "step": result["step"],
                                "candidate_hash": candidate["hash"],
                                "profile_hash": quote["profile_hash"],
                                "runtime_revision": self.backend.revision,
                                "size_bytes": output.stat().st_size,
                                "sha256": file_hash(output),
                            },
                        )
                        result["checkpoint_id"] = output_id
                    self.store.put(db, sid, "result", result)
                self.store.release(db, sid, job["id"])
                current.update(
                    state="cancelled"
                    if cancelled
                    else "budget_stopped"
                    if failure == "insufficient_budget"
                    else "failed"
                    if failure
                    else "succeeded",
                    lease=None,
                )
                if failure:
                    current.update(
                        error_code=failure, settlement="conservative_operation_bound"
                    )
                self.store.update(db, sid, "job", current)
        except Exception:
            if process.poll() is None:
                import signal

                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            with self.store.transaction() as db:
                current = self.store.get(db, sid, "job", job["id"])
                current.update(
                    state="reconciliation", error_code="infrastructure_failure"
                )
                self.store.update(db, sid, "job", current)
        finally:
            for reader in readers:
                reader.join(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            (self.backend.state_dir / (job["id"] + ".diagnostic.log")).write_text(
                "".join(diagnostics)
            )
            workspace.cleanup()
