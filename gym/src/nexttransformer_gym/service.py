"""Transport-independent policy. No participant code is imported by this service."""

import base64
import hashlib
import io
import json
import secrets
import time
import zipfile
from .common import GymError, canonical, identity, integer, new_id, require
from .package import CONTRACT, ingest
from .store import Store
from .api import MUTATIONS, READS, OPERATION_SCHEMAS, validate_parameters

TERMINAL = {"succeeded", "failed", "cancelled", "budget_stopped"}
REQUIRED_CHECKS = {"construction", "numerical", "causal", "checkpoint", "cost_coverage"}


class Service:
    def __init__(self, database, backend, catalog=None, history=None):
        self.store, self.backend = Store(database), backend
        self.catalog = catalog or {}
        self.history = history or {}

    def create_session(self, research_flops_limit, trial_id, max_concurrent=1):
        integer(research_flops_limit, 1)
        integer(max_concurrent, 1, 1)
        require(
            isinstance(trial_id, str) and bool(trial_id),
            "invalid_request",
            "A trusted trial binding is required.",
        )
        token, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        sid = new_id()
        with self.store.transaction() as db:
            self.store.put(
                db,
                sid,
                "session",
                {
                    "id": sid,
                    "phase": "researching",
                    "trial_id": trial_id,
                    "max_concurrent": max_concurrent,
                    "final_choice": None,
                    "integrity_violation": False,
                },
            )
            self.store.put(
                db, sid, "account", {"id": sid, "limit": str(research_flops_limit)}
            )
            for role, key in (("agent", token), ("verifier", verifier)):
                self.store.put(
                    db,
                    "control",
                    "credential",
                    {
                        "id": hashlib.sha256(key.encode()).hexdigest(),
                        "session_id": sid,
                        "role": role,
                    },
                )
        return {"session_id": sid, "agent_token": token, "verifier_token": verifier}

    def authenticate(self, db, token):
        try:
            return self.store.get(
                db, "control", "credential", hashlib.sha256(token.encode()).hexdigest()
            )
        except GymError:
            raise GymError("denied_access", "Invalid session credential.") from None

    def call(self, token, operation, params):
        require(
            isinstance(operation, str) and isinstance(params, dict),
            "invalid_request",
            "Invalid operation envelope.",
        )
        with self.store.transaction() as db:
            principal = self.authenticate(db, token)
            sid, role = principal["session_id"], principal["role"]
            if role == "verifier":
                require(
                    operation == "verify_submission",
                    "denied_access",
                    "Verifier can only inspect its bound final submission.",
                )
                return self.verify(db, sid, params)
            require(
                operation in READS | MUTATIONS,
                "denied_access",
                "Operation is outside the agent API.",
            )
            validate_parameters(operation, params)
            try:
                request = canonical([operation, params])
            except (ValueError, TypeError):
                raise GymError(
                    "invalid_request", "Parameters must contain finite JSON values."
                ) from None
            key = params.get("idempotency_key")
            if operation in MUTATIONS:
                require(
                    isinstance(key, str) and 0 < len(key) <= 128,
                    "invalid_request",
                    "A 1–128 character idempotency key is required.",
                )
                old = db.execute(
                    "SELECT request,response FROM retries WHERE owner=? AND key=?",
                    (sid, key),
                ).fetchone()
                if old:
                    require(
                        old[0] == request,
                        "conflict",
                        "Key is already bound to a different request.",
                    )
                    return json.loads(old[1])
            result = self.dispatch(db, sid, operation, params)
            if operation in MUTATIONS:
                db.execute(
                    "INSERT INTO retries VALUES(?,?,?,?)",
                    (sid, key, request, canonical(result)),
                )
            return result

    def session(self, db, sid, researching=False):
        session = self.store.get(db, sid, "session", sid)
        if researching:
            require(
                session["phase"] == "researching",
                "conflict",
                "Research has been frozen.",
            )
        return session

    def profile(self, pid):
        require(
            pid in self.backend.profiles,
            "resource_unavailable",
            "Unknown or unconfigured execution profile.",
        )
        return self.backend.profiles[pid]

    def cost(self, db, sid, candidate_id, profile_id):
        candidate = self.store.get(db, sid, "candidate", candidate_id)
        candidate["files"] = self.store.get(db, sid, "package", candidate_id)["files"]
        profile = self.profile(profile_id)
        report = self.backend.inspect(candidate, profile)
        require(
            report["coverage"] == "complete",
            "incomplete_coverage",
            "Runtime does not cover this candidate.",
        )
        return {
            **report,
            "candidate_id": candidate_id,
            "candidate_hash": candidate["hash"],
            "profile_id": profile_id,
            "runtime_revision": self.backend.revision,
            "profile_hash": identity(profile),
        }

    def quote(self, db, sid, p, operation="train"):
        self.session(db, sid, researching=True)
        candidate_id, pid = p.get("candidate_id"), p.get("profile_id")
        report = self.cost(db, sid, candidate_id, pid)
        profile = self.profile(pid)
        gpu_id = self.backend.validate_placement(
            profile, p.get("gpu_id"), p.get("server_id")
        )
        require(
            profile["kind"] == "research" or operation in {"validate", "debug"},
            "denied_access",
            "Final profiles cannot be used for research training.",
        )
        checkpoint_id, start = p.get("checkpoint_id"), 0
        if checkpoint_id:
            checkpoint = self.store.get(db, sid, "checkpoint", checkpoint_id)
            require(
                checkpoint["candidate_hash"] == report["candidate_hash"]
                and checkpoint["profile_hash"] == report["profile_hash"]
                and checkpoint["runtime_revision"] == self.backend.revision
                and checkpoint["complete"],
                "incompatible_checkpoint",
                "Checkpoint identity or complete state differs.",
            )
            start = checkpoint["step"]
        target = integer(p.get("target_update", 1), start + 1, profile["max_updates"])
        resource = p.get("resource_profile_id", profile["resource_profile_id"])
        require(
            resource == profile["resource_profile_id"],
            "resource_unavailable",
            "Resource profile does not match this execution profile.",
        )
        memory = self.backend.bounded_profile(profile).get("memory_limit_bytes")
        quota = profile.get("checkpoint_quota_bytes")
        integer(memory, 1)
        integer(quota, 1)
        required_memory = (
            report["parameters"] * 64
            + profile["batch_size"] * profile["sequence_len"] * profile["width"] * 256
        )
        require(
            required_memory <= memory,
            "resource_unavailable",
            "Candidate exceeds the declared conservative memory bound.",
        )
        storage = sum(c["size_bytes"] for c in self.store.all(db, sid, "checkpoint"))
        jobs = self.store.all(db, sid, "job")
        storage += sum(
            j.get("checkpoint_reservation_bytes", 0)
            for j in jobs
            if j["state"] not in TERMINAL
        )
        checkpoint_bound = (
            report["parameters"] * 64 + 16 * 1024 * 1024 if operation == "train" else 0
        )
        require(
            storage + checkpoint_bound <= quota,
            "resource_unavailable",
            "Session checkpoint storage quota is reserved or exhausted.",
        )
        amount = self.backend.bound(
            report, operation, target - start, profile, check=p.get("check")
        )
        integer(amount, 1)
        quote = {
            "candidate_id": candidate_id,
            "profile_id": pid,
            "operation": operation,
            "target_update": target,
            "start_update": start,
            "checkpoint_id": checkpoint_id,
            "resource_profile_id": resource,
            "server_id": self.backend.server_id,
            "gpu_id": gpu_id,
            "runtime_revision": self.backend.revision,
            "profile_hash": report["profile_hash"],
            "candidate_hash": report["candidate_hash"],
            "bound": str(amount),
            "cost_report": report,
            "expires_at": time.time() + 300,
            "checkpoint_reservation_bytes": checkpoint_bound,
            "affordable": amount <= int(self.store.budget(db, sid)["available"]),
        }
        if operation == "debug":
            quote["check"] = p["check"]
        return self.store.put(db, sid, "quote", quote)

    def queue(self, db, sid, quote_id):
        self.session(db, sid, researching=True)
        quote = self.store.get(db, sid, "quote", quote_id)
        require(
            quote["expires_at"] > time.time()
            and quote["runtime_revision"] == self.backend.revision
            and quote["profile_hash"] == identity(self.profile(quote["profile_id"])),
            "stale_quote",
            "Quote expired or its runtime/profile changed.",
        )
        require(
            not any(j["quote_id"] == quote_id for j in self.store.all(db, sid, "job")),
            "conflict",
            "Quote already consumed; read the original job.",
        )
        reserved_storage = sum(
            j.get("checkpoint_reservation_bytes", 0)
            for j in self.store.all(db, sid, "job")
            if j["state"] not in TERMINAL
        )
        stored = sum(c["size_bytes"] for c in self.store.all(db, sid, "checkpoint"))
        require(
            stored + reserved_storage + quote["checkpoint_reservation_bytes"]
            <= self.profile(quote["profile_id"])["checkpoint_quota_bytes"],
            "resource_unavailable",
            "Checkpoint capacity changed since quotation.",
        )
        job = {
            "id": new_id(),
            "checkpoint_reservation_bytes": quote["checkpoint_reservation_bytes"],
            "quote_id": quote_id,
            "state": "queued",
            "progress": 0,
            "lease": None,
            "cancel_requested": False,
            "operation": quote["operation"],
        }
        self.store.reserve(db, sid, job["id"], int(quote["bound"]))
        return self.store.put(db, sid, "job", job)

    def dispatch(self, db, sid, op, p):
        s = self.store
        if op == "describe_space":
            from .candidate_contract import describe_candidate_contract

            return {
                "schema_version": "gym.api.v1",
                "contract": CONTRACT,
                "language": "python",
                "plugin_interface": "nexttransformer.models.plugins",
                "profiles": list(self.backend.public_profiles()),
                "operations": sorted(READS | MUTATIONS),
                "operation_schemas": OPERATION_SCHEMAS,
                "candidate_contract": describe_candidate_contract(),
                "required_checks": sorted(REQUIRED_CHECKS),
                "runtime_revision": self.backend.revision,
            }
        if op == "get_resources":
            session = self.session(db, sid)
            jobs = s.all(db, sid, "job")
            running = sum(
                j["state"] in {"running", "cancelling", "reconciliation"} for j in jobs
            )
            if p.get("resource_profile_id") is not None:
                require(
                    any(
                        x["resource_profile_id"] == p["resource_profile_id"]
                        for x in self.backend.profiles.values()
                    ),
                    "not_found",
                    "Unknown resource profile.",
                )
            global_busy = any(
                json.loads(raw)["state"] in {"running", "cancelling", "reconciliation"}
                for (raw,) in db.execute("SELECT value FROM objects WHERE kind='job'")
            )
            server = self.backend.resources(global_busy)
            return {
                "schema_version": "gym.resources.v1",
                "observed_at": time.time(),
                "capability_revision": self.backend.revision,
                "profiles": list(self.backend.public_profiles()),
                "session_limits": {
                    "max_concurrent_experiments": session["max_concurrent"]
                },
                "own_running": running,
                "own_queued": sum(j["state"] == "queued" for j in jobs),
                "allocatable_slots": min(
                    server["allocatable_slots"],
                    max(0, session["max_concurrent"] - running),
                ),
                "server_count": 1,
                "reachable_server_count": 1,
                "servers": [server],
                "throughput": None,
                "throughput_reason": "not_measured",
            }
        if op == "list_components":
            from .history import page

            return page(
                [{"id": k, "reference_id": k} for k in self.catalog],
                p.get("cursor", ""),
                p.get("limit", 50),
            )
        if op in {"list_experiments", "read_experiment"}:
            from .history import page, summary

            scope = p.get("scope", "published")
            require(
                scope in {"published", "session"},
                "invalid_request",
                "Choose published or session history.",
            )
            if scope == "published":
                if op == "list_experiments":
                    return page(
                        [summary(x) for x in self.history.values()],
                        p.get("cursor", ""),
                        p.get("limit", 50),
                    )
                require(
                    p.get("experiment_id") in self.history,
                    "not_found",
                    "Unknown published experiment.",
                )
                return self.history[p["experiment_id"]]
            jobs = [j for j in s.all(db, sid, "job") if j.get("quote_id")]
            if op == "list_experiments":
                return page(
                    [
                        {
                            "id": j["id"],
                            "scope": "session",
                            "status": j["state"],
                            "operation": j["operation"],
                        }
                        for j in jobs
                    ],
                    p.get("cursor", ""),
                    p.get("limit", 50),
                )
            job = s.get(db, sid, "job", p.get("experiment_id"))
            require(
                job.get("quote_id") is not None,
                "not_found",
                "Job has no numerical experiment.",
            )
            quote = s.get(db, sid, "quote", job["quote_id"])
            candidate = s.get(db, sid, "candidate", quote["candidate_id"])
            return {
                "id": job["id"],
                "scope": "session",
                "status": job["state"],
                "candidate": candidate,
                "files": s.get(db, sid, "package", candidate["id"])["files"],
                "profile_id": quote["profile_id"],
                "server_id": job.get("server_id"),
                "result": s.get(db, sid, "result", job["id"])
                if job["state"] == "succeeded"
                else None,
            }
        if op == "read_reference":
            require(
                p.get("resource_id") in self.catalog,
                "not_found",
                "Unknown approved reference ID.",
            )
            return {
                "resource_id": p["resource_id"],
                "content": self.catalog[p["resource_id"]],
            }
        if op == "get_budget":
            return s.budget(db, sid)
        if op == "submit_candidate":
            self.session(db, sid, researching=True)
            try:
                blob = base64.b64decode(p["package_base64"], validate=True)
            except (KeyError, ValueError, TypeError):
                raise GymError(
                    "invalid_artifact", "Expected base64 ZIP package."
                ) from None
            require(
                len(s.all(db, sid, "candidate")) < 128,
                "resource_unavailable",
                "Session candidate storage quota reached.",
            )
            candidate = ingest(blob)
            for parent in candidate["parent_ids"]:
                s.get(db, sid, "candidate", parent)
            files = candidate.pop("files")
            candidate = s.put(db, sid, "candidate", candidate)
            s.put(db, sid, "package", {"id": candidate["id"], "files": files})
            return candidate
        if op == "list_candidates":
            limit, cursor = integer(p.get("limit", 50), 1, 100), p.get("cursor", "")
            require(isinstance(cursor, str), "invalid_request", "Invalid cursor.")
            items = [c for c in s.all(db, sid, "candidate") if c["id"] > cursor]
            page = items[:limit]
            return {
                "items": page,
                "next_cursor": page[-1]["id"] if len(items) > limit else None,
            }
        if op == "inspect_cost":
            return self.cost(db, sid, p.get("candidate_id"), p.get("profile_id"))
        if op == "quote_experiment":
            return self.quote(db, sid, p)
        if op == "run_experiment":
            return self.queue(db, sid, p.get("quote_id"))
        if op in {"validate_candidate", "debug_candidate"}:
            if op == "debug_candidate":
                require(
                    p.get("check") in REQUIRED_CHECKS,
                    "invalid_request",
                    "Unknown named diagnostic.",
                )
                if "shape" in p:
                    profile = self.profile(p.get("profile_id"))
                    require(
                        p["shape"] == [profile["batch_size"], profile["sequence_len"]],
                        "invalid_request",
                        "Diagnostic shape must match the pinned supported shape.",
                    )
            quote = self.quote(
                db, sid, p, "debug" if op == "debug_candidate" else "validate"
            )
            return self.queue(db, sid, quote["id"])
        if op == "calibrate_candidate":
            self.session(db, sid, researching=True)
            original = s.get(db, sid, "candidate", p.get("candidate_id"))
            grid = p.get("grid")
            require(
                isinstance(grid, list) and 0 < len(grid) <= 32,
                "invalid_request",
                "Provide at most 32 architecture choices.",
            )
            require(
                len(s.all(db, sid, "candidate")) + len(grid) <= 128,
                "resource_unavailable",
                "Calibration exceeds candidate storage quota.",
            )
            derived = []
            for architecture in grid:
                package = s.get(db, sid, "package", original["id"])["files"].copy()
                proposal = json.loads(package["proposal.json"])
                proposal["parent_ids"] = [original["id"]]
                package["proposal.json"] = canonical(proposal)
                package["architecture.json"] = canonical(architecture)
                out = io.BytesIO()
                with zipfile.ZipFile(out, "w") as archive:
                    for name, content in package.items():
                        archive.writestr(name, content)
                candidate = ingest(out.getvalue())
                candidate.pop("files")
                candidate = s.put(db, sid, "candidate", candidate)
                s.put(db, sid, "package", {"id": candidate["id"], "files": package})
                derived.append(
                    {
                        "candidate": candidate,
                        "cost": self.cost(
                            db, sid, candidate["id"], p.get("profile_id")
                        ),
                    }
                )
            job = s.put(
                db,
                sid,
                "job",
                {
                    "state": "succeeded",
                    "operation": "calibrate",
                    "progress": len(derived),
                    "quote_id": None,
                    "lease": None,
                    "cancel_requested": False,
                },
            )
            s.put(
                db,
                sid,
                "result",
                {"id": job["id"], "derived": derived, "analysis": "static_only"},
            )
            return job
        if op == "cancel_job":
            job = s.get(db, sid, "job", p.get("job_id"))
            if job["state"] == "queued":
                job["state"] = "cancelled"
                s.release(db, sid, job["id"])
            elif job["state"] == "running":
                job.update(state="cancelling", cancel_requested=True)
            return s.update(db, sid, "job", job)
        if op == "read_events":
            cursor, limit = (
                integer(p.get("cursor", 0)),
                integer(p.get("limit", 50), 1, 100),
            )
            rows = db.execute(
                "SELECT id,value FROM events WHERE owner=? AND id>? ORDER BY id LIMIT ?",
                (sid, cursor, limit),
            ).fetchall()
            return {
                "items": [{"id": r[0], **json.loads(r[1])} for r in rows],
                "next_cursor": rows[-1][0] if rows else cursor,
            }
        if op == "finalize":
            return self.finalize(db, sid, p)
        mapping = {
            "get_candidate": ("candidate", "candidate_id"),
            "get_job": ("job", "job_id"),
            "get_result": ("result", "job_id"),
            "get_submission": ("submission", "submission_id"),
        }
        if op in mapping:
            kind, field = mapping[op]
            value = s.get(db, sid, kind, p.get(field))
            if kind == "submission" and value.get("final_run_id"):
                run = s.get(db, sid, "final_run", value["final_run_id"])
                job = s.get(db, run["owner"], "job", run["job_id"])
                value["final_run_state"] = job["state"]
            if kind == "job":
                value.pop("lease", None)
            return value
        raise GymError("invalid_request", "Unknown operation.")

    def finalize(self, db, sid, p):
        session = self.session(db, sid, researching=True)
        candidate = self.store.get(db, sid, "candidate", p.get("candidate_id"))
        profile = self.profile(p.get("profile_id"))
        require(
            profile["kind"] == "final",
            "invalid_request",
            "Select an intended final profile.",
        )
        reports = [
            r
            for r in self.store.all(db, sid, "report")
            if r["candidate_hash"] == candidate["hash"]
            and r["runtime_revision"] == self.backend.revision
            and r["profile_hash"] == identity(profile)
        ]
        require(
            any(
                set(r["checks"]) >= REQUIRED_CHECKS
                and all(r["checks"].values())
                and r["admitted"]
                for r in reports
            ),
            "missing_evidence",
            "Run and pass bounded final-profile validation and admission before freezing.",
        )
        require(
            not session["integrity_violation"],
            "integrity_violation",
            "Session cannot be finalized.",
        )
        jobs = self.store.all(db, sid, "job")
        require(
            all(j["state"] in TERMINAL for j in jobs)
            and self.store.budget(db, sid)["reserved"] == "0",
            "unsettled_work",
            "Cancel or drain jobs and reconcile charges before freezing.",
        )
        submission = self.store.put(
            db,
            sid,
            "submission",
            {
                "candidate_id": candidate["id"],
                "candidate_hash": candidate["hash"],
                "configuration_hash": candidate["configuration_hash"],
                "profile_id": p["profile_id"],
                "profile_hash": identity(profile),
                "contract": CONTRACT,
                "runtime_revision": self.backend.revision,
                "evidence_ids": [r["id"] for r in reports],
                "verification_state": "pending_verification",
                "final_run_state": "not_scheduled",
            },
        )
        session.update(phase="submitted", final_choice=submission["id"])
        self.store.update(db, sid, "session", session)
        return submission

    def verify(self, db, sid, p):
        session = self.session(db, sid)
        submission = self.store.get(db, sid, "submission", p.get("submission_id"))
        require(
            session["final_choice"] == submission["id"],
            "denied_access",
            "Receipt is not this trial's final choice.",
        )
        if submission["verification_state"] != "pending_verification":
            return submission
        reports = [
            self.store.get(db, sid, "report", rid) for rid in submission["evidence_ids"]
        ]
        valid = (
            not session["integrity_violation"]
            and self.store.budget(db, sid)["reserved"] == "0"
            and submission["runtime_revision"] == self.backend.revision
            and any(
                r["admitted"]
                and set(r["checks"]) >= REQUIRED_CHECKS
                and all(r["checks"].values())
                and r["candidate_hash"] == submission["candidate_hash"]
                and r["profile_hash"] == submission["profile_hash"]
                for r in reports
            )
        )
        submission["verification_state"] = "verified" if valid else "rejected"
        return self.store.update(db, sid, "submission", submission)
