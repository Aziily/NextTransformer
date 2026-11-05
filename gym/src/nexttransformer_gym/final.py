"""Privileged final allocation and dispatch, never part of the agent HTTP API."""

import fcntl

from .common import identity, integer, require, new_id


def launch_final_run(service, session_id, submission_id, allocation, idempotency_key):
    """Local controller authority is required to call this entry point."""
    integer(allocation, 1)
    require(
        isinstance(idempotency_key, str) and bool(idempotency_key),
        "invalid_request",
        "Final launch requires an idempotency key.",
    )
    s = service.store
    with s.transaction() as db:
        submission = s.get(db, session_id, "submission", submission_id)
        require(
            submission["verification_state"] == "verified",
            "missing_evidence",
            "Verify the frozen deliverable first.",
        )
        require(
            submission["runtime_revision"] == service.backend.revision,
            "stale_quote",
            "Runtime changed after verification.",
        )
        profile = service.profile(submission["profile_id"])
        require(
            identity(profile) == submission["profile_hash"],
            "stale_quote",
            "Final profile changed after submission. Restore the admitted profile.",
        )
        request_hash = identity([submission_id, allocation, idempotency_key])
        previous = s.all(db, session_id, "final_run")
        if previous:
            require(
                previous[0]["request_hash"] == request_hash,
                "conflict",
                "Final choice already has a separately allocated run.",
            )
            return previous[0]
        require(
            profile.get("official_compatibility_verified")
            or profile["data"]["kind"] == "synthetic_fixture",
            "missing_evidence",
            "Formal runtime parity and admission must be independently verified.",
        )
        candidate = s.get(db, session_id, "candidate", submission["candidate_id"])
        report = service.cost(db, session_id, candidate["id"], profile["id"])
        official = profile["data"]["kind"] != "synthetic_fixture"
        if official:
            from .official import validate_plan, official_world

            validate_plan(profile.get("official_plan"), candidate["hash"])
            official_world(profile)
        bound = (
            allocation
            if official
            else service.backend.bound(report, "train", profile["max_updates"], profile)
        )
        require(
            allocation >= bound,
            "insufficient_budget",
            "Separate final allocation is smaller than the execution bound.",
        )
        # A distinct owner/account pins final artifacts beyond research-session cleanup.
        owner = new_id()
        s.put(db, owner, "account", {"id": owner, "limit": str(allocation)})
        s.put(db, owner, "candidate", candidate)
        s.put(db, owner, "package", s.get(db, session_id, "package", candidate["id"]))
        quote = s.put(
            db,
            owner,
            "quote",
            {
                "candidate_id": candidate["id"],
                "profile_id": profile["id"],
                "operation": "official" if official else "train",
                "target_update": profile["max_updates"],
                "start_update": 0,
                "checkpoint_id": None,
                "candidate_hash": candidate["hash"],
                "profile_hash": identity(profile),
                "runtime_revision": service.backend.revision,
                "cost_report": report,
                "bound": str(bound),
            },
        )
        job = {
            "id": new_id(),
            "quote_id": quote["id"],
            "operation": "official" if official else "train",
            "state": "queued",
            "progress": 0,
            "lease": None,
            "cancel_requested": False,
        }
        s.reserve(db, owner, job["id"], bound)
        s.put(db, owner, "job", job)
        run = s.put(
            db,
            session_id,
            "final_run",
            {
                "submission_id": submission_id,
                "owner": owner,
                "job_id": job["id"],
                "allocation": str(allocation),
                "request_hash": request_hash,
                "profile_id": profile["id"],
                "fresh_initialization": True,
                "attempts": 1,
            },
        )
        submission.update(final_run_state="queued", final_run_id=run["id"])
        s.update(db, session_id, "submission", submission)
        return run


def get_final_run(service, session_id, run_id):
    with service.store.transaction() as db:
        run = service.store.get(db, session_id, "final_run", run_id)
        job = service.store.get(db, run["owner"], "job", run["job_id"])
        submission = service.store.get(
            db, session_id, "submission", run["submission_id"]
        )
        submission["final_run_state"] = job["state"]
        service.store.update(db, session_id, "submission", submission)
        result = None
        if job["state"] == "succeeded":
            measured = service.store.get(db, run["owner"], "result", run["job_id"])
            if measured.get("official_compatible"):
                return {
                    "id": run_id,
                    "state": "succeeded",
                    "result": measured["record"],
                    "fresh_initialization": True,
                }
            # Fixture outputs never acquire an official rank, BPB or CORE claim.
            result = {
                "status": "fixture_complete",
                "official_compatible": False,
                "dev_bpb": measured["dev_bpb"],
                "bpb": None,
                "core": None,
                "rank": None,
                "candidate_hash": measured["candidate_hash"],
                "runtime_revision": measured["runtime_revision"],
            }
        return {
            k: v
            for k, v in {**run, "state": job["state"], "result": result}.items()
            if k not in {"owner", "request_hash"}
        }


def reconcile(service, owner, job_id, *, worker_confirmed_stopped):
    """Predeclared conservative crash settlement; never claim unknown work was free."""
    require(
        worker_confirmed_stopped is True,
        "unsettled_work",
        "Controller must establish worker termination before settling.",
    )
    with (service.backend.state_dir / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            require(False, "unsettled_work", "A scheduler still owns the worker slot.")
        with service.store.transaction() as db:
            job = service.store.get(db, owner, "job", job_id)
            if job.get("settlement") == "conservative_reserved_bound":
                return job
            # A controller crash can leave running/cancelling persisted before
            # any recovery code gets a chance to mark reconciliation.
            require(
                job["state"] in {"running", "cancelling", "reconciliation"},
                "conflict",
                "Only uncertain work can be reconciled.",
            )
            reservation = service.store.get(db, owner, "reservation", job_id)
            service.store.charge(
                db,
                owner,
                job_id,
                identity([job_id, "conservative_settlement"]),
                int(reservation["remaining"]),
            )
            job.update(
                state="failed", lease=None, settlement="conservative_reserved_bound"
            )
            service.store.update(db, owner, "job", job)
            return job
