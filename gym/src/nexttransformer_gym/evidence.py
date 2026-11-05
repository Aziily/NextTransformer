"""Bounded authenticated server evidence for Harbor's separate verifier."""

import hashlib
import hmac
from .common import canonical, require
from .service import TERMINAL


def snapshot(service, credential):
    sid = credential["session_id"]
    with service.store.transaction() as db:
        principal = service.authenticate(db, credential["token"])
        require(
            principal["role"] == "verifier" and principal["session_id"] == sid,
            "denied_access",
            "A trusted verifier binding is required.",
        )
        session = service.session(db, sid)
        submission = (
            service.store.get(db, sid, "submission", session["final_choice"])
            if session["final_choice"]
            else None
        )
        if submission:
            submission = service.verify(db, sid, {"submission_id": submission["id"]})
        value = {
            "schema_version": "gym.evidence.v1",
            "session_id": sid,
            "trial_id": session["trial_id"],
            "phase": session["phase"],
            "integrity_violation": session["integrity_violation"],
            "submission": submission,
            "budget": service.store.budget(db, sid),
            "jobs": [
                {"id": j["id"], "state": j["state"]}
                for j in service.store.all(db, sid, "job")
            ],
        }
    signature = hmac.new(
        credential["token"].encode(), canonical(value).encode(), hashlib.sha256
    ).hexdigest()
    return {"payload": value, "signature": signature}


def verify_receipt(receipt, signed, credential):
    value = signed["payload"]
    expected = hmac.new(
        credential["token"].encode(), canonical(value).encode(), hashlib.sha256
    ).hexdigest()
    require(
        hmac.compare_digest(expected, signed["signature"]),
        "denied_access",
        "Evidence signature mismatch.",
    )
    require(
        value["session_id"] == credential["session_id"],
        "denied_access",
        "Trial/session binding mismatch.",
    )
    submission = value["submission"]
    valid = (
        submission is not None
        and receipt["session_id"] == value["session_id"]
        and submission["id"] == receipt["submission_id"]
        and submission["candidate_hash"] == receipt["candidate_hash"]
        and submission["verification_state"] == "verified"
        and value["phase"] == "submitted"
        and not value["integrity_violation"]
        and value["budget"]["reserved"] == "0"
        and all(j["state"] in TERMINAL for j in value["jobs"])
    )
    return {
        "valid": bool(valid),
        "meaning": "submission_validity_only",
        "performance_score": None,
        "final_run_state": submission["final_run_state"]
        if submission
        else "not_scheduled",
    }
