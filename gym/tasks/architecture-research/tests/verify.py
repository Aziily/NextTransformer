import json
from pathlib import Path

from nexttransformer_gym.evidence import verify_receipt

# Harbor restores declared artifacts at their original source paths, regardless
# of the human-readable destination names in the host trial's artifact folder.
output = Path("/logs/verifier")
output.mkdir(parents=True, exist_ok=True)
try:
    receipt = json.loads(Path("/workspace/submission.json").read_text())
    if not isinstance(receipt, dict) or set(receipt) != {
        "session_id",
        "submission_id",
        "candidate_hash",
    }:
        raise ValueError("Malformed final receipt")
except (OSError, ValueError):
    report = {
        "valid": False,
        "status": "invalid_submission",
        "reason": "missing_or_malformed_receipt",
        "meaning": "submission_validity_only",
        "performance_score": None,
    }
else:
    # Failure to read/authenticate trusted evidence is an infrastructure error,
    # not an invented zero performance score or candidate-owned failure.
    try:
        report = verify_receipt(
            receipt,
            json.loads(Path("/export/evidence.json").read_text()),
            json.loads(Path("/run/verifier-auth/credential.json").read_text()),
        )
    except Exception:
        (output / "report.json").write_text(
            json.dumps(
                {
                    "status": "infrastructure_failure",
                    "reason": "trusted_evidence_unavailable_or_invalid",
                }
            )
        )
        raise
(output / "report.json").write_text(json.dumps(report))
(output / "reward.txt").write_text("1" if report["valid"] else "0")
