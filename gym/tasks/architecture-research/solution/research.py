"""Bounded, model-free integration oracle; all numerical work uses the Gym."""

import io
import json
import runpy
from pathlib import Path
from nexttransformer_gym import GymClient

binding = json.loads(Path("/run/gym-auth/credential.json").read_text())
gym = GymClient(binding["endpoint"], binding["token"])
make = runpy.run_path("/opt/gym/make_candidate.py")["package"]
observations = []
for activation in ("square", "tanh"):
    package = io.BytesIO()
    make(package, activation)
    candidate = gym.submit_candidate(package.getvalue(), idempotency_key=activation)
    quote = gym.quote_experiment(
        candidate_id=candidate["id"], profile_id="research-fixture", target_update=1
    )
    job = gym.run_experiment(
        quote_id=quote["id"], idempotency_key="train-" + activation
    )
    result = gym.wait_result(job["id"])
    observations.append({"candidate": candidate, "result": result})
selected = min(observations, key=lambda item: item["result"]["dev_bpb"])
candidate = selected["candidate"]
quote = gym.quote_experiment(
    candidate_id=candidate["id"],
    profile_id="research-fixture",
    checkpoint_id=selected["result"]["checkpoint_id"],
    target_update=2,
)
job = gym.run_experiment(quote_id=quote["id"], idempotency_key="continue")
gym.wait_result(job["id"])
checks = gym.validate_candidate(
    candidate_id=candidate["id"],
    profile_id="final-fixture",
    idempotency_key="final-checks",
)
gym.wait_result(checks["id"])
submission = gym.finalize(
    candidate_id=candidate["id"],
    profile_id="final-fixture",
    idempotency_key="final-choice",
)
Path("/workspace/submission.json").write_text(
    json.dumps(
        {
            "session_id": binding["session_id"],
            "submission_id": submission["id"],
            "candidate_hash": candidate["hash"],
        }
    )
)
Path("/workspace/research.json").write_text(json.dumps(observations))
