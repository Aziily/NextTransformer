"""Submit a native candidate and run a bounded experiment using HTTP only.

Run inside the Agent container. The operator owns resources, profiles and the
session's total FLOPs allocation; this script never imports training code.
"""

import argparse
import json
from pathlib import Path
import uuid

from nexttransformer_gym import GymClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Source-only native plugin ZIP")
    parser.add_argument(
        "--credential", type=Path, default=Path("/run/gym-auth/credential.json")
    )
    parser.add_argument("--profile", default="research-fixture")
    parser.add_argument("--target-update", type=int, default=2)
    parser.add_argument("--gpu", help="Allocated GPU UUID or physical index")
    parser.add_argument("--server", help="Node ID returned by get_resources")
    args = parser.parse_args()
    binding = json.loads(args.credential.read_text())
    gym = GymClient(binding["endpoint"], binding["token"])
    space, resources = gym.describe_space(), gym.get_resources()
    profile = next(p for p in space["profiles"] if p["id"] == args.profile)
    node = next(
        n for n in resources["servers"] if args.server is None or n["id"] == args.server
    )
    placement = {"server_id": node["id"]}
    if args.gpu is not None:
        placement["gpu_id"] = args.gpu
    elif profile["device"].startswith("cuda"):
        # Explicitly choose an available authorized card when one is observed.
        # If all are busy, omit gpu_id and let Gym queue until one becomes free.
        available = [gpu for gpu in node["gpus"] if gpu["available"]]
        if available:
            placement["gpu_id"] = available[0]["id"]
    key = uuid.uuid4().hex
    candidate = gym.submit_candidate(
        package_bytes=args.candidate.read_bytes(), idempotency_key=key + "-upload"
    )
    quote = gym.quote_experiment(
        candidate_id=candidate["id"],
        profile_id=args.profile,
        target_update=args.target_update,
        **placement,
    )
    if not quote["affordable"]:
        raise SystemExit("The quoted experiment exceeds the remaining session budget.")
    job = gym.run_experiment(quote_id=quote["id"], idempotency_key=key + "-run")
    # Retain this ID to track or cancel a queued experiment if the wait times out.
    print(json.dumps({"job_id": job["id"], "placement": placement}), flush=True)
    print(json.dumps(gym.wait_result(job["id"]), indent=2))
    print(json.dumps(gym.get_budget(), indent=2))


if __name__ == "__main__":
    main()
