"""Harbor's trusted sidecar collect hook; never installed in the agent image."""

import json
from pathlib import Path
from .backend import NativeBackend
from .service import Service
from .evidence import snapshot
from .server import atomic_json


def main():
    backend = NativeBackend(
        json.loads(Path("/opt/gym/profiles.json").read_text()),
        "/state",
        "/opt/platform",
    )
    service = Service("/state/gym.sqlite3", backend)
    credential = json.loads(Path("/run/verifier-auth/credential.json").read_text())
    atomic_json("/export/evidence.json", snapshot(service, credential))
    # Public signed evidence crosses container UIDs through Harbor's artifact copy.
    # Credentials remain in their separate, inaccessible volume.
    Path("/export/evidence.json").chmod(0o444)


if __name__ == "__main__":
    main()
