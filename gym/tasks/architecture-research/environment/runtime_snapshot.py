"""Observe kernel process state inside main; no environment or command arguments."""

import json
import os
import time
from pathlib import Path

processes = []
for directory in Path("/proc").iterdir():
    if not directory.name.isdecimal():
        continue
    try:
        status = dict(
            line.split(":", 1)
            for line in (directory / "status").read_text().splitlines()
            if ":" in line
        )
        processes.append(
            {
                "pid": int(directory.name),
                "name": status["Name"].strip(),
                "state": status["State"].strip(),
                "parent_pid": int(status["PPid"]),
                "memory": status.get("VmRSS", "unavailable").strip(),
            }
        )
    except (OSError, ValueError, KeyError):
        continue
value = {
    "schema_version": "gym.runtime.v1",
    "observed_at": time.time(),
    "source": "main_container_procfs",
    "processes": processes,
}
value["agent_state"] = None
value["agent_state_source"] = "agent_reported"
try:
    fd = os.open(
        "/workspace/agent-state.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    with os.fdopen(fd) as stream:
        import stat

        if stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raw = stream.read(65537)
            if len(raw) <= 65536:
                value["agent_state"] = json.loads(raw)
except (OSError, ValueError):
    pass
root = Path("/run/gym-runtime")
root.mkdir(mode=0o755, exist_ok=True)
(root / "status.json").write_text(json.dumps(value))
print(json.dumps(value))
