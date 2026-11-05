"""Filesystem/process/network isolation for native research plugin workers.

This isolates OS resources. It does not make arbitrary in-process Python a trusted
accounting authority; adversarial accounting certification is a separate gate.
"""

import shutil
import os
import sys
from pathlib import Path
from .common import require


def command(python, platform, request):
    executable = shutil.which("bwrap")
    require(
        executable is not None,
        "resource_unavailable",
        "Native plugin workers require bubblewrap; there is no unsandboxed research fallback.",
    )
    args = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    roots = {Path(p) for p in ("/usr", "/bin", "/lib", "/lib64") if Path(p).exists()}
    roots |= {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(python).resolve().parent.parent,
        Path(python).absolute().parent.parent,
        Path(__file__).resolve().parent,
        Path(platform) / "nexttransformer",
        Path(platform) / "nanochat",
    }
    if Path(python).is_symlink():
        target = Path(os.readlink(python))
        if target.is_absolute():
            roots.add(target.parent.parent)
    for path in sorted(roots, key=lambda p: (len(p.parts), str(p))):
        if any(path != parent and path.is_relative_to(parent) for parent in roots):
            continue
        args += ["--ro-bind", str(path), str(path)]
    for path in ("/etc/ld.so.cache", "/etc/localtime"):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    gpu = request.get("gpu") if request["mode"] != "inspect" else None
    args += [
        "--proc" if gpu else "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp/cache",
    ]
    if request.get("output"):
        directory = Path(request["output"]).parent
        args += ["--bind", str(directory), str(directory)]
    if request.get("checkpoint"):
        args += ["--ro-bind", request["checkpoint"], request["checkpoint"]]
    data = request["profile"].get("data", {"kind": "synthetic_fixture"})
    if request["mode"] != "inspect" and data["kind"] != "synthetic_fixture":
        for path in [
            *data["train_paths"],
            data["dev_path"],
            data["tokenizer"],
            data["token_bytes"],
        ]:
            resolved = str(Path(path).resolve())
            args += ["--ro-bind", resolved, resolved]
    if request["mode"] != "inspect" and str(
        request["profile"].get("device", "cpu")
    ).startswith("cuda"):
        require(
            gpu is not None
            and type(gpu.get("device_minor")) is int
            and gpu["device_minor"] >= 0,
            "resource_unavailable",
            "A scheduler GPU allocation is required.",
        )
        for node in (
            f"/dev/nvidia{gpu['device_minor']}",
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
        ):
            require(
                Path(node).exists(),
                "resource_unavailable",
                "Allocated CUDA device nodes are unavailable.",
            )
            args += ["--dev-bind", node, node]
        # Private PID namespace procfs exposes only the worker; driver metadata is
        # read-only. No host process tree, host network, or other GPU nodes.
        if Path("/proc/driver/nvidia").exists():
            args += ["--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia"]
    return [*args, "--", python, "-m", "nexttransformer_gym.worker"]
