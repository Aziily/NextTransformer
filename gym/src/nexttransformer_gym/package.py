"""Source-only native plugin packages; ingestion never imports participant code."""

import io
import json
import re
import stat
import zipfile
from .common import GymError, identity, require
from .candidate_contract import ROUTING_SEED, validate_model_field_types

LIMIT = 1 << 20
CONTRACT = "reference_plugin_v1"


def validate_metadata(method, declaration, architecture):
    from nexttransformer.models.plugins import FAMILIES, SHELL, _value
    from nexttransformer.models.registry import BUILTIN_METHODS, MIXER_KINDS

    require(
        method not in (*BUILTIN_METHODS, *MIXER_KINDS),
        "invalid_artifact",
        "A plugin cannot replace a registered method.",
    )
    require(
        isinstance(declaration, dict)
        and declaration.get("name") == method
        and declaration.get("family") in FAMILIES,
        "incompatible_contract",
        "Use the native method.json declaration.",
    )
    options = declaration.get("options", {})
    require(
        isinstance(options, dict)
        and all(
            re.fullmatch(r"[a-z][a-z0-9_]{1,40}", k) and _value(v)
            for k, v in options.items()
        ),
        "invalid_artifact",
        "Invalid native plugin options.",
    )
    require(
        isinstance(declaration.get("shell", {}), dict)
        and set(declaration.get("shell", {})) <= set(SHELL)
        and isinstance(declaration.get("debug_options", {}), dict)
        and set(declaration.get("debug_options", {})) <= set(options)
        and type(declaration.get("layer_multiple", 1)) is int
        and declaration.get("layer_multiple", 1) > 0,
        "invalid_artifact",
        "Invalid native shell, debug options or layer multiple.",
    )
    # These are the existing ModelConfig fields, not a second architecture DSL.
    validate_model_field_types(architecture)
    validate_model_field_types(declaration.get("shell", {}))
    require(
        architecture.get("method", method) == method,
        "incompatible_contract",
        "Config method must match the plugin.",
    )
    require(
        architecture.get("routing_seed", ROUTING_SEED) == ROUTING_SEED,
        "incompatible_contract",
        "The native routing seed is fixed by the execution protocol.",
    )
    require(
        isinstance(architecture.get("options", {}), dict)
        and set(architecture.get("options", {})) <= set(options),
        "invalid_artifact",
        "Config options must be declared by the plugin.",
    )


def ingest(blob):
    require(
        isinstance(blob, bytes) and len(blob) <= LIMIT,
        "invalid_artifact",
        "Package exceeds 1 MiB.",
    )
    try:
        files, total = {}, 0
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            require(
                len(archive.infolist()) <= 32,
                "invalid_artifact",
                "Too many archive entries.",
            )
            for entry in archive.infolist():
                name = entry.filename
                total += entry.file_size
                require(
                    total <= LIMIT
                    and not entry.flag_bits & 1
                    and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
                    and re.fullmatch(r"[A-Za-z0-9_./-]+", name) is not None
                    and all(p not in {"", ".", ".."} for p in name.split("/"))
                    and name not in files,
                    "invalid_artifact",
                    "Unsafe, duplicate, encrypted or oversized archive member.",
                )
                files[name] = archive.read(entry).decode("utf-8")
                require(
                    "\0" not in files[name], "invalid_artifact", "Source must be text."
                )
        proposal = json.loads(files["proposal.json"])
        architecture = json.loads(files["architecture.json"])
        method = proposal["method"]
        require(
            isinstance(method, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{1,40}", method)
            and method
            not in {
                "torch",
                "evolve",
                "harbor",
                "nanochat",
                "nexttransformer",
                "nexttransformer_gym",
            },
            "invalid_artifact",
            "Invalid plugin package name.",
        )
        require(
            proposal["schema_version"] == "gym.candidate.v1"
            and proposal["sdk_version"] == CONTRACT,
            "incompatible_contract",
            "Unsupported candidate contract.",
        )
        prefix = f"plugin/{method}/"
        required = {
            "proposal.json",
            "architecture.json",
            prefix + "method.json",
            prefix + "__init__.py",
        }
        require(
            required <= set(files)
            and all(
                name in required
                or (
                    name.startswith(prefix)
                    and name.endswith(".py")
                    and all(
                        part.isidentifier()
                        for part in name[len(prefix) : -3].split("/")
                    )
                )
                for name in files
            ),
            "invalid_artifact",
            "Only metadata and local Python modules inside the declared plugin are accepted.",
        )
        declaration = json.loads(files[prefix + "method.json"])
        validate_metadata(method, declaration, architecture)
        parents = proposal.get("parent_ids", [])
        require(
            isinstance(parents, list)
            and len(parents) <= 16
            and all(isinstance(p, str) for p in parents),
            "invalid_artifact",
            "Invalid candidate lineage.",
        )
        return {
            "files": files,
            "hash": identity(files),
            "configuration_hash": identity(architecture),
            "architecture": architecture,
            "declaration": declaration,
            "method": method,
            "parent_ids": parents,
            "contract": CONTRACT,
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        zipfile.BadZipFile,
        RuntimeError,
    ):
        raise GymError(
            "invalid_artifact", "Malformed candidate source package."
        ) from None
