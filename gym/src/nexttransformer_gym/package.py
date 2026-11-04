"""Source-only ingestion and a closed, non-executable FFN expression language.

Python is a serialization syntax here, never imported, compiled or evaluated.
Only one literal assignment is accepted; arithmetic is lowered by trusted code.
"""

import ast
import io
import json
import re
import stat
import zipfile
from .common import GymError, identity, require

LIMIT = 1 << 20
CONTRACT = "reference_plugin_v1"
OPS = {"input": 0, "relu": 1, "square": 1, "silu": 1, "tanh": 1, "add": 2, "mul": 2}


def expression(node, depth=0):
    require(
        depth < 8 and isinstance(node, dict) and set(node) == {"op", "args"},
        "unsupported_operator",
        "Use a bounded FFN expression tree.",
    )
    op, args = node["op"], node["args"]
    require(
        isinstance(op, str)
        and op in OPS
        and isinstance(args, list)
        and len(args) == OPS[op],
        "unsupported_operator",
        "Unsupported expression operator or arity.",
    )
    for arg in args:
        expression(arg, depth + 1)
    return node


def parse_source(source):
    try:
        body = ast.parse(source).body
        require(
            len(body) == 1
            and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
            and body[0].targets[0].id == "FFN_EXPRESSION",
            "incomplete_coverage",
            "This runtime accepts only FFN_EXPRESSION = <literal>; arbitrary Python is not executable.",
        )
        return expression(ast.literal_eval(body[0].value))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        raise GymError("invalid_artifact", "Invalid expression source.") from None


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
            "Invalid candidate method name.",
        )
        require(
            proposal["schema_version"] == "gym.candidate.v1"
            and proposal["sdk_version"] == CONTRACT,
            "incompatible_contract",
            "Unsupported candidate contract.",
        )
        prefix = f"plugin/{method}/"
        require(
            set(files)
            == {
                "proposal.json",
                "architecture.json",
                prefix + "method.json",
                prefix + "__init__.py",
            },
            "invalid_artifact",
            "Only the four declared source and metadata files are supported.",
        )
        declaration = json.loads(files[prefix + "method.json"])
        require(
            declaration
            == {
                "name": method,
                "family": "others",
                "hook": "ffn",
                "language": "ffn-expression-v1",
            },
            "incompatible_contract",
            "Use the documented FFN expression declaration.",
        )
        require(
            isinstance(architecture, dict)
            and set(architecture) == {"layers", "mlp_ratio"}
            and type(architecture["layers"]) is int
            and 1 <= architecture["layers"] <= 64
            and type(architecture["mlp_ratio"]) is int
            and 3 <= architecture["mlp_ratio"] <= 5,
            "invalid_artifact",
            "Architecture must contain layers (1–64) and mlp_ratio (3–5).",
        )
        parents = proposal.get("parent_ids", [])
        require(
            isinstance(parents, list)
            and len(parents) <= 16
            and all(isinstance(p, str) for p in parents),
            "invalid_artifact",
            "Invalid candidate lineage.",
        )
        tree = parse_source(files[prefix + "__init__.py"])
        return {
            "files": files,
            "hash": identity(files),
            "configuration_hash": identity(architecture),
            "architecture": architecture,
            "expression": tree,
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
