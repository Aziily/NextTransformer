"""Canonical HTTP operation definitions, also discovered by the MCP adapter."""

from copy import deepcopy
from jsonschema import Draft202012Validator

from .common import GymError

ID = {"type": "string", "minLength": 1}
CURSOR = {"type": "string", "default": ""}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100, "default": 50}
SCOPE = {"type": "string", "enum": ["published", "session"], "default": "published"}
CANDIDATE = {"candidate_id": ID, "profile_id": ID}
PAGE = {"cursor": CURSOR, "limit": LIMIT}
MUTATIONS = {
    "submit_candidate",
    "validate_candidate",
    "calibrate_candidate",
    "debug_candidate",
    "run_experiment",
    "cancel_job",
    "finalize",
}


def operation(description, properties=None, required=()):
    return {
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": deepcopy(properties or {}),
            "required": list(required),
            "additionalProperties": False,
        },
    }


OPERATION_SCHEMAS = {
    "describe_space": operation(
        "Read supported profiles, candidate_contract (native model config schema and merge rules), per-profile model_config_constraints and authoritative HTTP operation schemas."
    ),
    "get_resources": operation(
        "Observe nodes, discovered/allocated/available GPUs, memory and worker slots. Configuration is operator-owned.",
        {"resource_profile_id": ID},
    ),
    "list_components": operation(
        "List approved architecture source references with pagination.", PAGE
    ),
    "read_reference": operation(
        "Read one approved reference by ID; arbitrary filesystem paths are not accepted.",
        {"resource_id": ID},
        ["resource_id"],
    ),
    "get_candidate": operation(
        "Read this session's immutable candidate metadata and native model config.",
        {"candidate_id": ID},
        ["candidate_id"],
    ),
    "list_candidates": operation("List this session's candidates.", PAGE),
    "submit_candidate": operation(
        "Upload a base64-encoded source-only native plugin ZIP containing proposal.json, architecture.json (native ModelConfig), plugin/<method>/method.json and __init__.py. Read describe_space.candidate_contract first. Config/source changes create a new immutable candidate. No training scripts, weights or command overrides.",
        {"package_base64": {"type": "string", "minLength": 1}},
        ["package_base64"],
    ),
    "inspect_cost": operation(
        "Inspect the selected candidate/profile's parameter count and traced FLOPs before training.",
        CANDIDATE,
        CANDIDATE,
    ),
    "quote_experiment": operation(
        "Quote fixed-profile work to an absolute target update, optionally continuing a checkpoint and choosing an authorized node/GPU. Omit GPU to let Gym select an idle allocated card.",
        {
            **CANDIDATE,
            "target_update": {"type": "integer", "minimum": 1, "default": 1},
            "checkpoint_id": {"type": ["string", "null"]},
            "resource_profile_id": ID,
            "gpu_id": {
                "type": ["string", "null"],
                "description": "Allocated GPU UUID or numeric index as a string; read get_resources first.",
            },
            "server_id": {
                "type": ["string", "null"],
                "description": "Execution node ID returned by get_resources.",
            },
        },
        CANDIDATE,
    ),
    "run_experiment": operation(
        "Consume a quote and reserve its cost from the session's fixed research allocation. Gym queues and executes it.",
        {"quote_id": ID},
        ["quote_id"],
    ),
    "validate_candidate": operation(
        "Run charged bounded loading/numerical/causal/checkpoint/cost checks under an approved profile.",
        CANDIDATE,
        CANDIDATE,
    ),
    "debug_candidate": operation(
        "Run only the selected bounded diagnostic and necessary prerequisites, charging its own bound. Partial diagnostics never grant final admission; no arbitrary commands or test code.",
        {
            **CANDIDATE,
            "check": {
                "type": "string",
                "enum": [
                    "construction",
                    "numerical",
                    "causal",
                    "checkpoint",
                    "cost_coverage",
                ],
            },
            "shape": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        [*CANDIDATE, "check"],
    ),
    "calibrate_candidate": operation(
        "Statically inspect at most 32 native ModelConfig alternatives using the candidate's unchanged plugin. Each grid item replaces architecture.json completely, not as a patch; returns derived candidates and costs. Read describe_space.candidate_contract for types and profile constraints. Does not train or measure GPU throughput.",
        {
            **CANDIDATE,
            "grid": {
                "type": "array",
                "items": {"type": "object"},
                "minItems": 1,
                "maxItems": 32,
            },
        },
        [*CANDIDATE, "grid"],
    ),
    "get_job": operation(
        "Read job state, placement, waiting reason and measured worker execution details.",
        {"job_id": ID},
        ["job_id"],
    ),
    "get_result": operation(
        "Read the completed experiment result and charged budget.",
        {"job_id": ID},
        ["job_id"],
    ),
    "cancel_job": operation(
        "Cancel queued work or request cancellation at a reconciled numerical boundary.",
        {"job_id": ID},
        ["job_id"],
    ),
    "read_events": operation(
        "Read this session's lifecycle/accounting events.",
        {"cursor": {"type": "integer", "minimum": 0, "default": 0}, "limit": LIMIT},
    ),
    "get_budget": operation(
        "Read the fixed total, spent, reserved and remaining research FLOPs."
    ),
    "finalize": operation(
        "Freeze one admitted candidate; Gym retains its source/config. Final training is separately dispatched by the platform.",
        CANDIDATE,
        CANDIDATE,
    ),
    "get_submission": operation(
        "Read this session's frozen submission and final-run status.",
        {"submission_id": ID},
        ["submission_id"],
    ),
    "list_experiments": operation(
        "List published historical experiments or private experiments from this session only.",
        {"scope": SCOPE, **PAGE},
    ),
    "read_experiment": operation(
        "Read experiment results, model/config information, source references and provenance.",
        {"scope": SCOPE, "experiment_id": ID},
        ["experiment_id"],
    ),
}

for name, spec in OPERATION_SCHEMAS.items():
    spec["readOnly"] = name not in MUTATIONS | {"quote_experiment"}
    if name in MUTATIONS:
        spec["inputSchema"]["properties"]["idempotency_key"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
        }
        spec["inputSchema"]["required"].append("idempotency_key")

READS = set(OPERATION_SCHEMAS) - MUTATIONS
VALIDATORS = {
    name: Draft202012Validator(spec["inputSchema"])
    for name, spec in OPERATION_SCHEMAS.items()
}


def validate_parameters(operation, params):
    error = next(VALIDATORS[operation].iter_errors(params), None)
    if error is not None:
        path = ".".join(["params", *(str(item) for item in error.absolute_path)])
        raise GymError(
            "invalid_request",
            f"Invalid {path}: {error.validator} constraint failed; inspect describe_space for the input schema.",
        )
