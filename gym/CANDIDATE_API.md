# Candidate and ModelConfig inputs

An Agent may submit new native model configurations and new native plugin source.
`architecture.json` is the JSON representation of a partial native `ModelConfig`;
it is not an expression language. The same input contract applies to a single
Agent or a team. Agent orchestration does not change Gym's training permissions,
session budget or single numerical worker slot.

## Discover the contract

Call `describe_space` before constructing a candidate:

- `operation_schemas`: the authoritative HTTP operation parameter schemas, also
  used by MCP tools.
- `candidate_contract.model_config_schema`: native field names, JSON types and
  native defaults, derived from the repository's `ModelConfig` definition.
- `candidate_contract`: archive layout, size limits and configuration merge rules.
- `profiles[].model_config_constraints`: fixed values for each execution profile.

The JSON schema checks the input shape. Native `ModelConfig`, the plugin's
`check(config)`, cost admission and the selected profile additionally constrain
which combinations can run. A syntactically accepted upload is not admission.

## Upload source and configuration

Read `endpoint` and `token` from `/run/gym-auth/credential.json`. Send
`Content-Type: application/json` and `Authorization: Bearer <token>` to
`POST <endpoint>/v1/call`:

```json
{
  "schema_version": "gym.api.v1",
  "operation": "submit_candidate",
  "params": {
    "package_base64": "<base64-encoded ZIP bytes>",
    "idempotency_key": "candidate-001"
  }
}
```

Both parameters are required. The idempotency key is a 1–128 character string;
retry identical content with the same key, and use a new key for changed input.
The ZIP and its total uncompressed contents are each limited to 1 MiB, with at
most 32 UTF-8 text files. Include these files, without directory entries:

```text
proposal.json
architecture.json
plugin/research_model/method.json
plugin/research_model/__init__.py
plugin/research_model/<optional local Python modules>
```

`proposal.json` identifies the plugin and candidate lineage:

```json
{
  "schema_version": "gym.candidate.v1",
  "sdk_version": "reference_plugin_v1",
  "method": "research_model",
  "parent_ids": [],
  "hypothesis": "Compare a deeper native model with the same training protocol."
}
```

The first three fields are required. The method must match the plugin directory
and native `method.json.name`, and cannot replace a registered method. Optional
`parent_ids` contains at most 16 candidate IDs belonging to this session.
`hypothesis` is descriptive metadata, not an executable instruction.

Example `architecture.json`, compatible with the example plugin and fixture:

```json
{
  "layers": 4,
  "mlp_width": 128,
  "window_pattern": "L",
  "options": {
    "activation": "silu",
    "output_scale": 1.0
  }
}
```

| Input | Meaning and validation |
| --- | --- |
| Native architecture fields, e.g. `layers`, `mlp_width`, `window_pattern` | Agent-selected values; use the published types and native semantic constraints. Method-specific fields are not necessarily valid for every plugin. |
| `options` | JSON object containing only keys declared in `method.json.options`. Values are strings, finite numbers, booleans or recursively nested lists of these. The plugin checks their meaning; values are not evaluated as expressions. |
| `method` | Optional in `architecture.json`; if present, must equal `proposal.json.method`. |
| `width`, `heads`, `sequence_len`, `vocab_size`, `scale` | Optional; the current profiles fix these values. Explicit values must match `model_config_constraints.fixed`. |
| `routing_seed` | Optional; fixed to 1729. |
| Training settings, paths, commands, environment variables | Not ModelConfig input. They are rejected as unknown fields. |

Integer fields reject booleans and strings. Number fields accept finite JSON
numbers. `null` is not a substitute for an omitted field. Plugin source uses the
repository's native hooks; a `model` hook subclasses `ReferenceLM`. See the
`plugin-contract` reference and [package builder](examples/make_candidate.py).

Omitted fields resolve in this order, later values taking precedence:
native defaults → Gym defaults (`layers=24`, `mlp_width=6144`,
`window_pattern="L"`) → `method.json.shell` → `architecture.json`.
Profile-controlled values are filled in or checked for equality, not silently
overwritten. Options resolve separately: `method.json.options` →
`method.json.debug_options` for a debug profile → `architecture.json.options`.

The HTTP response is `{"schema_version":"gym.api.v1","result":{...}}`.
The candidate result contains `id`, `hash`, `configuration_hash`, `architecture`,
`declaration`, `method`, `parent_ids` and `contract`. Here `architecture` is the
submitted partial config; defaults are resolved for the chosen execution profile.
The SDK and MCP expose this result without the HTTP envelope. Gym stores the
source durably and does not import it during upload.

## Change only the configuration

Keep the plugin unchanged and upload another ZIP with the new `architecture.json`.
For a bounded static comparison, `calibrate_candidate` alternatively takes:

```json
{
  "candidate_id": "<existing candidate ID>",
  "profile_id": "research-fixture",
  "grid": [
    {"layers": 2, "mlp_width": 128, "options": {"activation": "silu"}},
    {"layers": 4, "mlp_width": 128, "options": {"activation": "silu"}}
  ],
  "idempotency_key": "config-comparison-001"
}
```

Each of the 1–32 grid items **replaces the complete architecture object**, rather
than patching the parent. Omitted fields use the merge rules above. The operation
returns a completed static-analysis job; `get_result(job_id=...)` returns
`derived[].candidate` and `derived[].cost`. It does not train these candidates.
Changing source/configuration creates a new immutable candidate. Existing
checkpoints cannot be continued with a changed candidate.

## Request training separately

`quote_experiment` takes required `candidate_id` and `profile_id`, and optional
`target_update` (absolute target, default 1), `checkpoint_id`,
`resource_profile_id`, `server_id` and `gpu_id`. Read the profile's `max_updates`
and `get_resources` for limits and allocated GPU IDs. These are experiment
parameters, never fields inside `architecture.json`.

Then call `run_experiment(quote_id=..., idempotency_key=...)`. Gym owns training,
data, optimizer and accounting. Numerical checks use the same research budget.
`finalize` freezes a validated candidate by ID; it does not accept a new config
or automatically launch final training.

## Errors

HTTP errors contain `error.code`, `hint`, `phase`, `retryable` and `details`;
the SDK raises `GymError`, and MCP preserves the error with `isError=true`.
Malformed archives, unknown config fields and wrong field types return
`invalid_artifact`. Upload-time method or seed mismatches return
`incompatible_contract`. Reusing an idempotency key for different input returns
`conflict`. Profile mismatches and native semantic failures are detected in the
isolated worker; the synchronous cost-audit path currently reports these as
`incomplete_coverage`, while numerical jobs expose their failure through
`get_job`/`get_result`. Inspect these results before proceeding.
