---
title: Arena task
description: The self-contained task statement to hand to an AI agent, together with a checkout of the repository.
---

This is the task statement for the NextTransformer auto-research arena. Give it to an AI agent, with or without a
research framework, together with a checkout of the repository. The binding rules are the [arena rules](../rules/) and
the [protocol](../../protocol/overview/).

## Goal

Design **one** language-model architecture that, trained from scratch under the frozen training contract, reaches a
lower validation bits-per-byte (BPB) or a higher CORE-22 score at update 46,875 than the models already on the
[leaderboard](../../results/leaderboard/). Both metrics are reported; the leaderboard ranks by either.

## What is fixed

| Control | Value |
|---|---|
| Data | ClimbMix at a pinned revision; one tokenizer (32,768 tokens); 2,048-token context; one global sample order |
| Budget | 49.152B input tokens · 1,048,576 tokens per update · 46,875 updates · seed 42 |
| Compute | Training cost within ±1% of 4.67239832544785 GFLOPs per input token, by the repository's operator ledger, including backward and optimizer work |
| Optimizer | Muon for block matrices, AdamW groups for embeddings, head and scalars; fixed learning rates and schedule |
| Evaluation | Validation BPB every 250 updates; full CORE-22 at 8K, 16K, 24K, 32K, 40K and 46,875 updates |

Parameters, memory and speed are not controlled; they are reported. The budget is training compute, so a model may trade
depth, width, recurrence, sparsity or state for one another, as long as the ledger agrees.

## What you may change

Add your method as a plugin under `nexttransformer/methods/<method>/` ([plugin guide](../../guides/plugins/)), with a
calibrated configuration, its proof set, tests and a model card. Do not edit the data pipeline, evaluation, optimizer,
protocol, cost ledger, trainer, shared model files, `nanochat/`, or other methods' files and evidence.
`python -m nexttransformer.arena --scope origin/main` checks your branch.

## How to work

```bash
uv sync --frozen --extra cpu && source .venv/bin/activate      # or --extra gpu on a CUDA machine
export NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=
python -m pytest tests/test_models_api.py -q                     # build and reload small models
```

- Read the [plugin guide](../../guides/plugins/), the [models API](../../getting-started/models/) and the test plugins
  in `tests/plugins/`. `build_model(debug_config("<method>"))` builds any registered method, plugins included, at debug
  size on a CPU.
- Calibrate your configuration to the cost anchor:
  `python -m nexttransformer.protocol.calibrate --methods <method> --cache .agent-work/cost-cache --output configs --results evidence/costs/<proof-set>`.
  Admission fails if your training cost differs by more than 1% or the ledger meets an operator it cannot charge.
- `python -m pytest -q -n auto` must pass. The registry-wide tests check gradients and that changing future tokens never
  changes earlier logits.
- **Research runs.** `python -m nexttransformer.train --smoke-steps N` trains the full-size model for N updates on the
  formal data stream (use its training loss), or write your own loop around `build_model` under `.agent-work/`, which is
  not committed. A research gym with MCP tools and metered compute ships with Arena v2.

## Rules for the research

1. Any scale, any compute, any number of experiments, as long as you record them.
2. Choose among candidates with training loss or with BPB on the **dev split**, `shard_06541.parquet`.
3. Do not use the validation shard `shard_06542.parquet` or CORE-22 to choose anything.
4. Literature and the internet are allowed; say whether you used them. If your method resembles published work, name it
   and explain the difference.
5. The formal run starts from scratch. Never load research weights into it.

## How to submit

1. Open a proposal issue. Before the formal run, post the fork commit SHA, calibrated configuration and cost proof there.
2. On your fork, train that configuration with `--formal` (8, 16 or 32 GPUs), audit the completed run with `python -m nexttransformer.audit_run`, and export it with `python -m nexttransformer.export_run`.
3. Once the audit is matched, open one PR with the implementation and result. Include the files listed in the [entry rules](../rules/#how-to-enter) and a [disclosure card](../disclosure/) with `status: complete`, checkpoint and log links. Link the preregistration issue.

Run `python -m nexttransformer.arena --check` and `python -m nexttransformer.export_run --check` before opening the PR. Report honestly: failed launches, every candidate you trained, and how much people helped. An entry that loses
to the baselines but is fully disclosed is a useful result.
