---
title: Arena rules
description: Arena Protocol v1, the rules for the research that precedes an entry's formal run and what it must disclose.
---

This is **Arena Protocol v1** (identifier `arena-v1`). The entered model is trained and evaluated under the unchanged
[reference protocol](../../protocol/overview/); these rules add only what governs the research before that run and what
an entry must disclose. Machine-readable values live in `nexttransformer/arena.py`.

The arena is trust-based in this version. Entrants run their own research and their own formal training; maintainers do
not re-run entries by default. Pinned data, a fixed seed, restored training state, published checkpoints and complete
disclosure make claims inspectable.

## An entry

An entry is a **researcher** and the **artifact** it produced.

- The researcher is the LLM or LLMs, the framework (or none), and the people involved, with their level of involvement.
- The artifact is one registered method with an admitted configuration and, once trained, one formal run.
- Each entry has one disclosure card, `results/arena/<entry>.json` ([fields](../disclosure/)).
- A method version has at most one scored run. Changing the method's semantics after admission makes a new method id.

## What an entry may change

| May add | Must not change |
|---|---|
| A method plugin `nexttransformer/methods/<method>/` | Anything the training contract fixes: data, tokenizer, packing, budget, batch, schedule, seed, evaluation points |
| A calibrated `configs/<family>/<method>.json` and its proof set under `evidence/costs/` | `nexttransformer/{protocol,evaluation,runtime,training,checks,native}/` and the command-line entry points |
| Tests under `tests/` | `nexttransformer/models/backbone.py` and `nanochat/` |
| A model card `results/methods/<method>.json` and a source-notes entry | Existing run records, proofs and audits |
| The run record, its audit under `evidence/audits/`, and a run-notes page | The protocol pages, these rules, the arena tooling, CI and the website |
| The entry's own disclosure card | Another entry's card |

A plugin needs no edit to shared model files; if a method truly needs one, a maintainer reviews it, and it invalidates
the proofs of every method that runs the edited file. `python -m nexttransformer.arena --scope origin/main` checks these
rules, and `python -m nexttransformer.protocol.fingerprint` confirms that every published method still matches its proof.

## Research

1. **Any scale, any compute, any number of experiments.** Nothing about the research is limited; all of it is disclosed.
2. **Select with the dev split or training loss.** The dev split is `shard_06541.parquet`, the last training shard in
   sorted order; a formal stream never reaches it.
3. **The validation shard and CORE-22 are reserved.** `shard_06542.parquet` and the CORE-22 tasks belong to the formal
   run. Do not use them to choose among candidates, settings or seeds.
4. **Literature and the internet are allowed.** Disclose whether they were used. Rediscovering a published method is a
   valid entry; the card names the closest prior methods and the difference.
5. **Research never supplies weights.** The formal run starts from scratch.

## How to enter

1. Open a proposal issue describing the method and run plan. Before the formal run, post the fork commit SHA, calibrated configuration and cost proof (within ±1%) in that issue. This dated record freezes the method for scoring.
2. Train the preregistered configuration on the fork with `--formal`. Audit the completed run and export its record.
3. Once the cumulative audit is `matched`, open one PR to upstream `main`. Link the preregistration and include the plugin, configuration, proof set, tests, model card, run record, audit and run notes. Submit the disclosure card with `status: complete` and links to the final checkpoint and training logs. The entry joins the leaderboard after review and merge.

The formal run uses the admitted configuration: its record's `model_hash` must equal the configuration's. Every formal
launch counts toward `formalAttempts`, including launches that failed; failed launches are recorded in the run notes.

## Disclosure

The card records, at minimum:

- **Researcher:** each LLM and version, the framework and version (or `none`), the people involved, and the involvement
  level below.
- **Research:** the period, candidates trained, formal attempts, GPU-hours (`null` when not metered), whether the
  internet was used, and the selection signal.
- **Novelty:** the closest prior methods and the difference.
- **Artifacts:** the final checkpoint and the training logs (required for `complete`); the research trace is encouraged.

Missing measurements are `null`, never zero.

| Level | Human involvement |
|---|---|
| **L0** | Fully autonomous: people start the task and provide compute, then do not intervene until submission |
| **L1** | Operations only: people fix environments, clusters or launches, and make no research decision |
| **L2** | People choose among the agent's proposals or give directional hints |
| **L3** | People lead the idea or write method code; the agents assist |

## Status and verification

| Status | Meaning |
|---|---|
| `admitted` | Configuration admitted on the fork; formal result pending |
| `complete` | Formal run published, complete and matched; the entry ranks on the leaderboard |
| `legacy` | Predates this protocol or does not meet it; shown with full disclosure and a reason, never ranked as an arena entry |
| `retracted` | Withdrawn or found invalid; kept visible with the reason, never ranked |

Each card also says who produced the result: `self-reported` (the default), `reproduced` (an independent party re-ran
the formal run, or replayed a prefix of it and matched the logged training loss), or `maintainer`.

## Ranking

Arena entries are ranked on the main leaderboard by CORE only, with the same rules as every other run: measured values
at update 46,875, nothing interpolated, incomplete runs unranked. Their validation BPB is measured and shown but not
ranked. The leaderboard marks each run's provenance and can filter by it. Research budgets are disclosed, not
controlled, so comparisons between researchers are descriptive.

## Checks

- `python -m nexttransformer.arena --check` validates every card and the records it names; CI runs it.
- `python -m nexttransformer.arena --scope origin/main` classifies a pull request's changed files as allowed, needing
  review, or blocked.
- The existing checks still apply: `export_run --check`, configuration admission, and the test suite.

## Versioning

Changing any rule here makes `arena-v2`. Cards keep the version they were entered under, and existing entries are never
rewritten.
