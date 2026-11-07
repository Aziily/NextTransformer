---
title: Roadmap
description: What is coming next to NextTransformer.
---

## By November 14, 2026: Arena v2, agent-native research

1. **Auto-research tasks in Harbor format.**
   - The repository runs as the gym's sidecar service and exposes its interfaces to the model.
   - The model develops inside a closed container.
   - The agent can read the repository's full history and search the web to find the best architecture.
2. **Separate research from formal evaluation.**
   - Research uses ClimbMix training data and the dev shard, `shard_06541.parquet`.
   - The formal validation shard, `shard_06542.parquet`, and CORE-22 are reserved for formal reporting, not candidate selection.
3. **The submission is a single file.** The gym then runs the final training itself.

4. **Gym interfaces.**
   - **Connecting:** the container receives an endpoint and token in `/run/gym-auth/credential.json`. Each call is a JSON
     `POST /v1/call` with an operation name and parameters; the Python `GymClient` wraps it without importing any
     training code.
   - **Information:** `describe_space`, `list_components`, `read_reference`, `get_resources` for the search space,
     available components, read-only reference code and resource limits.
   - **Pre-check:** `validate_candidate`, `calibrate_candidate`, `debug_candidate`, `inspect_cost` check a candidate's
     loading, FLOPs/token and parameter budget before it spends research compute.
   - **Fixed experiments:** `run_experiment` trains a candidate under a fixed profile and can continue from a checkpoint;
     `get_job`, `get_result`, `cancel_job` track it.
   - **Budget:** `quote_experiment` prices an experiment before it runs, and `get_budget` shows what remains. Every run
     draws on a fixed research allocation; when it runs out, experiments stop.
   - **Submission:** `finalize` freezes one candidate as the final file.

### Gym experiment plan

- **Models and harness:** Opus, GLM and DeepSeek, each researching independently through Claude Code and Gym MCP. Pin model and client versions before launch.
- **Initial settings:** share the same code/reference snapshot, ClimbMix data order, tokenizer, 2,048-token context, seed 42, 1,048,576 tokens/update and original optimizer/learning-rate protocol. Research uses training loss and dev only; formal validation and CORE-22 remain reserved.
- **Research strategy:** the model chooses its initial architecture, controls, candidate count, experiment lengths, search and stopping strategy. Do not prescribe training steps or search rounds.
- **Model calls:** no spending cap; record tokens and actual cost.
- **Research compute cap:** **1.96×10²⁰ FLOPs per model session**, shared across training, evaluation, checks and failed attempts, without per-candidate quotas. One session for each of the three models gives a total research cap of **5.88×10²⁰ FLOPs**. Final formal training is separate.
- **Budget basis:** Arena v1 Stage 2 recorded 17 attempts, including cancellations and retries. Estimate capacity at 20 anchor-equivalent 2,000-update runs: `20 × 2,000 × 1,048,576 × 4.67239832544785e9 ≈ 1.96×10²⁰ FLOPs`. This sizes the allocation; it neither prescribes 20 experiments of 2,000 updates nor reconstructs v1's actual FLOPs ledger.
- **Scheduling reference:** v1 search and transfer checks used `549.7 + 42.1 = 591.8` GPU-hours. With 20% headroom, allow **about 710 GPU-hours per model, or 89 hours on a comparable eight-GPU setup**. Actual time depends on the model's experiments and throughput; GPU-hours are not an additional hard cap.
- **Goal:** compare the architectures each model discovers autonomously under equal research compute, then verify final submissions under the same formal training protocol.

### Arena results example

Start with a [frontend example](../../arena-example.html): model, harness, research budget and status at the top; candidate dev results and baseline improvements in the middle; expandable architecture changes, formal results and evidence below.
Use historical Arena v1 research data for the example. Mark Opus, GLM and DeepSeek as planned and leave formal scores empty until real run records are connected.

## November 2026: scaling

### Agreed scope

- **4,096-token sequences** for the main pretraining stage.
- **200B-token and 400B-token training budgets**, measured as consumed training tokens with our tokenizer.
- **ClimbMix as the primary dataset**, retaining the existing **32,768-token tokenizer**.
- **Dense and sparse MoE models** at each token budget.
- **Validation BPB and CORE-22** remain part of the evaluation suite.

This is the scope of the next training recipe. It will have its own versioned protocol, with pinned data and
explicit compute accounting. The existing 49.152B-token, 2K benchmark remains the reference for published results.

### Open for community feedback

The [scaling RFC in Discussion #22](https://github.com/Interactive-Active-AI/NextTransformer/discussions/22) contains the candidate model configurations and asks for advice on:

- **Data composition:** whether 4K pretraining should supplement ClimbMix, which datasets to use, and how to mix or
  sample longer documents.
- **Dense/MoE configurations:** model sizes, depth and width, expert counts and routing, and whether to add
  matched-compute comparisons. The candidate parameter counts and model shapes are still under review.
- **Evaluation beyond CORE:** **HELMET's RAG tasks** and **RULER** are the initial candidates for controlled 2K/4K
  evaluation on base models. The final benchmark selection and any later context extension remain open.

Community feedback will help prioritize pilot ablations and finalize the recipe before the full runs.

## Upcoming results and evidence

- **Hybrid results:** GDN + QSA. Its model card is already in the observatory, marked
  "Results coming soon". Each result is published through the same [result pull request](../guides/submit-a-result/)
  as any contribution.
- **Audits:** independent run-directory audits for the looped and sparse runs; pilot and resume acceptance evidence
  for the published hybrids.

## Planned

- **Checkpoints and the tokenizer on the Hugging Face Hub.**
- Port the Others implementation (MQA, GQA, MLA) into the current code.

Have an idea or want to help? Join the [scaling discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions/22) or open another
[discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions).
