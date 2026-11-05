---
title: Roadmap
description: What is coming next to NextTransformer.
---

## By November 14, 2026: agent-native research

- **Arena v2** with full support for autonomous agent research: a research gym with MCP tools to submit, calibrate,
  train and score candidates on the dev split while validation and CORE stay sealed.
- **Metered compute and research traces** for every attempt, so research cost is disclosed automatically.
- **Self-improving research loops**: agents improve how they search, not only what they find.
- **Leaderboards by researcher**: which LLM and framework found what, at what research cost.

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
