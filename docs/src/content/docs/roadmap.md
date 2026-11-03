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

- Extend the reference training recipe to **400B tokens** and to **larger models**, with the same protocol discipline:
  pinned data, matched compute and a cost ledger.

## Upcoming results

- **Hybrids:** QSA and MoBA hybrids.
- **Audits:** independent run-directory audits for the looped and sparse runs; pilot and resume acceptance evidence for
  the published hybrids.

Their model cards are already in the observatory, marked "Results coming soon". Each result is published through the
same [result pull request](../guides/submit-a-result/) as any contribution.

## Planned

- **Checkpoints and the tokenizer on the Hugging Face Hub.**
- Port the Others implementation (MQA, GQA, MLA) into the current code.

Have an idea or want to help? Open a [discussion](https://github.com/Interactive-Active-AI/NextTransformer/discussions).
