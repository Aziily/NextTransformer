---
title: Evidence
description: Cost proofs that admit configurations, and audits of finished runs.
---

## Cost proofs (`evidence/costs/`)

A proof records, for one configuration, every counted operator of the forward, backward, auxiliary, recomputation and
optimizer work at full size, weighted by the stages of the schedule, and the fingerprint of the source files the method
runs.

| Proof set | Role |
|---|---|
| `plugin-interface-20261031` | **Current.** The configurations of the built-in methods point to their proofs here |
| `arena-v1-stage3-20261103` | Proof of the arena v1 entry AttnRes-Canon (`attnres_full_canon_abcd`), with its SSSL anchor; its configuration points here. `summary.json` also lists the Block AttnRes variant, which is not registered |
| `mhc-logsinkhorn-20261028` | Earlier set after mHC's log-space Sinkhorn-Knopp; counts equal the current set |
| `hybrid-kernels-20261026-release` | Earlier set of the frozen source the published hybrid runs trained on |
| `layout-20261024` | Earlier set from before the package reorganization; model hashes and counts equal later sets |

Every set carries the SSSL anchor proof (`dense.json`). `python -m nexttransformer.protocol.fingerprint` checks that
every published method still matches its proof; the slow test suite reproduces representative proofs from scratch at
full size.

## Run audits (`evidence/audits/`)

Audited runs include a cumulative audit. Use `python -m nexttransformer.audit_run` on a local run directory; frozen submissions can also include a contributor-supplied audit, as for [GSA + QK-norm](../runs/gsa_qknorm/). An audit checks the
ledger, every training, validation and CORE event, the independent final BPB, the final checkpoint and the provenance
of every segment, and records the outcome with the scientific quantities, fingerprints and data layout. Audits never
contain host names, paths or device names.

Published proofs and audits are never edited. A changed method gets a new proof; formal runs keep the proof and source
fingerprints they were admitted with.
