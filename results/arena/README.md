# Arena disclosure cards

One card per arena entry: `results/arena/<entry>.json`. The rules are in the [arena rules](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/); the
observatory's arena view and the leaderboard's provenance badges are built from these files.
This page owns the stored fields. Follow [CONTRIBUTING.md](../../CONTRIBUTING.md) and the
[Arena submission guide](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/#how-to-enter) for submission steps.
`python -m nexttransformer.arena --check` validates every card; CI runs it. Start from
[the template](../../docs/public/arena/card.template.json).

| Field | Meaning |
|---|---|
| `id`, `name` | Entry id (the file name) and display name |
| `methodId` | The registered method; its model card is `results/methods/<methodId>.json` |
| `runIds` | Run records in `results/runs/` for this method; required once `status` is `complete` |
| `arenaProtocol` | `arena-v1` |
| `status`, `statusNote` | `admitted`, `complete`, `legacy`, or `retracted`; `legacy` and `retracted` need a note |
| `submitted` | Date the fork commit and configuration were preregistered in the proposal issue, `YYYY-MM-DD` |
| `verification` | `self-reported`, `reproduced`, or `maintainer` |
| `researcher.people` | People involved: `name`, optional `handle`, and `role` |
| `researcher.llms` | Each LLM: `name`, `version` (`null` if unknown), and `role` |
| `researcher.framework` | Auto-research framework: `name` (`none` if none), `version`, `url`, `note` |
| `researcher.involvement`, `involvementNote` | Human involvement level `L0`–`L3` and a short justification |
| `research.start`, `research.end` | Research period, `YYYY-MM-DD` |
| `research.baseCommit` | Repository commit the research started from |
| `research.candidatesTrained` | Distinct designs trained beyond smoke tests |
| `research.formalAttempts`, `formalNote` | Formal launches, including failed ones, and what happened to each |
| `research.gpuHours`, `hardware`, `computeNote` | Research and formal GPU-hours (`null` when not metered), the compute used, and how the figure was obtained |
| `research.internet` | Whether the internet or literature search was used |
| `research.selectionSignal` | What chose among candidates: any of `train-loss`, `dev-bpb`, `validation-bpb`, `core`, `other` |
| `novelty.closestPrior`, `novelty.statement` | Closest prior methods and the difference |
| `artifacts.checkpointUrl`, `logsUrl` | Final checkpoint and training logs; HTTPS links, required for `complete` |
| `artifacts.traceUrl`, `historyUrl`, `codeUrl` | Research trace, a written research history, and the code; optional |

`validation-bpb` and `core` are reserved signals: an `admitted` or `complete` entry cannot list them. A `legacy`
entry lists what it actually used. Unknown values are `null`, never zero.
