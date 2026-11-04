# Contributing

Thank you for helping. NextTransformer compares architectures under one fixed
[protocol](https://interactive-active-ai.github.io/NextTransformer/docs/protocol/overview/), so every contribution
arrives as a pull request that keeps that protocol intact. The
[documentation](https://interactive-active-ai.github.io/NextTransformer/docs/) has the full guides; this page is the
short version.

## How to contribute

1. **Open an issue first** for a new method ([proposal form](https://github.com/Interactive-Active-AI/NextTransformer/issues/new/choose))
   or for any change to shared code. Questions and ideas go to
   [Discussions](https://github.com/Interactive-Active-AI/NextTransformer/discussions).
2. **Fork the repository** and work on a branch of your fork.
3. **For a new method**, discuss the run plan in the issue, then calibrate and test the plugin on your fork, complete a `--formal` run, and audit and export its result.
   Only then open one pull request into upstream `main` with the implementation and result together. Link the proposal issue.
4. Run the local checks below before submitting. For Arena entries, preregister the fork commit and calibrated configuration in the proposal issue before the formal run.

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).

## What a pull request contains

| Contribution | Files | Guide |
|---|---|---|
| A new result for an existing method | `results/runs/<method>.json` from `export_run`, its audit `evidence/audits/<method>.json` from `audit_run`, a run-notes page, and the tables refreshed by `docs/scripts/generate.py` | [Submit a result](https://interactive-active-ai.github.io/NextTransformer/docs/guides/submit-a-result/) |
| A new method | A plugin `nexttransformer/methods/<method>/`, a calibrated `configs/<family>/<method>.json` and its cost proof under `evidence/costs/`, tests, a model card `results/methods/<method>.json`, a source-notes entry, and the formal run artifacts from the row above | [Method plugins](https://interactive-active-ai.github.io/NextTransformer/docs/guides/plugins/) |
| Documentation or the website | Pages under `docs/`, code under `website/` | [Documentation README](docs/README.md), [website README](website/README.md) |
| A change to shared code | Agreed in an issue first; see the guide for the audit it needs | [Changing shared code](https://interactive-active-ai.github.io/NextTransformer/docs/guides/changing-shared-code/) |
| An arena entry | The method and audited formal result in one PR linked to the preregistration issue, plus the completed disclosure card with checkpoint and training log links | [Arena rules](https://interactive-active-ai.github.io/NextTransformer/docs/arena/rules/) |

## Local checks

```bash
uv sync --frozen --extra cpu
export CUDA_VISIBLE_DEVICES= NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1
uv run python -m nexttransformer.checks.changes --base origin/main   # what your change needs
uv run python -m pytest -q -n auto                                   # the same fast suite CI runs
```

CI runs the same commands. Results-only and documentation-only pull requests skip the test suite. Slow tests
(full-size cost-audit reproductions and multi-rank layouts) run with `pytest -m slow`; maintainers trigger them in
CI when a change needs them.

## Rules that keep comparisons fair

- Never change the data, tokenizer, token budget, global batch, schedule, seed or evaluation points for one method.
- A new method changes no shared file; its cost proof covers its own plugin and the shared core.
- Published runs, their audits and cost proofs are never edited; a new version of a method gets a new method id.
- Publish complete formal runs only, with their cumulative audit; record failed attempts in the run notes.
- Keep machine details, local paths and credentials out of the repository. CI's public-content check enforces this.
- Keep code, comments and documentation in English (the Chinese documentation is a translation).
