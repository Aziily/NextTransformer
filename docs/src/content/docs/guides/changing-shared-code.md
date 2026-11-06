---
title: Changing shared code
description: What happens when a change touches code that published methods run, and how it is reviewed.
---

Most contributions never touch shared code: a new method is a [plugin](../plugins/), and a result is data. Some
changes do need shared code, for example a speed-up of a common module or a bug fix. Because published results were
trained with that code, these changes follow a stricter path.

## 1. Discuss first

Open an issue that explains why the change cannot be a plugin hook. If a missing hook is the problem, maintainers add
the hook instead. Changes to the training contract (the protocol, cost ledger, evaluation, trainer or data pipeline)
from outside contributors are refused by CI until a maintainer has agreed and labelled the pull request
`core-approved`.

## 2. What CI does

- `python -m nexttransformer.checks.changes` lists the published methods whose fingerprinted files changed, and the
  pull request is labelled `needs-audit`.
- `python -m nexttransformer.protocol.fingerprint` fails while a published method's proof no longer matches its code.
  The check stays red until a maintainer has re-audited the affected methods.

## 3. Two kinds of change

**Changes that keep every published method identical** (refactors, speed-ups, fixes that do not affect numerics) need
two checks:

- **Bitwise comparison.** `python -m nexttransformer.checks.golden --base origin/main` builds every method at debug
  size on the base branch and on the pull request, runs one forward and backward pass, and compares the loss, every
  parameter and gradient, the parameter names and the model hash. All must be identical. This covers CPU code paths;
  a change to a GPU-only kernel also needs a bounded GPU check.
- **Execution speedups.** `--speedups` replaces 13 shared functions at run time and refuses to start once any of
  them changes. Editing one does not block your pull request; the fast tests warn, and the patch is reviewed
  separately (see `nexttransformer/runtime/speedups/README.md`).
- **Full audit.** A maintainer runs the *Full audit* workflow, which re-audits the affected methods with the PyTorch
  build the proofs record and compares every operator count with the existing proofs. Only the source fingerprints
  may change. The maintainer then commits the regenerated proofs to the pull request and merges it.

**Changes that alter a published method's numerics or compute** cannot be merged as ordinary pull requests: the
published runs were trained with the old behaviour, and the repository must still reproduce them. Such a change
becomes a new method version (a new method id or a configuration switch, leaving the old one intact), a
maintainer-led fix that annotates the affected results, or, if it changes the contract, a new protocol version.
