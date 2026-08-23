# design/

Working notes for building torch-preflight. Not user documentation — [README.md](../README.md)
is the user-facing doc, and nothing in here ships in the wheel.

**This directory is public**, and deliberately so: the RFCs and spike write-ups are the
evidence behind the numbers the tool prints, which is most of why anyone should trust an
OOM prediction. Commercial design notes live in the private `torch-preflight-cloud` repo
instead, in the private `torch-preflight-cloud` repo — including the RFC that defines the
free/paid boundary, which is itself on the private side of it.

| Folder | What goes in it | Lifetime |
|---|---|---|
| [TODO.md](TODO.md) | Actionable backlog, grouped by phase. Things we have decided to do. | Items get deleted when done |
| [IDEAS.md](IDEAS.md) | Unfiltered parking lot. No commitment implied. | Grows freely; promote to TODO or RFC when real |
| [rfcs/](rfcs/) | Designs big enough to need agreement *before* code exists | Permanent record, superseded not deleted |
| — [0001](rfcs/0001-vram-estimator.md) | Pre-flight VRAM estimation | Implemented |
| — [0003](rfcs/0003-severity-and-ci-gating.md) | What severity means, and what should fail a build | Implemented |
| [spikes/](spikes/) | Time-boxed experiments answering one uncertain question | Permanent record of what we learned |
| — [0001](spikes/0001-meta-device-activation-capture.md) | Measuring activations on a device that allocates nothing | Complete |
| — [0002](spikes/0002-scanning-real-training-repos.md) | Pointing the tool at real training code | Complete |

Corpus scans live in [`tests/corpus/`](../tests/corpus/): a pinned set of 13 real training
repositories, scanned and diffed against a committed baseline. Run it after a rule change —
it is how 22 rule bugs were found that the unit tests could not see, since our fixtures are
written with the same assumptions as the rules and agree with them by construction.

Measurement scripts live in [`tests/calibration/`](../tests/calibration/), not here — they
are public tooling, since the calibration numbers are only credible if anyone can
reproduce them.

## When to write which

**RFC** — the change touches architecture, adds a dependency, changes a public
interface, or would be expensive to undo. Write it, get agreement, then build.
An RFC is cheap; rewriting a subsystem is not.

**Spike** — we do not know if something is technically possible, and the answer changes
the design. Time-box it, write down the answer, throw the code away. A spike that turns
into production code is a spike that was not a spike.

**TODO** — we already know what to do and how.

**IDEA** — worth remembering, not worth deciding on yet.

## Conventions

- RFCs and spikes are numbered sequentially and never renumbered: `0001-short-slug.md`.
- Every RFC carries a `Status:` line — `Draft` · `Accepted` · `Implemented` · `Superseded by NNNN`.
- Update the status when reality changes. A stale `Accepted` on something we abandoned is
  worse than no document.
- Record decisions *with their reasoning*. Six months from now the reasoning is the only
  part that still has value.
