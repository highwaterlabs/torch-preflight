<p align="center">
  <img src="https://raw.githubusercontent.com/highwaterlabs/torch-preflight/main/assets/logo.svg" width="88" alt="torch-preflight logo"/>
</p>

<h1 align="center">torch-preflight</h1>

<p align="center"><b>The linter that understands autograd — and the VRAM estimator that
tells you what will fit.</b></p>

<p align="center">
  <a href="https://github.com/highwaterlabs/torch-preflight/actions/workflows/ci.yml"><img src="https://github.com/highwaterlabs/torch-preflight/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://pypi.org/project/torch-preflight/"><img src="https://img.shields.io/pypi/v/torch-preflight" alt="PyPI"/></a>
  <a href="https://pypi.org/project/torch-preflight/"><img src="https://img.shields.io/pypi/pyversions/torch-preflight" alt="Python versions"/></a>
  <a href="https://github.com/highwaterlabs/torch-preflight/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/torch-preflight" alt="License"/></a>
</p>

<p align="center">
  <a href="https://github.com/highwaterlabs/torch-preflight/tree/main/docs"><b>Docs</b></a> &middot;
  <a href="https://github.com/highwaterlabs/torch-preflight/blob/main/docs/rules.md"><b>Rules</b></a> &middot;
  <a href="https://github.com/highwaterlabs/torch-preflight/blob/main/docs/vram-estimation.md"><b>VRAM estimation</b></a> &middot;
  <a href="https://github.com/highwaterlabs/torch-preflight/blob/main/docs/cli.md"><b>CLI</b></a>
</p>

A static analyzer for PyTorch training code. It catches VRAM leaks and silent convergence
bugs at commit time, projects peak memory for a training run or a serving deployment before
you rent the GPU, and can fail the run at step 0 instead of OOMing at step 400.

No GPU. No PyTorch. It never imports or executes your code.

```console
$ torch-preflight check train.py

train.py
  7:19  error   TG001 (CRITICAL_OOM)
  `losses.append(...)` stores a tensor that is still attached to the autograd graph;
  every iteration's graph is retained in VRAM.
    7 │     losses.append(loss)
      │                   ^^^^
  help: Use `.item()` to keep just the scalar value, or `.detach()` to keep the tensor
        without its graph.
  fix:  add .detach() (run with --fix)

Found 1 error in 1 file(s).
```

<p align="center"><i>One line, one wasted GPU hour. Caught in milliseconds, before it runs.</i></p>

- 🔍 **13 rules for bugs `ruff` and `flake8` cannot see** — retained autograd graphs, missing
  `zero_grad()`, unscaled gradient accumulation, DDP without a `DistributedSampler`, a model
  stuck in `eval()` after validation, doubled softmax, unseeded runs, GPU sync in a hot loop
- 🧮 **VRAM estimation that covers the whole job** — training, encoder-decoder, and
  autoregressive **generation with the KV cache**, plus the change that would make it fit
- 🛡️ **`VRAMGuard`** — fails a run at step 0 rather than OOM at step 400, measuring your
  model's real activation memory without allocating a byte
- 🛠️ **Autofixes** via concrete syntax tree rewrites, so formatting and comments survive untouched
- 📊 **Measured, not guessed** — every constant calibrated against real hardware, **3.7% mean
  error** versus measured peaks
- 🤫 **Quiet on real code** — **20 findings across PyTorch's own 2,285 files**, roughly one per
  hundred, every one read by hand
- ⚡ **No GPU and no PyTorch required** — pure static analysis over [LibCST](https://github.com/Instagram/LibCST); a CI job asserts torch is never imported
- 🐍 Python 3.9–3.13, `pyproject.toml` config, pre-commit hook, GitHub Action, SARIF output

`ruff` and `flake8` understand Python. They don't understand autograd graphs, gradient
accumulation, or what `num_workers=0` does to eight GPUs waiting on one CPU. torch-preflight
is built for the bugs that only cost money once you're paying for a GPU.

## Table of contents

- [Getting started](#getting-started)
- [The line that costs you a GPU hour](#the-line-that-costs-you-a-gpu-hour)
- [Will this fit on the GPU I'm about to rent?](#will-this-fit-on-the-gpu-im-about-to-rent)
  - [Serving, not just training](#serving-not-just-training)
  - [Fail at step 0, not step 400](#fail-at-step-0-not-step-400)
  - [It reads your config, not just your code](#it-reads-your-config-not-just-your-code)
- [Why you can trust the numbers](#why-you-can-trust-the-numbers)
- [Integrations](#integrations)
- [Documentation](#documentation)
- [What stays free](#what-stays-free)

## Getting started

```bash
pip install torch-preflight
```

```bash
torch-preflight check ./src/                        # lint a tree
torch-preflight check ./src/ --fix                  # apply the safe fixes
torch-preflight estimate train.py --gpu a100-80gb   # will this training run fit?
torch-preflight estimate --model llama-3-8b --gpu a100-80gb \
    --generate --batch-size 16 --max-context 8192   # will this deployment fit?
torch-preflight explain TG003                       # why a rule exists, and what it costs
```

The base install has no heavy dependencies. `torch-preflight[hub]` adds Hugging Face
architecture lookup; `torch-preflight[vram]` adds exact meta-device profiling.

## The line that costs you a GPU hour

```python
for batch, targets in loader:
    loss = criterion(model(batch), targets)
    losses.append(loss)          # ← nothing backwards this: every activation stays in VRAM
```

You have written this. Everyone has. `loss` still carries its computational graph, so
appending it retains every intermediate activation on the way to it — and the next step's,
and the next. Memory climbs linearly until CUDA gives up, hours in.

**Where most tools would stop, and get it wrong.** Move one line and the same code costs
almost nothing:

```python
for batch, targets in loader:
    loss = criterion(model(batch), targets)
    loss.backward()              # frees the saved tensors as it traverses
    losses.append(loss)          # ← now retains graph *nodes*, not activations
```

`backward()` releases each node's saved tensors on its way through, so the second version
holds host-side bookkeeping rather than VRAM. Measured on a 13×256 MLP
([the harness is in the repo](https://github.com/highwaterlabs/torch-preflight/blob/main/tests/calibration/measure_retention.py)):
**560 KiB of activations retained per iteration** in the first version, **0 KiB** in the
second, with the same ~30 graph nodes held either way. Same one-line fix, two very different
bugs — so torch-preflight reports the first as an error and the second as a warning, instead
of claiming both will OOM your GPU.

**Why this is hard:** `losses.append(x)` is only a bug when `x` carries a graph. torch-preflight
runs a dataflow pass to find out, tracing values across assignments, arithmetic, tensor
methods and function scopes, and refusing to propagate through `.detach()`, `.item()` or
`argmax`. So `losses.append(loss.item())` stays silent, and so does anything inside
`torch.no_grad()`. It also tracks whether a backward pass still *needs* what you stored —
pipeline-parallel schedules and chunked loss modules hold graphs on purpose, and telling them
to `.detach()` would break the training run rather than speed it up. A linter that
pattern-matched on `.append(` would be unusable.

See [all 13 rules →](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/rules.md)

## Will this fit on the GPU I'm about to rent?

```console
$ torch-preflight estimate finetune.py --gpu a100-80gb

Model      llama-2-7b  (arch-snapshot)   6.74 B params
Config     amp · AdamW · batch 4 · seq 2048
Read from  batch_size=finetune.py:9, optimizer=finetune.py:7, seq_len=finetune.py:13

  weights            25.10 GiB  ███
  gradients          25.10 GiB  ███
  optimizer state    50.21 GiB  ██████
  autocast cache     12.55 GiB  ██
  activations        67.42 GiB  ████████
  CUDA context         135 MiB  █
  fragmentation      18.94 GiB  ██
  ──────────────────────────────────────────────
  projected peak    199.45 GiB   (179.51 GiB – 219.40 GiB)

Target     NVIDIA A100 80GB (78.0 GiB usable)   →   256% of capacity   ✗ OOM

What would make it fit:
  ✗  − 35.36 GiB  →  164.09 GiB   flash attention / SDPA
       mathematically equivalent, removes the O(seq²) attention term
  ✗  − 66.30 GiB  →  133.15 GiB   gradient checkpointing
       same result, roughly 30% slower
  ✗  − 41.61 GiB  →  157.84 GiB   8-bit AdamW (bitsandbytes)
       quantised optimizer state, minimal quality impact
  ✗  −112.56 GiB  →   86.89 GiB   flash attention + checkpointing + 8-bit AdamW
                                  + halve micro-batch
       even stacked together these do not fit; this needs a larger GPU, more
       devices, or a parameter-efficient method such as LoRA
```

A single 80GB A100 is the wrong tool for a full 7B fine-tune at sequence 2048. Better to
learn that now than after the instance is running.

Every field is **read out of your script** — model, batch size, sequence length, precision,
optimizer, sharding — and the report says which line each came from, so you can check it.
Nothing is imported or executed. **41 architectures** ship built in, **23 GPUs** and
**34 cloud instances** are known by name (`--gpu p4de.24xlarge` works), and anything else is
measured exactly on PyTorch's meta device without allocating a byte.

Every other estimator stops at the number. The list of what to *change* is the part you
actually wanted.

### Serving, not just training

Generation is a different memory shape, and `--generate` models it:

```console
$ torch-preflight estimate --model llama-3-8b --gpu a100-80gb \
      --generate --batch-size 16 --max-context 8192 --dtype pure-bf16

Config     pure-bf16 · batch 16 · context 8192 · generation (KV cache)

  weights            14.96 GiB  ███████████
  activations           20 MiB  █
  KV cache           16.00 GiB  ████████████
  ...
  projected peak     32.68 GiB   (29.41 GiB – 35.95 GiB)

Target     NVIDIA A100 80GB (78.0 GiB usable)   →   42% of capacity   ✓ FITS
```

The KV cache is usually the term that decides the answer, and it is where grouped-query
attention pays off: llama-3-8b's 8 KV heads against 32 query heads make its cache a quarter
of the multi-head equivalent. Llama-2-7b, which has none, needs **64 GiB of cache against
12.6 GiB of weights** at batch 32 and 4096 tokens.

Decoding also collapses the activation term — one token attends against the cache, so the
O(context²) attention matrix never materialises.

### Fail at step 0, not step 400

```python
from torch_preflight import VRAMGuard

with VRAMGuard(model, optimizer=optimizer, batch_size=32, image_size=224):
    train()
```

Parameters, gradients and optimizer state come from the live model and are exact.
Activations are **measured from your module**, by running one forward pass against
meta-device parameters — that allocates nothing, never touches your model, and costs
milliseconds. It raises only when the run cannot fit even at the optimistic end of the
interval; aborting a job on a guess would be worse than the OOM.

### It reads your config, not just your code

DeepSpeed ZeRO stage and CPU offload are parsed out of the JSON or dict your script points
at — reading JSON is not executing code — so a ZeRO-3 run with `offload_optimizer` is
charged for what actually sits on the device. T5 and Whisper are modelled as
encoder-decoders, with the cross-attention term a decoder-only formula cannot express.

See [VRAM estimation →](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/vram-estimation.md)

## Why you can trust the numbers

Memory estimators are easy to write and easy to be quietly wrong about. So:

|  |  |
|---|---|
| **Constants are measured** | Activation coefficients from `saved_tensors_hooks` on the meta device; allocator behaviour and CUDA context from a real GPU. Measurement showed the published Megatron constants are a midpoint of two regimes — models with dropout retain 3× the attention tensors — so Llama-class models are charged the cheaper rate they actually pay. |
| **Projections are checked** | **3.7% mean absolute error** against measured peaks for GPT-2, BERT, DistilBERT and ResNet-50 on a T4. Harness and fixtures in [`tests/calibration/`](https://github.com/highwaterlabs/torch-preflight/tree/main/tests/calibration/), so you can re-run them. |
| **Gaps are stated, not papered over** | `offload_param` streams parameters in, so the resident set is smaller than the weights term — we have not measured it, so the report says the real peak is lower rather than inventing a fraction. Grouped-query attention does *not* reduce training activations (measured: `repeat_kv` materialises full-size K/V), so it is not modelled as if it did. |
| **It refuses to guess** | An unrecognised model reports `UNKNOWN` and widens the interval rather than inventing a parameter count. Verdicts are bands with an error range, never a fabricated "95% risk" score. |
| **It stays quiet** | **20 findings across PyTorch's own 2,285 files** — about one per hundred — every one read by hand. Triage is also where most of our own bugs come from: three findings we had filed as "intentional, so suppress it" turned out to be the rule misunderstanding deferred backward, and a scan of seven training repos found nine more. Each is now regression-tested against the file that exposed it. |

**435 tests.** A typical project lints in well under a second; PyTorch's entire 2,285-file
source tree takes about four minutes with all 13 rules.

## Integrations

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/highwaterlabs/torch-preflight
    rev: v0.5.0
    hooks:
      - id: torch-preflight
```

```yaml
# .github/workflows/lint.yml
- uses: highwaterlabs/torch-preflight@v0
  with:
    paths: src/
    format: github      # inline PR annotations
```

SARIF output feeds GitHub code scanning; JSON feeds everything else. Set `target_gpu` in
`pyproject.toml` and CI fails on a projected OOM before the job is ever submitted.

By default the build fails on **errors** — wrong results and OOM. If the repository's product
*is* training runs, add `fail-on: warning` to catch the smaller defects too: retained graphs
whose backward has already run, per-element device syncs, unseeded runs. Tuning observations
like `num_workers=0` are notes and never fail a build, which is what keeps that setting
usable. See [RFC 0003](design/rfcs/0003-severity-and-ci-gating.md).

See [CI integration →](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/ci.md)

## Documentation

| | |
|---|---|
| [Rules](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/rules.md) | All 13 rules, and the false positives deliberately suppressed |
| [VRAM estimation](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/vram-estimation.md) | Custom architectures, CI gating, `VRAMGuard`, accuracy |
| [CLI reference](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/cli.md) | Commands, flags, exit codes, autofixes |
| [Configuration](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/configuration.md) | `pyproject.toml` and inline suppression |
| [CI integration](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/ci.md) | GitHub Action, pre-commit, SARIF |
| [Architecture](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/architecture.md) | How the analysis pipeline works |
| [Development](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/development.md) | Tests, adding a rule, roadmap |

Design notes live in [`design/`](https://github.com/highwaterlabs/torch-preflight/tree/main/design/), including the
[RFC](https://github.com/highwaterlabs/torch-preflight/blob/main/design/rfcs/0001-vram-estimator.md) behind the estimator and the
[spike](https://github.com/highwaterlabs/torch-preflight/blob/main/design/spikes/0001-meta-device-activation-capture.md) the cost model rests on.

## What stays free

MIT licensed. These are commitments, not just current state:

- **Every rule that has ever shipped free stays free.**
- **The estimator, the remediation solver and `VRAMGuard` stay complete** — not a demo tier.
- **The rule API stays open**, so anyone can write and ship their own rules.
- **The calibration method and data stay public and reproducible.** Numbers are only worth
  trusting if you can check them.

A hosted service may come later for things that genuinely need a server or a team.
Nothing above is part of that.

## Contributing

Issues and pull requests are welcome. Adding a rule is one file plus a `@register`
decorator — see [development](https://github.com/highwaterlabs/torch-preflight/blob/main/docs/development.md) for the walkthrough and the test
conventions.

## License

MIT — see [LICENSE](https://github.com/highwaterlabs/torch-preflight/blob/main/LICENSE).
