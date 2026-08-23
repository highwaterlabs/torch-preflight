# Changelog

All notable changes to torch-preflight are recorded here. This project follows
[semantic versioning](https://semver.org/).

## [0.5.0] — 2026-08-24

Accuracy. 0.4.0 was validated against PyTorch's own source; this release is what happened
when the tool was pointed at **thirteen repositories whose product is training runs** —
5,273 files across `pytorch/examples`, `pytorch/tutorials`, `torchtune`, `litgpt`, `trl`,
`LLaMA-Factory`, `transformers/examples`, `nanoGPT`, `minGPT`, `composer`, `peft`, `OpenRLHF`
and `axolotl` — and every finding was read by hand.

That found **one** bug worth reporting upstream and **seventeen in these rules**. Errors on
the two corpora fell from 88 to 16, with no true positive lost.

### ⚠️ Behaviour change

- **Unseeded runs are now a `note` unless the seeding is partial.** Seeding often lives in
  the launcher or the job scheduler rather than the training script, and some runs
  deliberately want variance — that is a choice this tool cannot see, so it no longer claims
  the code is defective. **Partial** seeding — torch seeded, NumPy not — stays a `warning`,
  because the intent is visible in the code and a generator is escaping anyway. Under
  `fail_on = "warning"` the first case no longer gates a build.
- **Builds that failed on a false positive will now pass.** Worth stating plainly rather
  than filing under "fixed": most of the changes below remove findings.

### Fixed

Grouped by the kind of mistake, because the kinds repeat.

**A name was trusted over the code next to it.**
- `AutoTokenizer.from_pretrained(...)` and feature extractors counted as models, so
  `tokenizer(x["question"])` was a forward pass and TG002 reported a missing `no_grad`
  around *tokenisation*.
- `DiffusionPipeline.from_pretrained(...)` likewise, so appending the **PIL image** it
  returns was reported as a retained autograd graph.
- `accelerator.backward(loss)` seeded `accelerator` itself as a live tensor. Accelerate,
  Fabric and DeepSpeed engines invert the usual shape — the tensor is the argument.
- A local helper's `return` is now read before trusting the caller's variable name.
  `train()` returning `loss.item() / n` hands back a float, whatever the caller calls it.

**The rule did not know what it was about.**
- **TG007 flagged the exact code its own hint recommends.** Every finding was
  `correct += (predicted == labels).sum().item()` in a validation loop. Its batch-loop
  exemption matched iterable *names*, so it missed `dev_iter` and `valloader`; it now
  requires evidence of per-element iteration.
- TG013 treated `.to("cpu")` as a redundant upload, and told a checkpointing routine to
  hoist the `.to(device)` that restores the model afterwards.
- TG002 reported that a function "never calls `.backward()`" with the call nine lines below,
  in an adversarial-attack tutorial that backwards through its test loop on purpose.

**Not training code at all.**
- **Test suites.** `model.eval()` followed by a forward inside a pytest test was reported;
  the exemption existed but only applied to the function *name*. Now keyed on the module.
- **Checkpoint plumbing.** `state_dict[f"{name}.lora_A.weight"] = lora_A` and
  `model_args[k] = checkpoint_model_args[k]` are startup surgery, not a training loop. This
  is why `nanoGPT/train.py` reported two `CRITICAL_OOM` errors; it now scans clean.
- **Tensor buffers.** Filling a preallocated `torch.zeros(...)` chunk by chunk is the
  memory-efficient idiom, not an accumulating container.
- **Triton kernels.** `@triton.jit` functions contain no autograd; `dv += tl.dot(...)` in a
  hand-written flash-attention backward is arithmetic.

**Retention a backward pass depends on**, continued from 0.4.0. A holder is exempt when a
value derived from it is returned, not only when the container itself is — `logps =
torch.cat(all_logps); return logps` hands the caller a graph the objective backwards.

**Values produced under `torch.no_grad()`** no longer propagate a graph. The standard
evaluation loop wraps only the forward and uses the result after the block, and the check
was positional.

### Added

- [`tests/corpus/scan.py`](tests/corpus/scan.py) — thirteen pinned repositories, 3,455
  files, diffed against a committed baseline. Two guards, both of which caught real bugs
  while being written: file counts are asserted (a pruned checkout once scanned **0 files**
  and reported every finding as removed), and the summary always states that removals are
  ambiguous — a fixed false positive and a silenced true positive look identical.
- [Spike 0002](design/spikes/0002-scanning-real-training-repos.md), the full write-up.

### Known gaps

Two false-positive causes are understood and deliberately unfixed, each with one finding and
a filed issue: container element types ([#55](https://github.com/highwaterlabs/torch-preflight/issues/55))
and the `label` naming heuristic on non-tensors ([#56](https://github.com/highwaterlabs/torch-preflight/issues/56)).
Neither has a live failing case to verify a fix against, and changing a heuristic with
nothing observable to check is how a false negative was introduced during this work.

## [0.4.0] — 2026-08-18

Two rounds of triaging real training code. torch-preflight was run over seven repositories
whose product is training runs — `pytorch/examples`, `pytorch/tutorials`, `torchtune`,
`litgpt`, `trl`, `LLaMA-Factory` and `transformers/examples/pytorch`, 1,615 files — and every
TG001, TG003, TG005 and TG014 finding was read by hand. That produced **one** bug worth
reporting upstream and **nine in this tool**. Errors across those repos fell from 63 to 32
with no new findings.

### ⚠️ Breaking: what fails your build has changed

If you rely on `torch-preflight` to fail CI, read this before upgrading.

- **TG001 on an already-backwarded tensor is now a `warning`, not an `error`.** The classic
  `loss.backward(); losses.append(loss)` no longer fails a build under the default
  `fail_on = "error"`. This is a correction, not a relaxation: `backward()` frees each node's
  saved tensors as it traverses, so that shape retains **0 bytes** of activations. The
  measurement is in `tests/calibration/measure_retention.py` and
  [RFC 0003](design/rfcs/0003-severity-and-ci-gating.md) §2. Storing a tensor that *nothing*
  backwards is still an error, and still the CUDA OOM the rule was written for.
- **TG004 is now a `note`**, and notes never gate a build.
- To restore the previous behaviour for either, set `fail_on = "warning"`, or re-level the
  individual rule:

  ```toml
  [tool.torch-preflight.severity]
  TG001 = "error"
  ```

### Added

- **Severity has a definition** ([RFC 0003](design/rfcs/0003-severity-and-ci-gating.md)), on
  one axis — what happens if you ship this. `error` means the run is wrong or dies; `warning`
  means a real defect with a bounded blast radius; `note` means the code is correct but
  untuned. The deciding question is "is this code defective, or merely untuned?", and it is
  written down in `docs/rules.md` so future rules are not levelled by feel.
- `fail_on = "warning"` is now recommended in the README and `docs/ci.md` for repositories
  whose product is training runs. TG004 becoming a note is what makes that setting usable —
  it was 207 of the 318 findings across the seven repos, 13% of every file scanned.
- `tests/calibration/measure_retention.py`, which measures what a retained graph actually
  holds. On a 4-layer transformer: **186 MiB of activations per iteration** when nothing
  backwards them — enough to OOM an 80 GB A100 in ~440 steps — against **0 KiB** when
  something does, with ~15 KiB/iteration of host bookkeeping either way.

### Fixed

- **TG001 no longer fires on retention a backward pass depends on.** Rewritten around
  reachability rather than syntax, because `torch.stack(losses).mean()` is a throwaway
  reduction when it is logged and load-bearing when it becomes the training loss, and the two
  are written identically. Pipeline-parallel schedules, chunked loss modules that accumulate
  per-chunk losses and return the total, RL objectives that stack per-step losses, and
  `torch.distributed.autograd` chains are all quiet now. Following the old hint would have
  broken those runs rather than saving memory.
- **TG005 no longer reads an attention softmax as the model's output activation.**
  `pytorch/examples/gat/main.py` binds `self.softmax = nn.Softmax(dim=1)` to normalise
  attention coefficients and correctly ends in `F.log_softmax`. Attention softmax appears in
  every transformer and GNN, so constructing the layer is not evidence; final position in an
  `nn.Sequential` is.
- **TG005 reads the layer class rather than the attribute name.**
  `self.softmax = nn.LogSoftmax(dim=1)` feeding `NLLLoss` is correct code — and is what
  PyTorch's own char-RNN tutorial does, which we were reporting as a convergence bug.
- **TG014 recognises gradient-level rescaling.** torchtune weights each micro-batch loss by
  its token count and applies `scale_grads_(params, 1/num_tokens)` before `step()`, a
  token-mean across uneven micro-batches. Our hint would have told them to divide again and
  shrink the gradient by the accumulation factor. Also resolved when the rescaler is called
  through an alias.
- **Values produced under `torch.no_grad()` no longer propagate a graph.** The standard
  evaluation loop wraps only the forward and uses the result after the block, where a
  positional check cannot see it.

### Changed

- The README no longer claims a retained graph costs "every intermediate activation" in all
  cases; it shows both shapes and the measured difference between them. The previous wording
  overstated the tool's own headline example.
- Corrected stale figures: PyTorch's source produces **20** findings across 2,285 files, not
  23, and "every one triaged as deliberate" was no longer true once three of them turned out
  to be our own bug.
- The documented pre-commit `rev:` pins had drifted to `v0.1.0` and `v0.2.0` and now track the
  current release.

### Known gaps

Two false-positive causes are understood, measured and deliberately not fixed here, because
each needs a design change rather than a patch — both are recorded in
[IDEAS.md](design/IDEAS.md) with the file and line that exposed them: within-file callee
return types, and flow sensitivity within a scope. The latter is why the `no_grad` fix above
does not yet clear the standard Hugging Face evaluation loop.

## [0.3.1] — 2026-08-17

**No changes to the Python package.** This release exists so the GitHub Action can be
published to the Marketplace from a tag whose tree is correct — `v0.3.0` predates both
fixes below.

### Fixed

- **The Action pinned `actions/setup-python@v5`**, which targets the deprecated Node 20
  runtime. The earlier pin bump updated `.github/workflows/*.yml` and missed `action.yml`
  at the repo root — the composite action users actually execute — so our own CI was clean
  while every consumer of the Action got a deprecation warning in their logs.
- **`uses: highwaterlabs/torch-preflight@v0` did not resolve.** The README and `docs/ci.md`
  both documented that line and no `v0` ref existed, so every copy-paste of our own CI
  instructions failed. There is now a floating `v0` tag, a release-workflow job that moves
  it after a successful publish, and a smoke test that runs the documented line verbatim
  and asserts both the clean-tree pass and the non-zero exit on a tree with bugs.

### Changed

- The Action's description now names both halves of the tool rather than only the linter.
  It is the line that appears in Marketplace search results.

## [0.3.0] — 2026-08-14

Seven new rules, and a VRAM estimator that now covers serving as well as training.

### Added

**Rules — 6 to 13.**

- **TG006** binary cross-entropy paired with the wrong activation: `sigmoid` into
  `BCEWithLogitsLoss` (applied twice), raw logits into `BCELoss` (`nan` on the first
  negative value), and the numerically fragile-but-correct `sigmoid` + `BCELoss`, which
  warns rather than errors. Autofixable when the sigmoid is inline.
- **TG007** a GPU sync (`.item()`, `.cpu()`, `.numpy()`) inside a loop nested in the
  training step, or `torch.cuda.synchronize()` every step. One sync per step is *not*
  flagged — that is what TG001 tells you to write, and there is a test asserting the two
  rules never contradict each other.
- **TG008** a training run whose randomness is unseeded, or seeded for only some of torch /
  NumPy / `random`. Names which generator is missing; partial seeding is the usual shape.
- **TG011** `model.eval()` in an epoch loop with no matching `train()`, so only the first
  epoch trains with dropout on and batch-norm updating.
- **TG012** a `DataLoader` under DDP with no `DistributedSampler`: every rank iterates the
  whole dataset, so N GPUs train as one. Errors on training loaders, warns on evaluation.
- **TG013** a host-to-device transfer repeated every iteration — loop-invariant data, a
  `torch.*` host factory, or the model itself. Batch transfers are not flagged.
- **TG014** gradient accumulation without dividing the loss, which scales the summed
  gradient by the accumulation count — arithmetically the same as an N× learning rate.
  Autofix rewrites `loss.backward()` as `(loss / N).backward()`.

**Estimator.**

- **KV cache and generation sizing.** `--generate` and `--max-context` model autoregressive
  decoding: the cache is `2 · layers · kv_heads · head_dim · context · batch · dtype`, and
  it is where grouped-query attention pays off. Detected automatically from `.generate(...)`
  or `use_cache=True`.
- **Encoder-decoder models.** T5 and Whisper estimate activations instead of reporting
  unknown, including the cross-attention term a decoder-only formula cannot express.
  Validated to 2.5% worst case over 12 shapes, with two model sizes held out of the fit.
- **DeepSpeed config parsing.** The ZeRO stage is read from the dict or JSON your script
  points at rather than assumed to be stage 2, and `offload_optimizer` removes optimizer
  state and the fp32 master copy from the device.
- `tests/calibration/verify_snapshot.py` checks the bundled architecture snapshot against
  the live Hugging Face configs.

### Fixed

- **`--inference` charged an attention matrix that decoding never builds.** It ran the
  training activation formula, so a GPT-2 generation estimate at batch 32 and 4096 context
  read 105 GiB. Generation now costs a single decode step.
- **`Provenance.criteria` leaked across scopes.** Two functions each binding `criterion`
  collided and whichever was parsed last decided the loss class for both, so a correct
  `BCELoss` call could be reported as an error against `BCEWithLogitsLoss`. **This affected
  TG005.**
- The release workflow's `workflow_dispatch` rehearsal skipped the artifact download, so an
  incompatible upload/download pair would only have surfaced during a real release.
- Inference estimates no longer name an optimizer in the config summary.

### Changed

- Calibration extended to ResNet-50, confirming the meta-device activation measurements
  against a real allocator (+5.6% and +1.4%). Mean absolute error stays 3.7% across 8 runs.
- Grouped-query attention is deliberately **not** modelled as reducing training activations.
  Measured: `transformers.repeat_kv` materialises full-size K/V, so retained bytes are
  identical across `kv_heads` of 16, 8, 4 and 2. It does shrink the KV cache, which is
  modelled.
- GitHub Actions pins moved off the deprecated Node 20 runtime.

### Known gaps

Stated rather than papered over, and tracked as issues:

- `CUDA_CONTEXT_BYTES` is still a single Tesla T4 measurement ([#21](https://github.com/highwaterlabs/torch-preflight/issues/21)).
- The LM-head cost per logit is not batch-invariant, so GPT-2 at batch 8 stays ~12% under
  ([#22](https://github.com/highwaterlabs/torch-preflight/issues/22)).
- `offload_param` is detected but not subtracted, so those runs are over-estimated
  ([#24](https://github.com/highwaterlabs/torch-preflight/issues/24)).
- Paged-attention runtimes (vLLM, TensorRT-LLM) manage the KV cache in blocks; the estimate
  is the right order of magnitude, not their occupancy.

## [0.2.0] — 2026-08-13

### Fixed

- **`VRAMGuard` no longer treats activation memory as zero.** It profiled parameters
  exactly and had no way to see activations, so the term silently read zero — against a
  real ResNet-50 at batch 32 it projected 0.61 GiB where the card peaked at 1.86 GiB
  (−67.5%). Under-estimating is the direction that keeps a guard quiet through the OOM it
  exists to prevent. The guard now measures activations from the live module by running it
  against meta-device parameters, which allocates nothing and leaves the model untouched;
  the same case is now +8.8%. Pass `example_input` for models whose input shape cannot be
  derived from `seq_len` or `image_size`, or `measure_activations=False` to skip it. If the
  module cannot run on meta tensors the term is reported unknown, never zero.

### Changed

- LM-head backward transient raised from 6 to 10 bytes per logit element, replacing a
  two-point fit with a measured vocabulary sweep (8k–128k vocabularies at two batch sizes).
  Mean absolute error against measured peaks improves from 4.4% to 3.7%.
- Calibration fixtures extended with ResNet-50 peaks, confirming the meta-measured CNN
  activations against a real allocator (+5.6% and +1.4%).

## [0.1.0] — 2026-08-13

First release. [On PyPI](https://pypi.org/project/torch-preflight/0.1.0/).

### Linter

- **TG001** tensors stored with their autograd graph attached (`losses.append(loss)`,
  `total += loss`, `cache[k] = out`)
- **TG002** evaluation or inference running without `torch.no_grad()`
- **TG003** `.backward()` in a loop with no `zero_grad()` and an optimizer step
- **TG004** `DataLoader` starving a CUDA device (`num_workers=0`, no `pin_memory`)
- **TG005** `softmax` before a loss that expects raw logits
- **TG010** projected peak VRAM exceeding the configured `target_gpu`
- Grad-provenance dataflow analysis, so a finding depends on whether a value actually
  carries a graph rather than on what the line looks like
- Autofixes as concrete syntax tree rewrites, preserving formatting and comments
- Terminal, JSON, SARIF and GitHub annotation output
- `# noqa: TG001`, `# torch-preflight: ignore[...]` and `# torch-preflight: skip-file` suppression

### VRAM estimation

- `torch-preflight estimate` projects peak memory from a training script without importing or
  executing it
- Remediation solver reporting which change would make a run fit
- 41 bundled architectures, 23 GPUs, 34 cloud instance types
- Exact profiling of arbitrary models on PyTorch's meta device (`[vram]` extra)
- Hugging Face architecture lookup (`[hub]` extra)
- `VRAMGuard` context manager, failing a run at step 0 rather than OOM at step 400
- Constants calibrated against measured hardware; 5.0% mean absolute error against
  measured peaks

### Packaging

- Python 3.9–3.13
- Base install has no heavy dependencies and never imports torch, asserted in CI
- Pre-commit hook and GitHub Action
