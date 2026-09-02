# TODO

Decided work, grouped by phase. Delete items when done — git remembers them.
Undecided things live in [IDEAS.md](IDEAS.md).

---

## Phase 0 — static linter core ✅ done

TG001–TG005, provenance analysis, engine, autofixer, 4 reporters, CLI, config,
suppressions, pre-commit hooks, GitHub Action, CI matrix.

## Phase 1 — VRAM static tier ✅ done

Per RFC [0001](rfcs/0001-vram-estimator.md). No new **required** dependencies.

- [x] `vram/hardware.py` — 24 GPUs + 21 cloud instances, usable (not advertised) capacity
- [x] `vram/costmodel.py` — six-term model, constants isolated in a CALIBRATION block
- [x] `vram/archdb.py` + bundled snapshot — 40 architectures, ships in the wheel
- [x] `vram/extract.py` — RunConfig from the CST, with per-field provenance
- [x] `vram/solver.py` — remediation search with mutual-exclusion groups and a capped
      greedy stack
- [x] `torch-preflight estimate` / `torch-preflight gpus`
- [x] Risk banding with error intervals; no fabricated probability
- [x] TG010, gated on `target_gpu`, silent at low confidence, never touches the network
- [x] `tests/calibration/` — parameter formula enforced to 3% against published counts
- [x] Base-install CI job asserting torch/huggingface_hub never load

### Calibration ✅ activation side done

- [x] Spike [0001](spikes/0001-meta-device-activation-capture.md) — GO. `saved_tensors_hooks`
      works on meta, dedup via `storage._cdata`, fused attention is *not* a blind spot
- [x] `tests/calibration/measure_activations.py` — measures the activation coefficients on
      the meta device, no GPU required
- [x] Coefficients are now architecture-dependent and measured: dropout triples the
      quadratic term, so Llama-class models are no longer over-charged (−44% on the
      llama-2-7b activation estimate)
- [x] Constants pinned by tests — changing one without re-measuring fails the suite

### Before shipping

- [x] **Triaged every torch finding.** My guess that they were "plausibly intentional
      retention" was wrong: four of the five were our bugs. 10 findings -> 5, all of the
      survivors deliberate graph retention. Each fix is regression-tested:
      - **`models_to_test` read as a validation dataloader.** In a test suite half the
        identifiers contain "test". An eval-loader name must now also look like something
        you iterate batches from (`loader`/`dataset`/`batches`/...).
      - **`get_loss(...)` treated as a torch functional** because it ends in `_loss`,
        returning grad-bearing regardless of its arguments. That suffix heuristic now
        applies only to the known `F.*` functionals when called by a bare name.
      - **`torch.autograd.grad(...)`** returns detached tensors unless `create_graph=True`.
      - **Name leakage across functions.** `prov.models` was a flat, file-wide set, so
        `prepared = DistributedDataParallel(...)` in one helper made an unrelated
        `prepared` a thousand lines away look like a model. Both `models` and `grad`
        lookups are now scope-aware and stop at the first scope that binds the name — the
        same bug also let an outer `loss` leak into a nested helper with its own `loss`.

      Remaining 5, all judged true-but-intentional (a `# noqa` case, not a rule change):
      - `_inductor/fx_passes/numeric_utils.py:151,152` — TG003, a gradient-comparison
        harness that deliberately calls `backward(retain_graph=True)` in an optimizer loop
      - `distributed/pipelining/schedules.py:304` — TG001, pipeline parallelism must hold
        microbatch losses until their backward runs
      - `dist_autograd_test.py:2086,2098` — TG001, a dict deliberately chaining
        graph-attached tensors across ranks

      **The judgment on the three TG001 lines was wrong** — see the deferred-backward entry
      below. "Intentional, so `# noqa` it" is only the right answer when the intent is
      invisible to us; here it was structural and we could have detected it. The TG003 line
      stands.

      Reproduce with: `torch-preflight check <site-packages>/torch -f json`

### Known gaps that came out of building it

- [x] **CNN activation memory measured** for all eleven vision models, on the meta device
      with no GPU — I had wrongly filed this as needing one. Batch-linearity and
      area-scaling are verified at measurement time, not assumed.
- [x] `CUDA_CONTEXT_BYTES` (135 MiB) and `FRAGMENTATION_FRACTION` (0.105) measured on a
      Tesla T4; end-to-end peaks recorded for GPT-2, BERT, DistilBERT and ResNet-50. Mean
      absolute error against the eight measured peaks is **3.7%**.
- [x] **The meta-measured CNN activations check out against a real allocator.** ResNet-50
      on a T4 came in at +5.6% (batch 16) and +1.4% (batch 32) — the first end-to-end
      confirmation that measuring on a device that allocates nothing predicts a device that
      does. A re-measurement with the vision runs included puts fragmentation at 0.098
      against the shipped 0.105; the difference is within run-to-run spread and the shipped
      value errs high, so it stands.
- [x] **LM-head retained bytes measured**: exactly 4.00 per logit element, across five
      shapes and three vocabularies, precision-independent. Split out from the fitted
      backward-transient part so the evidence for each is visible.
- [x] **LM-head backward transient measured**, replacing the two-point fit. The sweep it
      needed now exists (the sweep in `measure_cuda.py --models`): four vocabularies from 8k to
      128k at two batch sizes on a tiny body, eight peaks spanning 16x in logit count.
      Least squares gives 15.72 bytes per logit of peak, 14.22 after dividing out
      fragmentation, minus the measured 4 retained — so the constant went 6 -> 10 and mean
      absolute error 4.4% -> 3.7%. Still *not* the fixture optimum of 14: GPT-2 is the only
      measured peak with an LM head, so that "fit" is two points again, and they disagree
      in sign.
- [ ] **The per-logit cost is not batch-invariant** — tracked in [#22](https://github.com/highwaterlabs/torch-preflight/issues/22).
- [x] **`VRAMGuard` ignored activations entirely** — `profile_live_model` returns exact
      parameter counts and no shape, so the term silently read zero. Measured against a
      real ResNet-50 at batch 32 the guard projected 0.61 GiB where the card peaked at
      1.86: **−67.5%**, and *under*-estimating is the direction that keeps a guard quiet
      through the OOM it exists to prevent. It now measures activations from the live
      module via `functional_call` on meta parameters — no allocation, no mutation of the
      caller's model — giving +8.8% on the same case.
- [ ] **The guard's measurement is forward-only**, so for a language model it sees the
      logits but not the loss temporaries that peak during backward. The static estimator
      models those; the guard does not.
- [ ] **Calibration covers one GPU architecture** — tracked in [#21](https://github.com/highwaterlabs/torch-preflight/issues/21).
      Partly answered on 2026-08-24. `CUDA_CONTEXT_BYTES` measured on a Colab T4 (torch 2.11)
      and a Kaggle T4 (torch 2.10): both **135 MiB to the byte**, matching the original and
      each other, including the 105/131/135 progression and every fragmentation figure to two
      decimals. Reproducible across torch versions, drivers and providers — which was worth
      knowing and was not known.
      Still one architecture. Every measurement is Turing, so the extrapolation across the
      other 22 cards in `hardware.py` remains untested. The free route to a second
      architecture narrowed the same day: PyTorch has dropped Pascal, so the P100 cannot run
      (#62). Colab's L4 (Ada) or an A100 are what is left.
- [x] **Encoder-decoder activations measured** for T5 and Whisper (8 snapshot entries),
      so they estimate instead of reporting unknown. A decoder-only formula cannot express
      these: there are two sequence lengths, and a decoder layer carries a third attention
      block whose K/V projections run at the *encoder* length.
      `tests/calibration/measure_encoder_decoder.py` fits per-family coefficients on the
      meta device. Validated against direct measurement to **2.5% worst case** over 12
      shapes, including whisper-medium and t5-large which were held out of the fit.
      `params_from_transformer_shape` now counts cross-attention too (T5 exact, Whisper
      within 2.7% -- its conv frontend and learned position tables are not in the formula).

      Three separate collinearities had to be broken, and **every degenerate version
      reported a better residual than the correct one**, which is worth remembering the
      next time a fit here looks clean:
      - encoder-linear against cross-KV: identical columns when `L_enc == L_dec`, which is
        true of every T5 and Whisper size. 0.00% residual, enc_linear 26.84 against a true
        48.34. Fixed by measuring the encoder alone first.
      - linear against quadratic, and decoder-linear against cross-attention: Whisper's
        encoder length is fixed at 1500 and every size uses head_dim 64, so those columns
        are exactly proportional -- unidentifiable in principle. Unconstrained it returned
        `dec_linear = 0.16`. Both quadratic terms are pinned to the separately measured
        attention coefficient; T5, where the split *is* identifiable, fits cross-attention
        free at 6.03 against the pinned 6.0.
- [x] **Grouped-query attention: the premise was wrong, no change needed.** I had filed
      this as "the activation formula ignores kv_heads and will over-estimate GQA models".
      Measured, it does not. `transformers.repeat_kv` expands K/V to the full head count
      and reshapes; reshaping a non-contiguous expand copies, so autograd retains full-size
      K/V exactly as MHA would — retained bytes are bit-identical across kv_heads 16/8/4/2.
      GQA saves parameters (already applied) and KV cache (not modelled, inference-only),
      not training activations. `enable_gqa=True` does avoid it (0.65x at an 8x ratio), but
      in transformers 5.x that path exists only in the exporters and one model. Charging
      the cheap rate would under-estimate every mainstream GQA model. Both directions are
      now pinned by tests so this cannot be "optimised" back.

      Note for anyone re-measuring: this must be done on the **CPU**, not the meta device.
      `enable_gqa` falls back to the math backend on meta and materialises the expanded
      K/V, so every variant measures identical and the effect is invisible. Spike 0001's
      "fused attention is not a blind spot" holds for plain SDPA but not for this flag.
- [x] **KV cache modelled**, and with it a correction to what `--inference` meant.
      `2 * layers * kv_heads * head_dim * context * batch * dtype`: plain arithmetic over a
      known allocation, not a measured constant. This is where grouped-query attention
      actually pays off — llama-3-8b's 8 KV heads against 32 query heads make its cache a
      quarter of the multi-head equivalent, which is the mirror image of the training-side
      finding that GQA does *not* reduce activations.

      Adding it exposed a real modelling error: `inference_only` ran the **training**
      activation formula, so a GPT-2 generation estimate at batch 32 and 4096 context read
      105 GiB — it charged a 4096x4096 attention matrix per layer that decoding never
      builds. Generation now costs a single decode step, one token attending against the
      cache. llama-2-7b at batch 32 / 4096 context now reads 12.6 GiB of weights against
      64 GiB of cache, which is the shape everyone who has sized a serving deployment
      recognises.

      Detected from `.generate(...)` / `use_cache=True` rather than from `inference_only`,
      because a plain forward pass caches nothing. `--generate` and `--max-context` expose
      it on the CLI. When only half the context is known the report says so: the cache is
      sized by prompt *plus* generated tokens, and under-counting is what lets a server OOM
      mid-request.
- [ ] **Paged attention is not modelled.** vLLM and TensorRT-LLM manage the cache in blocks
      with their own allocator, so the contiguous figure above is the right order of
      magnitude but not their actual occupancy.
- [x] **DeepSpeed ZeRO stage is now read, not assumed.** The comment said the stage "is in
      a JSON config we cannot read", which was untrue: it is either a dict literal in the
      same file or a path to a JSON file beside it, and reading JSON is not executing code.
      Handles `deepspeed.initialize(config=...)` by path, by variable and inline, plus
      `TrainingArguments(deepspeed=...)`. Falls back to the old stage-2 assumption when the
      config is genuinely unresolvable (built by a function call). Path traversal outside
      the source tree is refused, and missing or malformed JSON degrades quietly.
      This matters: stage 3 shards parameters too, so it is the difference between a 70B
      model fitting and not.
- [x] **DeepSpeed `offload_optimizer` modelled.** ZeRO-Offload runs the optimizer step in
      CPU memory, so neither the optimizer state nor the fp32 master copy is resident. For
      AdamW that is 8 bytes per parameter plus a 4-byte master copy — usually the largest
      single term, and the reason people enable offload. On llama-2-7b at ZeRO-3 across 8
      ranks it takes the projection from 23.79 GiB to 13.38 GiB.
- [ ] **`offload_param` is read but deliberately not subtracted** — tracked in
      [#24](https://github.com/highwaterlabs/torch-preflight/issues/24).
- [x] **GitHub Actions pins bumped off Node 20**: checkout v4->v7, setup-python v5->v7,
      upload-artifact v4->v7, download-artifact v4->v8. Rehearsed on `main` via
      `workflow_dispatch` and green.
      The first rehearsal only half-covered the risk: `publish` is skipped on a manual run,
      and that is the job holding `download-artifact`, so the upload was exercised and the
      download never was — the pair would have failed for the first time during a real
      release, after the version number was spent. `release.yml` now has a `rehearse` job
      that runs on `workflow_dispatch` only, downloads the artifact and asserts the publish
      step would find exactly one wheel and one sdist.

## Phase 2 — exact tier

- [x] Run spike [0001](spikes/0001-meta-device-activation-capture.md) — **GO**
- [x] `vram/providers/meta.py` + `[vram]` extra. Exact parameter counts and measured
      activations for arbitrary models via `module:factory` entry points, with
      `--model-args key=value`. Both spike traps are handled: parameter storages are
      excluded and dedup keys on `storage._cdata`. Cross-checked against
      `params_from_transformer_shape` — measuring and deriving agree, which is the
      strongest validation available without a GPU. Failures degrade to UNKNOWN rather
      than crashing, and `check` never reaches this path.
- [x] Model autodetection Layer 2 — `vram/autodetect.py`. Folds constructor arguments and
      module-level constants, resolves the class through the file's own imports, and
      meta-instantiates it. Guarded by an import-safety check: a module whose top level
      builds objects or calls functions is refused, because importing it would do that
      work for real. Unresolvable arguments are named, never guessed.
- [x] `[hub]` integration tested properly. Real `config.json` files are captured in
      `tests/fixtures/hub/` and replayed offline; a `network`-marked test (deselected by
      default, `pytest -m network`, plus a non-blocking CI job) hits the live hub.
      **This found two bugs the hand-written offline tests could not**, because I had
      authored both the field mapping and the fixtures so they agreed:
      - `tie_word_embeddings` absent means **True** in transformers, not False. Defaulting
        to False double-counted `vocab x hidden` and put GPT-2 31% over its real size.
      - DistilBERT names its fields `dim` / `hidden_dim` / `n_heads` / `tie_weights_`, and
        `hidden_dim` is the FFN width, not the model width. It resolved to nothing at all.
      All four captured models now derive within 0.1% of their published counts.

## Phase 3 — runtime ✅ done

- [x] `VRAMGuard` context manager — exact parameter/gradient/optimizer accounting from the
      live model, inferring optimizer kind and precision from the objects themselves.
      Raises only on `CERTAIN_OOM`; anything less certain warns, because aborting a job on
      a guess is worse than the OOM it would prevent. `strict=True` opts into raising.
      Exported lazily from `torch_preflight` so the base install still never imports torch.
- [x] Verification against `torch.cuda.max_memory_allocated()` on exit, exposed as
      `guard.measured_peak` and `guard.accuracy`.
- [ ] Feed real `guard.accuracy` measurements into the calibration fixtures — tracked in
      [#23](https://github.com/highwaterlabs/torch-preflight/issues/23).

### Rules beyond the first five

- [x] **TG006 — binary cross-entropy activation mismatch.** Four cases: `sigmoid` into
      `BCEWithLogitsLoss` (double sigmoid, error, autofixable when inline), the same via a
      variable, raw logits into `BCELoss` (`nan` on the first negative value, error), and
      `sigmoid` + `BCELoss`, which is *correct* but numerically fragile, so it warns rather
      than errors. Clean on torch's 2,285 files after one round of triage.

      Two bugs found while building it, both now regression-tested:
      - **`Provenance.criteria` was a flat name -> class map**, so two functions each
        binding `crit` collided and whichever was parsed last decided the class for both.
        A correct `BCELoss` call was reported as a double-sigmoid error against
        `BCEWithLogitsLoss`. Now scope-aware, matching `grad` and `models`. **This
        affected TG005 too.**
      - **Three false positives in `torch/testing/_internal/common_nn.py`**: flagging any
        `nn.Sigmoid()` construction in a file that mentioned `BCEWithLogitsLoss` anywhere.
        A bare `sigmoid = nn.Sigmoid()` local used to build a reference implementation is
        not a model ending in a sigmoid; only final position in an `nn.Sequential` is.
- [x] **TG014 — gradient accumulation without scaling the loss.** Keys off the modulo
      guard (`if (i + 1) % accum == 0: optimizer.step()`) rather than a variable name,
      because that guard *is* what makes several backward passes share one step. Silent
      when the loss is divided inline, by reassignment or with `/=`, and when a framework
      owns the scaling (Accelerate, HF Trainer, Lightning, DeepSpeed all divide internally,
      so telling someone to divide again would introduce the mirror-image bug). Autofix
      rewrites `loss.backward()` as `(loss / N).backward()`, which scales only what
      autograd sees so later logging of `loss` reports the same value as before. Clean on
      torch's 2,285 files.

      Also corrected a TG003 fixture: `test_tg003_quiet_for_gradient_accumulation`
      asserted a whole-file `codes(...) == []` on a snippet that accumulates without
      dividing — a real TG014 finding. The fixture now scales the loss, so it is genuinely
      correct accumulation and the whole-file assertion still holds.
- [x] **TG012 — DataLoader under DDP with no DistributedSampler.** Every rank iterates the
      whole dataset in the same order, so N ranks compute gradients on identical batches and
      DDP averages them to no effect: N GPUs training as one, on 1/N of the data per epoch.
      Errors on training loaders, warns on evaluation ones (duplicated validation wastes
      work but computes the right number). Silent when a sampler or `batch_sampler` is
      passed, and when Lightning, Accelerate or the HF Trainer is present — they inject a
      sampler, so flagging them would tell someone to shard already-sharded data.
      Clean on torch's 2,285 files.

      Caught the file-wide-leakage pattern a third time before shipping: `uses_distributed`
      is a per-file fact, so firing on it alone flagged every DataLoader in a file where any
      *one* function set up DDP — including single-process helpers. The marker must now be
      in the loader's own function, or at module level where it governs everything.
      Regression-tested both ways.
- [x] **TG011 — `model.eval()` in an epoch loop with no matching `train()`.** `eval()` is
      sticky, so the usual train-then-validate loop trains properly only on the first epoch;
      after that dropout is off and batch-norm normalises with statistics it has stopped
      updating. Requires the whole shape before firing — a backward pass, a validation
      iteration, an `eval()` and no matching `train()` in the same loop — and matches the
      *receiver*, so `model.backbone.eval()` to freeze batch-norm during fine-tuning is not
      flagged, while `model.train()` is still accepted as restoring it because `train()`
      recurses. Clean on torch's 2,285 files.
- [x] **TG013 — a host-to-device transfer repeated every iteration.** Narrower than the
      idea implied, deliberately: `.to(device)` is a *no-op* when the tensor is already
      resident, so the naive "flag `.to()` in a loop" rule would mostly flag things that
      cost nothing. Fires on three shapes that do cost: loop-invariant data copied each
      step, a `torch.*` host factory built then transferred, and `model.to(device)` inside
      the loop (no copy, but `Module.to` walks every parameter each time). Silent on batch
      transfers, `x = x.to(device)` self-assignment, factories that already pass `device=`,
      and dtype casts.

      Triaging torch's source took it from **57 findings to 8**, and each round was a real
      bug in the rule:
      - dotted receivers were checked whole, so `for shard in shards: shard.tensor.to(...)`
        did not count as loop-bound. Now the *root* name is what matters.
      - `.to(dtype)` was read as a device move. Device targets now need positive evidence —
        `.cuda()`, `device=`, a name containing "device", or a `"cuda"`/`"cpu"` literal —
        and anything else is left alone rather than guessed at.
      - the destination can itself vary per iteration: torch's `clip_grad` loops *over
        devices* and calls `clip_coef.to(device)`, which cannot be hoisted.
      - comprehension targets are not `LoopFrame` bindings, so `[t.cuda(r) for t in ts]`
        looked loop-invariant.
      The surviving 8 are all `torch.<factory>(...).to(device)` in test infrastructure —
      true double allocations, 0.0035 findings/file, in line with the 0.0033 baseline. Kept
      rather than tuned away. Random factories get a `device=` hint only, never "hoist it",
      since a fresh draw each iteration is the point.
- [x] **TG007 — a GPU sync inside a loop nested in the training step.** The rule the IDEAS
      entry warned about, and the warning was the whole design problem: `.item()` once per
      step is exactly what TG001 tells you to write, so flagging it would have the tool
      contradict itself. Resolved by keying off *nesting* rather than the call — the training
      step is the loop containing `.backward()`, a sync directly in it is once per step and
      fine, a sync in a loop inside it drains the pipeline per element. Comprehensions count.
      `torch.cuda.synchronize()` in the training loop is flagged separately as an
      unconditional drain. There is a test asserting the TG001 case stays silent; if it ever
      fires, the two rules are giving opposite advice and one has to change.
      6 findings on torch, all deliberate `cuda.synchronize()` in one distributed test file.
- [x] **TG008 — a training run whose randomness is unseeded.** Three independent generators
      (torch, NumPy, `random`) and seeding one does nothing for the others, so partial
      seeding is the usual shape rather than none. Only fires on code that trains, names
      which generator is unseeded, and recognises `seed_everything` / `set_seed`. Two false
      positives found against torch and fixed: `torch.rand(..., generator=g)` is
      deliberately controlled randomness, and a random *helper* in a file that trains
      elsewhere is a library function whose caller owns seeding — the file-wide leakage
      pattern, now for the fourth time. 4 findings on torch, all true-but-intentional in
      test infrastructure.
- [ ] TG009 only, in [IDEAS.md](IDEAS.md), and I would **skip it**: in-place ops on tensors
      needed for backward already raise a precise runtime error from PyTorch itself, so a
      pre-flight check adds little over what the interpreter tells you. It also needs real
      alias analysis. Worth an RFC if it is ever wanted, not worth a rule now.

- [x] **The documented Action usage was broken.** README and `docs/ci.md` both said
      `uses: highwaterlabs/torch-preflight@v0`, and no `v0` ref existed — only `v0.1.0`,
      `v0.2.0`, `v0.3.0`. Every copy-paste of our own CI instructions would have failed to
      resolve. Nobody hit it because nobody had used the Action yet, and publishing to the
      GitHub Marketplace is exactly what would have changed that.
      Fixed three ways: a floating `v0` tag; a `major-tag` job in `release.yml` that moves
      it after `publish` succeeds, so it can never point at a version that failed to reach
      PyPI; and `action-smoke.yml`, which runs the documented line verbatim and asserts both
      the clean-tree pass and the non-zero exit on a tree with bugs. CI tested the package
      thoroughly and had never once tested the thing users are told to write.

- [x] **The Node 20 pin bump missed `action.yml`.** It updated
      `.github/workflows/*.yml` and not the composite action at the repo root — so our own
      CI was clean while every *user* of the Action still got the deprecation warning in
      their logs. Found in the `action-smoke.yml` output, which is the first thing that had
      ever run the Action as a consumer does. Worth remembering that "our workflows" and
      "the workflow we ship" are different surfaces.

- [x] **Action published to the GitHub Marketplace**, from `v0.3.1` rather than `v0.3.0`.
      That distinction mattered: `v0.3.0`'s tree still had the old description and the
      deprecated Node 20 pin, because both fixes landed after it was tagged, and the
      Marketplace flow offers exactly that tag by default. Cutting 0.3.1 — a release with no
      package changes at all — was the cheap way to make the listing point at a correct
      tree. Live at https://github.com/marketplace/actions/torch-preflight
      Three inbound channels now: PyPI, the Marketplace, and one Reddit post.
- [x] **0.3.1 released.** The `major-tag` job ran for real for the first time and moved `v0`
      to the release commit unattended, which is the whole reason it was automated.
- [x] **Merged branches deleted again**, all nine verified merged first; only `main` remains
      on either side. This is the second time they have piled up after a release — worth
      considering whether `--delete-branch` on merge should just be the default.

- [x] **TG001 no longer fires on a deferred backward.** Came out of asking whether we should
      raise PRs against PyTorch and other repos for the findings we had. Re-reading the three
      TG001 lines to write those PRs showed they were *our* bug, not torch's, and the earlier
      "intentional, so `# noqa` it" verdict had been too generous to the rule:
      - `pipelining/schedules.py:304` holds each microbatch loss so the schedule can backward
        it when it reaches that microbatch. Taking our hint would have broken the schedule.
      - `dist_autograd_test.py:2086,2098` builds a dict chaining graph-attached tensors
        precisely so `dist_autograd.backward(ctx, [res[i].sum()])` can traverse it.

      Both are the same shape: **retention that a later backward depends on**, which is the
      one case where TG001's advice is not merely noisy but wrong. So a container is now
      exempt once an element read out of it reaches a backward pass — on the element
      directly, through a backward-taking call at any depth (the `dist_autograd` read is
      behind a list, a call and a subscript), or by being returned to a caller who does it.
      `total += loss` followed by `total.backward()` is exempt for the same reason, and that
      was a latent false positive nothing had hit yet.

      Three things worth keeping:
      - **The exemption needs a backward, not any read.** `torch.stack(self.outputs).mean()`
        at epoch end is the classic Lightning reduction and needs no graph, so that stays a
        finding. A read that only reduces or logs proves the retention was avoidable.
      - **Only subscript reads count as escaping via `return`.** `return loss` is the
        Lightning leak's own shape; `return self._internal_losses[i]` is a getter for a held
        graph. Collapsing the two would have silenced the rule's most common true positive.
      - **`self.*` matches file-wide, bare locals only within their function.** Instance
        state genuinely crosses methods — that is the whole pipelining shape — but scoping
        the local case is the file-wide fact leakage that has now bitten `models`,
        `criteria`, `uses_distributed` and TG008. Fifth time; the shared scoped-fact helper
        in [IDEAS.md](IDEAS.md) keeps earning its place.

      Findings on torch's 2,285 files: 23 -> 20. Nothing else moved.

- [x] **Scanned seven real training repos and triaged the high-value findings by hand.**
      1,615 files, 318 findings — 0.20/file against torch's 0.0087, which is the difference
      between scanning training code and scanning a framework. Reading every TG001/003/005/014
      site produced **one** PR-able finding and **nine bugs in us**, so the tool was not ready
      to file anything. Repos: `pytorch/examples`, `pytorch/tutorials`, `torchtune`, `litgpt`,
      `trl`, `LLaMA-Factory`, `transformers/examples/pytorch`. Raw JSON and the full triage
      are reproducible with `torch-preflight check <repo> --format json`.

      Fixed on this branch, verified by re-scanning: **errors 63 -> 32, twenty false positives
      removed, zero new findings.**
      - **TG001 rewritten around backward-reachability** (48 -> 32). The old exemption was
        syntactic and could not tell `torch.stack(losses).mean()` used for logging from the
        same line used as the training loss. Now a holder is exempt when its value reaches a
        backward or a *derived* return, computed as a fixpoint over assignment edges. Killed
        the chunked-loss false positives in `torchtune/modules/loss/`, the RL objectives in
        `trl` and `examples/reinforcement_learning/actor_critic.py`, and LLaMA-Factory's PPO.
      - **TG001 severity split, and it needed a measurement.** `backward()` frees each node's
        saved tensors as it traverses, so a tensor stored *after* its backward retains graph
        nodes and no activations. `tests/calibration/measure_retention.py` walks the live
        graph: **560 KiB of activations per iteration** retained when nothing backwards them,
        **0 KiB** when something does, ~30 nodes per iteration either way. So error/CRITICAL_OOM
        for the first and warning/PERFORMANCE_WARN for the second. **Our README's headline
        example was the second shape**, which means the flagship claim overstated the flagship
        example; it now shows both and explains the difference.
        RSS was the obvious instrument and the wrong one — three runs of identical code spread
        over 5.8-20.9 KiB/iteration, so the harness measures the graph directly instead.
      - **TG005 no longer reads a submodule as the model's output activation.**
        `pytorch/examples/gat/main.py` has `self.softmax = nn.Softmax(dim=1)` for **attention
        coefficients** and correctly ends in `F.log_softmax`. Attention softmax is in every
        transformer and GNN, so constructing the layer cannot be the evidence — final position
        in a `Sequential` is, matching the TG006 sigmoid fix.
      - **TG005 reads the constructor, not the attribute name.**
        `char_rnn_generation_tutorial.py` binds `self.softmax = nn.LogSoftmax(dim=1)`, which is
        correct for `NLLLoss`. We reported PyTorch's own tutorial as a convergence bug because
        the name won over a constructor two lines away.
      - **TG014 recognises gradient-level rescaling.** torchtune weights each micro-batch loss
        by its token count and applies `scale_grads_(params, 1/num_tokens)` before `step()` — a
        token-mean across uneven micro-batches, a *better* normalisation than dividing by the
        step count. Our hint would have introduced a real bug. Also resolves the call through
        an alias, since the distributed recipe binds `self._grad_scaler = training.scale_grads_`
        so it can wrap it in `torch.compile` — the same "read the binding, not the name" lesson
        as TG005, in the same afternoon.
      - **Values produced under `no_grad` no longer propagate a graph.** The standard eval loop
        wraps only the forward and uses the result after the block, where a positional check
        cannot see it.

      Two causes deliberately **not** fixed here, both needing design work rather than a patch,
      recorded with their evidence in [IDEAS.md](IDEAS.md): within-file callee return types
      (6 findings) and flow sensitivity within a scope (6 findings, and the reason the `no_grad`
      fix does not yet clear the Hugging Face eval loop).

      What survives as genuinely PR-able: **TG003 x3 in
      `pytorch/examples/distributed/tensor_parallelism/`** — `backward()` then `step()` in a
      loop with no `zero_grad()` anywhere in the file, in three official examples people copy.
      One line each.

      Also worth deciding separately: **TG004 is 207 of the 318**, 13% of every file scanned,
      all "unset `num_workers`" or "no `pin_memory`". Factually right and not actionable at that
      volume — tutorials keep it simple deliberately. Note `litgpt` came back **137 files, 0
      errors**, so careful code does scan clean.

- [x] **Filed the tensor-parallelism PR** — [pytorch/examples#1424](https://github.com/pytorch/examples/pull/1424).
      Three one-line `zero_grad()` additions, placed after `optimizer.step()` to match the
      sibling `distributed/FSDP2/example.py` rather than the `mnist` convention of putting it
      before the forward. Re-verified against live `main` first, since the clone was a day old.
      Deliberately left out a second finding in the same directory: `sequence_parallel_example.py`
      has no `manual_seed` where its two siblings do, but its own comment says "input can be
      different across all ranks", so it reads as intentional — and bundling it would have given
      a reviewer something to argue about. One finding, one PR. The tool is credited in a single
      line at the bottom rather than the top.
      **The single upstream-reportable result from 318 findings across 1,615 files.**
- [x] **Decided and implemented RFC 0003** — [what severity means, and what should fail a build](rfcs/0003-severity-and-ci-gating.md).
      Written because the TG001 split stopped the classic `losses.append(loss)` from failing
      CI by default. The measurement says the split is right: on a 4-layer transformer the
      never-backwarded case retains **186 MiB/iteration** of activations and OOMs an 80 GB
      A100 in ~440 steps, while the backwarded case retains **0** activations and ~15 KiB of
      host bookkeeping — 1.4 GiB per 100k steps. Roughly 12,000x apart, and the gap widens
      with model size.
      The real finding is that `warning` is overloaded: it holds both that leak and TG004's
      "you did not set `pin_memory`", which was **207 of the 318** findings in the seven-repo
      scan. Proposal is to keep the split, move TG004 to `note`, keep `fail_on = error` as the
      default while documenting `fail_on = warning` for training repos, and add per-rule
      severity overrides. Breaking for anyone relying on TG001 to fail their build, so it
      needs a minor version and a release note.
      Two measurement mistakes are recorded in §2 because they nearly produced the opposite
      answer: RSS alone spread 5.8-20.9 KiB/iteration across identical runs, and extrapolating
      host cost from node counts predicted 280 KiB/iteration against a measured 12-16.

      Shipped: TG004 is a `note`, the three levels are defined in `docs/rules.md` on one axis
      (*what happens if you ship this*), and `fail_on = "warning"` is recommended in the README
      and `docs/ci.md` for repos whose product is training runs. Open questions resolved in §7 —
      TG007 and TG013 stay warnings because they name a specific defect rather than an unset
      default; notes print but never gate, because hiding them trades a measured noise problem
      for an unmeasurable trust one.
      **Per-rule severity overrides already existed** — `Config.severity_overrides`, applied in
      `engine.py`, with `TG004 = "note"` as the worked example in `docs/configuration.md`. The
      RFC proposed building them before anyone checked; §4.4 records that.
      Verified across the seven repos: 207 notes / 59 warnings / 32 errors, `fail_on=error`
      fails 5 of 7 and `fail_on=warning` fails 6 of 7, with litgpt clean at both.
      README's torch figures were stale and are corrected to **20 findings / 2,285 files** (the
      TG001 fix removed three), along with "every one triaged as deliberate", which stopped
      being true the moment three of them turned out to be ours. The torch
      scan is a poor source of PR-able findings and that is structural, not bad luck: 20 of
      the 23 are in `torch/testing/_internal`, where a deliberate `cuda.synchronize()` or a
      host-side factory in test setup costs nothing, and the framework itself contains no
      training loops for the rules to be about. Our rules are about *training scripts*, so
      the targets are repos that contain them — `pytorch/examples` and `pytorch/tutorials`
      rather than `pytorch/pytorch`, then `transformers/examples/pytorch`, `torchtune`,
      `litgpt`, `trl`, `LLaMA-Factory`. A bug in an examples repo is worth more than the same
      bug in core, because those files get copy-pasted into thousands of projects.
      Every finding gets read by hand before anything is filed. The base rate says why:
      TG013 went 57 -> 14 -> 8 on this codebase, and the pipelining case above would have
      been a rejected PR in `pytorch/pytorch` arguing against a design we had not understood.

- [x] **0.4.0 released**, and the Marketplace listing republished from it. The changelog leads
      with a breaking-change block rather than burying it under "Changed", because the TG001
      downgrade and TG004 becoming a note both mean a previously-red build goes green. Verified
      from the published wheel rather than the working tree: fresh venv, 11 packages, no torch,
      and the classic `losses.append(loss)` exiting 0 by default and 1 under `--fail-on warning`.
      Also caught while bumping: the documented pre-commit `rev:` pins had drifted to `v0.1.0`
      in `docs/ci.md` and `v0.2.0` in the README. Same class as the `@v0` tag that did not
      resolve — instructions we publish but never execute. A test that runs the documented
      pre-commit config the way `action-smoke.yml` runs the documented Action would close the
      whole class; filed as an idea rather than done.

- [x] **The standard Accelerate evaluation loop no longer reports a TG001 error**
      ([#46](https://github.com/highwaterlabs/torch-preflight/pull/46)). Two causes:
      `accelerator.backward(loss)` seeded `accelerator` itself as a live tensor, because the
      collector seeds whatever `.backward()` is called on and Accelerate inverts that shape;
      and a single `main()` binding `outputs` in both a training and an evaluation loop shared
      one key, so the detached binding inherited grad-ness from its sibling. Bindings now carry
      their enclosing loop identities. Python has function scope rather than block scope, so
      this says "detached *within this loop*" rather than shadowing the name outright.

      **The part worth remembering is the bug I put in on the way.** Detachment first
      propagated whenever a binding was not *provably* grad-bearing — absence of proof treated
      as evidence. That silenced `loss = compute_loss(...)` followed by `losses.append(loss)`,
      and silenced it even when `loss.backward()` was called on the name.

      Neither check caught it. All 437 tests passed, because every TG001 fixture assigns from
      something resolvable like `criterion(model(batch), y)`. And the seven-repo scan reported
      **zero new findings**, which is structurally blind here: a true positive that stops firing
      is indistinguishable from a false positive that got fixed. I had reported 24 removals;
      **14 were findings being silenced**, 8 of them genuine. The honest figure is 10.
      *A wild scan is evidence about false positives only.* False negatives need fixtures that
      deliberately exceed what the analysis can resolve, and there are now two.

- [x] **Merged branches deleted a third time**, and `v0` moved unattended again. Three releases
      in, both are reflexes rather than decisions — the repo setting for auto-deleting head
      branches would remove one of them permanently.

- [x] **Re-triaged every finding, warnings and notes included — and the worst rule was a
      warning.** 283 findings across the seven repos, read rather than clustered.
      - **TG007 was 6/6 false, and each one was the rule reporting its own advice.** Every
        finding was `correct += (predicted == labels).sum().item()` in a validation loop
        nested in a training loop, which is *verbatim* what its hint tells you to write. The
        batch-loop exemption matched iterable **names** (`loader`, `dataloader`, `batches`)
        and missed `dev_iter` and `valloader`. Fixed by requiring evidence of per-element
        iteration — a loop over `range(...)` — rather than lengthening the name list.
      - **TG002 said `test()` "never calls `.backward()`" while the call sat nine lines
        below.** `fgsm_tutorial.py` iterates `test_loader` and backwards through it on
        purpose, because an adversarial attack needs gradients w.r.t. the input. A carve-out
        meant for functions that both train and validate now checks the backward is not in
        *this* loop.
      - **TG001's bare-name return rule was too blunt.** `return logps` where
        `logps = torch.cat(all_logps)` is not the container being handed back; it is a
        reduction of it. A returned bare name now counts unless it is itself a holder.
        6 findings across `trl` and torchtune.

      14 removed, no new findings. Errors 21 -> 13, warnings 55 -> 49.

      **Reading the warnings and notes is what found the worst of it.** Errors had been
      re-read three times across this work; TG007 had never been looked at since it shipped,
      because it never produced an error. Worth making the next scan read every level.

      Still true and left alone: the 8 torchtune `running_loss += current_loss` warnings, and
      TG003 x3, already fixed upstream in pytorch/examples#1424. **TG004 verified accurate** —
      a sample of the 207 notes were all genuine `DataLoader` calls missing `num_workers` or
      `pin_memory`; the problem was only volume, which making it a note already solved.

- [x] **Finished triaging TG013 (10): 3 false, 7 true.** Two causes, both narrow:
      - **A download is not a redundant upload.** `pinmem_nonblock.py` loops 100 times over
        `tensor.to("cpu", non_blocking=True)` on a tensor created with `device="cuda"` — in a
        tutorial whose whole subject is measuring transfer behaviour. Wrong twice over, and
        `_device_argument` was accepting a `"cpu"` literal as a destination.
      - **Restoring the device after a deliberate `.cpu()` is required.** `fast_neural_style`
        does `transformer.eval().cpu()`, writes a checkpoint, then `transformer.to(device)
        .train()`. Hoisting that out would leave the model on the host for the rest of
        training.

      The 7 true ones are all the mild `device=` advice the rule is meant to give — a host
      factory or a `torch.tensor(python_list)` inside a loop — plus one genuinely hoistable
      constant, `self.STOP_TOKENS_TENSOR.to(self._device)` in torchtune.
- [x] **`from_pretrained` no longer makes a preprocessor a model.** It is in `MODEL_WRAPPERS`
      and matched before anything could object, so `AutoTokenizer.from_pretrained(...)` was a
      model and `tokenizer(x["question"])` was a forward pass — TG002 reported a missing
      `no_grad` around *tokenisation*. Same for feature extractors and image processors. Two
      findings, and the guard test asserts a real `AutoModelForCausalLM` still counts.
- [x] **Split TG008 rather than demoting it.** Applying the RFC 0003 test gives two answers,
      because the rule reports two things. *No seeding at all* is a choice we often cannot see —
      seeding frequently lives in the launcher, not the script — so that is a `note`. *Partial*
      seeding, torch seeded and NumPy not, is a defect whose intent is visible in the code, and
      stays a `warning`. On the seven-repo corpus that is **30 notes and 1 warning**, and the one
      that remains is the informative one. Shipped: measured on the corpus as exactly **30 notes
      and 1 warning**, and total warnings across the seven repos fall 46 -> 16.
- [x] **Wrote up the scan** as [spike 0002](spikes/0002-scanning-real-training-repos.md), and
      indexed both spikes and RFC 0003 in `design/README.md`, which had never listed them.
      Re-measured torch to check the README's figure rather than trusting it: still **20 findings
      across 2,285 files**, 18 of them in `torch/testing/_internal`.
- [ ] **Two known false-positive causes remain unfixed**, both surfacing in TG002: non-model
      callables read as forward passes (`tokenizer(...)`, `feature_extractor(...)`), and
      deliberate gradient use for attribution (`captumyt.py`). Also still open, from TG001:
      the `label` naming heuristic on non-tensor lists, and container element types.

## Cross-cutting

- [x] Name and org settled: package `torch-preflight`, org `highwaterlabs`, deliberately
      distinct so the company is not named after another project's trademark.
      `torch-guard` had to be abandoned: PyPI rejects names that collide after separators
      are stripped, and it normalises to the existing `torchguard`. Exact-name
      availability is not registerability — see IDEAS.md.
- [x] Public repo live at `highwaterlabs/torch-preflight`, MIT, CI green. The private
      `torch-preflight-cloud` repo is still not needed; RFC 0002 lives there when it exists.
- [x] "What stays free" section in the README, per RFC 0002 §6.
- [x] `design/` is tracked and public; README links verified.
- [x] Committed and pushed; README split into `docs/` and rewritten (PRs #1, #2).

- [x] **0.1.0 published to PyPI**, verified by installing from PyPI into a clean
      virtualenv: 9 packages, no torch, CLI and estimator both working.
      The first release attempt failed because the rename lived only on a local branch,
      so `main` — and the tag — still packaged `torch-guard`. The build validated the
      version against the tag but never the name, so it reached the upload step and died
      on an opaque OIDC rejection. A name guard now fails at build time instead.
- [x] **0.2.0 published**, the VRAMGuard activation fix. Minor rather than patch: the guard
      now fails and warns on configurations it previously waved through. Every release gate
      was run locally before tagging, because a PyPI version number can never be reused.
      Releases exist for both versions; repo description, topics and logo are set.
- [x] *Declined:* renaming the local working directory from `torch-guard` to
      `torch-preflight`. Kept as-is deliberately.
- [ ] **Version the private repo.** RFC 0002 sits unversioned in `~/Dev/torch-preflight-cloud/`.
- [x] Replaced the hardcoded test-count badge with live PyPI version, Python-version and
      licence badges, which update themselves.
- [x] **Merged branches deleted**, locally and on the remote. All 14 were verified merged
      into `main` first; only `main` remains on either side.
- [x] **Snapshot refresh process defined**: `tests/calibration/verify_snapshot.py`
      compares the bundled snapshot against the live hub configs. Deliberately a verifier
      rather than a regenerator, so a renamed upstream field cannot silently rewrite the
      numbers in a diff too large to review. Cadence and rationale in
      `tests/calibration/README.md`. Currently 6 entries verified clean.
- [x] **Stress-tested against torch's own source** (2285 files). False-positive rate after
      fixes: **0.0033 findings/file** — 3 findings in 900 files, all "true but intentional"
      deliberate graph retention (pipeline parallelism, distributed autograd tests). Three
      real rule bugs found and fixed, each now regression-tested:
      - `x.requires_grad = <expr>` seeded grad on any value, not just `True`; a detached
        leaf now also clears any seed (hit `torch.utils.checkpoint`)
      - "loss" matched mid-word in helper names with no grad flowing in
        (`_multilabelmarginloss_reference`); loss-named helpers now need a grad-bearing arg
      - `dist_autograd.backward(ctx, [loss])` accumulates into an RPC context, not `.grad`
      - TG003 now also requires an optimizer step in the loop: `.backward()` with nothing
        applying the gradients has nothing to go stale

- [x] **PERFORMANCE fixed: 14 min -> 51 s on torch (2285 files), a ~16x speedup.**
      Three changes, each measured:
      1. **One shared traversal** for all rules (`RuleDispatcher`) instead of one walk per
         rule — rules are no longer visitors, they are handlers reading dispatcher-owned
         state. 367 -> 92 ms/file (4.0x).
      2. **Deferred position resolution.** `PositionProvider.resolve` cost 3.1 s on a
         1.3 MB file — *more than parsing it* — and ~99.7% of files have no findings at
         all. Diagnostics now carry the node and positions are filled in afterwards, only
         when a file actually reported something. 92 -> 67 ms/file (5.5x cumulative).
      3. **Process pool across files**, with `--jobs`. Sequential when `--fix` is on,
         because the fixer replaces nodes by identity in the tree the rules ran against.
         Falls back to sequential if the environment refuses to fork.
      Parallel and sequential results are asserted identical in `tests/test_engine.py`.

      **That 51 s was measured at six rules and is no longer current.** At thirteen the same
      tree takes ~4m18s, and the cost scales with rule count: on a fixed 158 files, one rule
      is 5.4 s and ten are 26.4 s. The single traversal still does what it promised — the new
      rules add ~0.1 s between them — but every rule recomputes `dotted_name` and
      `final_attr` on the same nodes. Memoising those on the dispatcher is filed in
      [IDEAS.md](IDEAS.md). A normal project is unaffected: this repo's own `src/` lints in
      1.3 s.
