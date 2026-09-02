# Pre-flight VRAM estimation

Before renting the GPU, ask whether the run fits:

```bash
$ torch-preflight estimate finetune.py --gpu a100-80gb

Model      llama-2-7b  (arch-snapshot)   6.74 B params
Config     amp · AdamW · batch 4 · seq 2048
Read from  batch_size=finetune.py:9, optimizer=finetune.py:7,
           precision=finetune.py:15, seq_len=finetune.py:13

  weights            25.10 GiB  ██
  gradients          25.10 GiB  ██
  optimizer state    50.21 GiB  ███
  activations       114.00 GiB  ███████████████
  CUDA context         600 MiB  █
  fragmentation      21.44 GiB  ██
  ──────────────────────────────────────────────
  projected peak    236.44 GiB   (212.79 GiB – 260.08 GiB)

Target     NVIDIA A100 80GB (78.0 GiB usable)   →   303% of capacity   ✗ OOM

What would make it fit:
  ✗  − 88.00 GiB  →  148.44 GiB   flash attention / SDPA
       mathematically equivalent, removes the O(seq²) attention term
  ✗  −119.28 GiB  →  117.16 GiB   gradient checkpointing
       same result, roughly 30% slower
  ✗  − 41.42 GiB  →  195.02 GiB   8-bit AdamW (bitsandbytes)
       quantised optimizer state, minimal quality impact
  ~  −165.13 GiB  →   71.30 GiB   flash attention / SDPA + gradient checkpointing
                                  + 8-bit AdamW + halve micro-batch (2x accumulation)
       lands inside the card but not inside the error margin — it may well run,
       but there is no headroom for a fragmented allocator
```

`✓` fits with the error margin, `~` fits the point estimate but not the margin, `✗` does
not fit. A single 80GB A100 is genuinely the wrong tool for a full 7B fine-tune at
sequence 2048 — and the point is to learn that before renting one.

The model, batch size, sequence length, precision, optimizer and sharding strategy are all
read out of the script — nothing is imported or executed. Override any of them:

```bash
torch-preflight estimate train.py --gpu rtx4090 --batch-size 1 --seq-len 512 --dtype pure-bf16
torch-preflight estimate --model llama-2-7b --gpu 8xa100-80gb --sharding zero-3
torch-preflight estimate --params 13B --gpu-memory 48GiB
torch-preflight gpus --instances        # every known GPU and cloud instance
```

Cloud instance names work directly: `--gpu p4de.24xlarge`, `--gpu ml.p5.48xlarge`,
`--gpu a2-ultragpu-8g`.

## Install extras

```bash
pip install torch-preflight              # linter + static VRAM estimation, no torch needed
pip install "torch-preflight[hub]"       # + look up unknown architectures on the HF hub
pip install "torch-preflight[vram]"      # + exact meta-device profiling
```

Extras add *dependencies only* — the wheel is byte-identical either way, and the base
install never imports torch.

## Custom architectures

Models outside the bundled snapshot can be measured exactly, with `pip install
"torch-preflight[vram]"`:

```bash
torch-preflight estimate --model mypkg.models:build_gpt \
    --model-args layers=24 --model-args hidden=1024 \
    --gpu a100-80gb --batch-size 8 --seq-len 1024 --dtype amp
```

The model is instantiated on PyTorch's `meta` device — **zero bytes allocated, no GPU
required** — so parameter counts are exact rather than estimated, and a forward pass under
`saved_tensors_hooks` captures precisely the tensors autograd retains for backward. That
is activation memory by definition, not by formula.

This path imports and executes your code, so it is opt-in and explicit. `torch-preflight check`
never reaches it.

## As a CI gate

Declare the hardware your team trains on and TG010 fails the build when a config will not
fit it:

```toml
[tool.torch-preflight]
target_gpu = "rtx4090"
```

TG010 is deliberately conservative. It stays silent unless `target_gpu` is set, it needs a
model it can identify, it **never fires on a low-confidence estimate**, and it never
touches the network.

## How accurate is it?

```
peak = weights + gradients + optimizer state + master weights
     + activations + CUDA context + fragmentation
```

Parameter counts are exact for models in the bundled snapshot, and the analytic formula for
everything else is within ~1% of published counts (enforced by `tests/calibration/`).

The activation coefficients are **measured**, not assumed: `tests/calibration/measure_activations.py`
captures the tensors autograd actually retains via `saved_tensors_hooks` and fits them
against sequence length. It runs on the meta device, so it allocates zero bytes and needs
no GPU. That measurement showed the published constants are a midpoint of two regimes —
models with dropout retain three tensors of `b·a·s²` in the attention path, models without
retain one — so torch-preflight charges Llama-class models the cheaper rate they actually pay.

**Grouped-query attention does not reduce the activation estimate, and that is deliberate.**
GQA shrinks the K/V projections, so it looks like it should shrink activations — but
`transformers.repeat_kv` expands K/V back to the full head count and reshapes, which copies,
so autograd retains full-size K/V exactly as multi-head attention would. Measured, retained
bytes are identical across `kv_heads` of 16, 8, 4 and 2. GQA saves parameters (which the
formula does apply) and KV cache (inference, not modelled) — not training activations.
`scaled_dot_product_attention(..., enable_gqa=True)` genuinely avoids the copy, but almost
nothing uses it yet, and modelling the cheap path would under-estimate every mainstream GQA
model.

The allocator constants are measured on a real GPU, and end-to-end projections are checked
against measured peaks from actual training steps:

| | measured | estimated | error |
|---|---|---|---|
| GPT-2, batch 4 × seq 128 | 2.97 GiB | 3.04 GiB | +2.5% |
| GPT-2, batch 8 × seq 256 | 5.81 GiB | 5.09 GiB | −12.4% |
| BERT-base, batch 4 × seq 128 | 2.44 GiB | 2.39 GiB | −2.1% |
| BERT-base, batch 8 × seq 256 | 3.38 GiB | 3.33 GiB | −1.7% |
| DistilBERT, batch 4 × seq 128 | 1.53 GiB | 1.48 GiB | −3.4% |
| DistilBERT, batch 8 × seq 256 | 1.95 GiB | 1.94 GiB | −0.4% |
| ResNet-50, batch 16 × 224px | 1.24 GiB | 1.31 GiB | +5.6% |
| ResNet-50, batch 32 × 224px | 1.99 GiB | 2.02 GiB | +1.4% |

Mean absolute error 3.7%, on a Tesla T4. Regenerate with
`tests/calibration/measure_cuda.py --models`.

The ResNet-50 rows matter beyond their own accuracy: those activations were measured on the
meta device, which allocates nothing, and they predict a real allocator to within 6%. That
is the assumption the whole no-GPU-required approach rests on, now tested rather than
asserted.

Every estimate carries an interval, and the verdict bands account for it — there is no
fabricated "95% failure risk" probability, because there is no data to calibrate one
against. If the model cannot be identified, torch-preflight reports `UNKNOWN` rather than
guessing a parameter count.

DeepSpeed configs are read rather than assumed: the ZeRO stage comes from
`zero_optimization.stage` in the dict literal or the JSON file your script points at, and
`offload_optimizer` removes the optimizer state and fp32 master copy from the device, since
ZeRO-Offload runs that step in CPU memory. `offload_param` is detected but **not** subtracted
— the resident working set depends on prefetch depth and hasn't been measured here, so the
weights term stands and the report tells you the real peak is lower.

**Known gaps**, tracked in [design/TODO.md](../design/TODO.md):

- The LM-head cost per logit element is **not batch-invariant** in the measurements —
  ~19.7 bytes at batch 4 against ~14.7 at batch 8, consistently across four vocabularies. A
  single constant cannot express that, so GPT-2 at batch 8 × seq 256 remains ~12% under.
  Encoder models, which have no LM head, land within 4%.
- Entry-point profiling measures the forward pass only, so the transient where a
  checkpointed layer is recomputed during backward is still modelled analytically.
- Calibration covers one GPU *architecture* (Turing) and four model families.
  `CUDA_CONTEXT_BYTES` has now been measured three times — the original T4, plus a Colab T4
  on torch 2.11 and a Kaggle T4 on torch 2.10 — and all three agree at **135 MiB to the
  byte** (141,426,688), with the same 105 / 131 / 135 MiB progression through init, cuBLAS
  and cuDNN. So it is reproducible across torch versions, driver stacks and providers.
  What that does **not** establish is whether it varies by architecture: every measurement
  is Turing. Larger or newer cards plausibly differ, and `hardware.Gpu.context_mib` exists
  to hold per-card numbers as they arrive. The free route to a second architecture has
  narrowed — PyTorch dropped Pascal, so the P100 cannot run at all.
- Encoder-decoder families beyond T5 and Whisper (BART, Pegasus, MarianMT) have no
  measured coefficients, so they report activations as unknown rather than borrowing
  another family's numbers.

## Sizing a serving deployment

Generation is a different memory shape from training, and `--generate` models it:

```bash
torch-preflight estimate --model llama-3-8b --gpu a100-80gb --generate \
    --batch-size 16 --max-context 8192 --dtype pure-bf16
```

```
Config     pure-bf16 · batch 16 · context 8192 · generation (KV cache)

  weights            14.96 GiB  ███████████
  activations           20 MiB  █
  KV cache           16.00 GiB  ████████████
  ...
```

Two things change relative to a training estimate.

**The KV cache appears**, and it is usually the term that decides the answer. Each layer
keeps one key and one value per token generated so far:

```
2 (K and V) × layers × kv_heads × head_dim × context × batch × dtype
```

This is where grouped-query attention pays off. GQA does *not* reduce training activations —
that is measured and explained above — but the cache stores one K/V pair per **KV head**, so
llama-3-8b's 8 KV heads against 32 query heads make its cache a quarter of the multi-head
equivalent. For llama-2-7b, which has no GQA, a batch of 32 at 4096 tokens needs 64 GiB of
cache against 12.6 GiB of weights.

**Activations collapse.** Decoding feeds a single token forward and attends against the
cache, so the O(context²) score matrix never materialises. Estimating generation with the
training formula is a large over-estimate — it charges an attention matrix per layer that
decoding never builds.

`--generate` is inferred automatically from `.generate(...)` or `use_cache=True` in a script.
A plain forward pass caches nothing, so `--inference` alone does not imply it.

The cache is sized by prompt **plus** generated tokens. When only one of those is visible the
report says so, because under-counting the context is what lets a server OOM mid-request.
Paged-attention runtimes (vLLM, TensorRT-LLM) manage the cache in blocks with their own
allocator, so treat this as the right order of magnitude rather than their exact occupancy.

## Guarding a run at runtime

The estimator answers "will this fit?" before the job is submitted. `VRAMGuard` answers it
*inside* the process, against the model that actually exists:

```python
from torch_preflight import VRAMGuard

with VRAMGuard(model, optimizer=optimizer, batch_size=32, seq_len=2048):
    train()
```

```
VramRiskError: torch-preflight: this configuration is projected to need 701 MiB
(456 MiB-946 MiB) on limit 256MiB, which has 0.2 GiB usable. Verdict: CERTAIN_OOM.
  breakdown: weights 128 MiB, gradients 128 MiB, optimizer state 256 MiB, ...
  smallest change that fits: 8-bit AdamW (bitsandbytes) + pure bf16 weights
```

Parameters, gradients, optimizer state and the autocast cache come from the live model and
are exact; the optimizer kind and precision are read off the objects you pass.

Activations are **measured from your module**, not guessed. Given a `seq_len`, an
`image_size` or an explicit `example_input`, the guard runs one forward pass against
meta-device parameters with `saved_tensors_hooks` attached: that captures exactly what
autograd would retain while allocating nothing and leaving your model untouched — same
device, same dtype, same mode, in a few milliseconds. Without a shape, or if the module
cannot run on meta tensors (a `.item()` in `forward`, a custom autograd function), the term
is reported unknown and the interval widens. It is never assumed to be zero; for a
ResNet-50 at batch 32 the activations outweigh everything else combined, and a guard that
under-counts them stays quiet through exactly the OOM it was installed to catch.

It **raises only when the run cannot fit even at the optimistic end of the interval** —
anything less certain is a warning, because aborting a training job on a guess is worse
than the OOM it was trying to prevent. `strict=True` opts into raising on likely failures
too.

Needs `pip install "torch-preflight[vram]"`. On exit, `guard.measured_peak` and
`guard.accuracy` compare the projection against what the run actually used.

