"""The memory cost model.

```
peak = weights + gradients + optimizer state + master weights
     + activations + CUDA context + fragmentation
```

Pure arithmetic, zero dependencies. Both the static and the meta-device providers feed the
same function, which is the whole point of the split in RFC 0001 §3.

**Every constant in the CALIBRATION section is an empirical value, not a derived truth.**
They come from published accounting (the Megatron-LM activation formula) and from commonly
reported allocator behaviour. They are wrong in some regimes. `tests/calibration/` exists to
measure how wrong, and any change to these numbers should move a calibration fixture.
"""

from __future__ import annotations

from typing import List, Optional

from .types import (
    MIB,
    Confidence,
    MemoryBreakdown,
    ModelKind,
    ModelProfile,
    OptimizerKind,
    PrecisionMode,
    RiskBand,
    RunConfig,
    Sharding,
    TransformerShape,
    VramReport,
)

# ============================== CALIBRATION CONSTANTS ==============================

#: Per-process CUDA context: driver state, kernels, cuBLAS/cuDNN workspaces. Exists before
#: your first tensor and is invisible to ``torch.cuda.memory_allocated()``.
#:
#: MEASURED at 135 MiB on a Tesla T4 (torch 2.11, CUDA 12.8) — 105 MiB after init, rising
#: to 131 after the first cuBLAS call and 135 after cuDNN. We previously assumed 600 MiB
#: on no evidence. Measured three times now -- the original T4, a Colab T4 on torch 2.11 and
#: a Kaggle T4 on torch 2.10 -- agreeing at 135 MiB to the byte, with the same 105/131/135
#: progression through init, cuBLAS and cuDNN. That makes it reproducible across torch
#: versions, drivers and providers, but every measurement is **Turing**: whether the context
#: varies by architecture is still open, so :attr:`hardware.Gpu.context_mib` overrides this
#: per device as data arrives.
CUDA_CONTEXT_BYTES = 135 * MIB

#: The caching allocator reserves more than it hands out.
#:
#: MEASURED at 0.105 mean (range 0.062-0.125) across six real models on a T4 -- GPT-2,
#: BERT and DistilBERT at two shapes each.
#:
#: A synthetic sweep of hand-built transformer stacks gave 0.059, and adopting that would
#: have been a mistake: toy stacks allocate a handful of uniform tensors, while real models
#: churn through embeddings, masks and head projections of many different sizes. Calibrate
#: allocator behaviour against real models, not against a microbenchmark.
FRAGMENTATION_FRACTION = 0.105

#: Activation accounting per transformer layer:
#:     bytes ~= s * b * h * (ACT_LINEAR_COEFF + ACT_ATTN_COEFF * a * s / h)
#:
#: MEASURED on torch 2.13 by tests/calibration/measure_activations.py, which sweeps the
#: sequence length and fits alpha*s + beta*s^2 to the bytes autograd actually retains.
#: The published Megatron-LM constants (34, 5) sit between the two regimes below.
#:
#: Dropout is the discriminator. With p>0 the attention path retains three tensors of
#: b*a*s^2 (softmax output, dropout mask, dropout output) instead of one, tripling the
#: quadratic term. Modern LLMs ship p=0.0, which short-circuits and saves nothing.
ACT_LINEAR_COEFF_DROPOUT = 36.0
ACT_LINEAR_COEFF_NO_DROPOUT = 32.0
ACT_ATTN_COEFF_DROPOUT = 6.0
ACT_ATTN_COEFF_NO_DROPOUT = 2.0

#: Backwards-compatible midpoints, used when the architecture is unknown.
ACT_LINEAR_COEFF = 34.0
ACT_ATTN_COEFF = 5.0
ACT_REFERENCE_DTYPE_BYTES = 2

#: With full per-layer checkpointing only the layer input is stored. MEASURED at exactly
#: 2.00 — one fp32 tensor of s*b*h, normalised to the 2-byte reference.
CHECKPOINT_ACT_COEFF = 2.0

#: During recompute, one layer's full activations are live at once.
CHECKPOINT_RECOMPUTE_LAYERS = 1

#: Inference keeps only a couple of layers' activations alive at a time.
INFERENCE_LIVE_LAYERS = 2

#: An LM head produces [batch, seq, vocab] logits, which at a 128k vocabulary rivals the
#: entire transformer stack. The cost splits into two parts with very different evidence
#: behind them, so they are named separately rather than hidden in one constant.
#:
#: MEASURED. ``saved_tensors_hooks`` on the meta device, across five (batch, seq, vocab)
#: combinations spanning three vocabularies, gives exactly 4.00 bytes per logit element
#: retained for backward — one fp32 copy, the cross-entropy input. The head projection
#: itself retains ~0.02 bytes per element: its input is the hidden state, already counted
#: in the transformer term, and its weight is a parameter. Precision-independent, because
#: ``cross_entropy`` upcasts to fp32 whatever the autocast dtype.
LM_HEAD_RETAINED_BYTES = 4

#: MEASURED, by the vocabulary sweep this constant previously lacked. Forward-only capture
#: cannot see the backward pass, where the logits gradient and the softmax-backward
#: workspace are live simultaneously; this is the gap between what is retained and what the
#: allocator peaks at.
#:
#: The sweep in ``measure_cuda.py --models`` isolates the term on a deliberately tiny body
#: (4 layers, n_embd 256) so the vocabulary projection dominates, across four vocabularies
#: (8k / 32k / 50257 / 128k) at two batch sizes — eight peaks spanning 16x in logit count.
#: Least squares over all eight gives 15.72 bytes per logit element of *peak*; dividing out
#: FRAGMENTATION_FRACTION leaves 14.22 bytes allocated, of which LM_HEAD_RETAINED_BYTES is
#: the measured 4. Hence 10.
#:
#: Deliberately NOT the value that minimises error against measured_peaks.json. That
#: optimum is 14, but GPT-2 is the only fixture there with an LM head, so "the fit" is once
#: again two points — and they pull in opposite directions (b4 over-estimates, b8 under-),
#: which no single per-logit constant reconciles. Fitting to them would absorb unrelated
#: systematic error into whichever parameter scales with b*s*vocab.
#:
#: Known limitation: the per-logit cost is not batch-invariant in the measurements —
#: ~19.7 bytes/logit at batch 4 against ~14.7 at batch 8. A constant cannot express that,
#: so GPT-2 at batch 8 stays ~12% under. Tracked in design/TODO.md.
LM_HEAD_BACKWARD_TRANSIENT_BYTES = 10

#: Encoder-decoder activation coefficients, per architecture family.
#:
#: MEASURED by tests/calibration/measure_encoder_decoder.py across three sizes of each
#: family on the meta device. A decoder-only stack has one sequence length; these have two,
#: and a decoder layer carries a third attention block whose K/V projections run at the
#: *encoder* length. Neither is expressible in the decoder-only formula, which is why these
#: models used to report activations as unknown rather than guess.
#:
#: ``attn`` is not fitted. It is pinned to the separately measured self-attention
#: coefficient (6.0 with dropout, 2.0 without) and reused for cross-attention, because
#: otherwise the fit is degenerate: Whisper's encoder length is fixed at 1500 and every
#: Whisper size uses head_dim 64, which makes the linear and quadratic columns exactly
#: proportional. Unconstrained it returned a decoder linear coefficient of 0.16. T5, where
#: the split *is* identifiable, fits cross-attention free at 6.03 against the pinned 6.0.
#:
#: Families are separate because the architectures differ in what they retain: T5 carries
#: relative position bias and dropout, Whisper has a conv frontend and p=0.0.
ENCODER_DECODER_COEFFS = {
    "t5": {
        "enc_linear": 48.44, "dec_linear": 60.58, "cross_kv": 4.05, "attn": 6.0,
    },
    "whisper": {
        "enc_linear": 33.20, "dec_linear": 40.21, "cross_kv": 4.19, "attn": 2.0,
    },
}

# ===================================================================================


def transformer_activation_bytes(
    shape: TransformerShape, config: RunConfig, seq_len: int
) -> int:
    """Activation memory for a transformer, in bytes, for one device's micro-batch.

    ``kv_heads`` is deliberately not used here, though it *is* applied to the parameter
    count. Grouped-query attention shrinks the K/V projections, so the intuition is that it
    should shrink activations too — it does not, under the implementation essentially
    everyone runs. ``transformers.repeat_kv`` expands K/V to the full head count and
    reshapes, and reshaping a non-contiguous expand copies, so autograd retains full-size
    K/V exactly as multi-head attention would. MEASURED: retained bytes are bit-identical
    across kv_heads of 16, 8, 4 and 2 (``tests/test_calibration.py``).

    ``F.scaled_dot_product_attention(..., enable_gqa=True)`` genuinely avoids it, and there
    the saving is real (0.65x at an 8x group ratio) — but in transformers 5.x that path
    appears only in the exporters and a single model. Modelling the cheap path would
    under-estimate every mainstream GQA model, and under-estimating is what lets a run OOM.

    The score matrix is full-head under GQA either way: K/V are broadcast to the query
    heads, so the O(s^2) term uses ``shape.heads`` and is unaffected.
    """
    b = config.batch_size
    s = seq_len
    h = shape.hidden
    a = shape.heads
    dtype_scale = config.precision.activation_bytes / ACT_REFERENCE_DTYPE_BYTES

    if shape.uses_dropout:
        linear_coeff, attn_coeff = ACT_LINEAR_COEFF_DROPOUT, ACT_ATTN_COEFF_DROPOUT
    else:
        linear_coeff, attn_coeff = ACT_LINEAR_COEFF_NO_DROPOUT, ACT_ATTN_COEFF_NO_DROPOUT

    # Per-layer, per-batch linear term: everything that is not the attention score matrix.
    linear = linear_coeff * s * b * h

    # The attention score matrix is the O(s^2) term. Flash attention never materialises it.
    attention = 0.0 if config.flash_attention else attn_coeff * a * s * s * b

    if config.gradient_checkpointing:
        # Only layer boundaries are stored, plus one layer live during recompute.
        stored = CHECKPOINT_ACT_COEFF * s * b * h * shape.layers
        live = (linear + attention) * CHECKPOINT_RECOMPUTE_LAYERS
        return int((stored + live) * dtype_scale)

    if config.inference_only:
        return int((linear + attention) * INFERENCE_LIVE_LAYERS * dtype_scale)

    return int((linear + attention) * shape.layers * dtype_scale)


def decode_step_activation_bytes(shape: TransformerShape, config: RunConfig) -> int:
    """Activations for one autoregressive decode step.

    Generation does not re-run the whole sequence. Each step feeds a *single* token forward
    and attends against the cache, so the activation cost is a one-token slice and the
    O(seq^2) score matrix never materialises -- the history lives in the KV cache instead,
    which is accounted for separately.

    Modelling this with the training formula at full context is what made a GPT-2 generation
    estimate read 105 GiB: it charged a 4096x4096 attention matrix per layer that decoding
    never builds.
    """
    b = config.batch_size
    h = shape.hidden
    a = shape.heads
    context = config.max_context or config.seq_len or shape.max_position or 1
    dtype_scale = config.precision.activation_bytes / ACT_REFERENCE_DTYPE_BYTES

    linear_coeff = (
        ACT_LINEAR_COEFF_DROPOUT if shape.uses_dropout else ACT_LINEAR_COEFF_NO_DROPOUT
    )
    # One token through the stack, plus the attention scores of that token against the whole
    # cached context: [batch, heads, 1, context] rather than [batch, heads, seq, seq].
    linear = linear_coeff * b * h
    scores = 0.0 if config.flash_attention else 2.0 * a * context * b
    return int((linear + scores) * INFERENCE_LIVE_LAYERS * dtype_scale)


def encoder_decoder_activation_bytes(
    shape: TransformerShape, config: RunConfig, seq_len: int
) -> Optional[int]:
    """Activation memory for an encoder-decoder model (T5, Whisper).

    Two sequence lengths are in play. The encoder runs at ``shape.encoder_seq_len`` where
    the architecture fixes one — Whisper always sees 1500 positions regardless of how long
    the audio is — and otherwise at the configured length, which is the right default for
    T5-style sequence-to-sequence training where source and target are comparable.

    Returns ``None`` for a family with no measured coefficients, so the caller reports the
    term as unknown and widens the interval rather than inventing a number.
    """
    coeffs = ENCODER_DECODER_COEFFS.get(shape.activation_family or "")
    if coeffs is None:
        return None

    b = config.batch_size
    h = shape.hidden
    a = shape.heads
    s_dec = seq_len
    s_enc = shape.encoder_seq_len or seq_len
    attn = coeffs["attn"]
    dtype_scale = config.precision.activation_bytes / ACT_REFERENCE_DTYPE_BYTES

    encoder = shape.layers * (
        coeffs["enc_linear"] * h * s_enc
        + (0.0 if config.flash_attention else attn * a * s_enc * s_enc)
    )
    # A decoder layer adds cross-attention: its scores are b*a*s_dec*s_enc, and its K/V
    # projections run at the encoder length even though the layer is a decoder layer.
    decoder = shape.decoder_layers * (
        coeffs["dec_linear"] * h * s_dec
        + coeffs["cross_kv"] * h * s_enc
        + (0.0 if config.flash_attention else attn * a * s_dec * s_dec)
        + (0.0 if config.flash_attention else attn * a * s_dec * s_enc)
    )
    total = (encoder + decoder) * b

    if config.gradient_checkpointing:
        stored = CHECKPOINT_ACT_COEFF * b * h * (
            shape.layers * s_enc + shape.decoder_layers * s_dec
        )
        live = total / max(shape.layers + shape.decoder_layers, 1)
        return int((stored + live) * dtype_scale)
    if config.inference_only:
        live_fraction = INFERENCE_LIVE_LAYERS / max(
            shape.layers + shape.decoder_layers, 1
        )
        return int(total * live_fraction * dtype_scale)
    return int(total * dtype_scale)


def kv_cache_bytes(shape: TransformerShape, config: RunConfig) -> int:
    """Bytes held by the key/value cache during autoregressive decoding.

    Each layer keeps one K and one V entry per token generated so far, so the cache grows
    linearly with context and never shrinks within a sequence::

        2 (K and V) * layers * kv_heads * head_dim * context * batch * dtype

    This is where grouped-query attention actually pays off. GQA does **not** reduce training
    activations -- ``transformer_activation_bytes`` explains why, and it is measured -- but
    the cache stores one K/V pair per *KV head*, so Llama-3-70B's 8 KV heads against 64 query
    heads make the cache 8x smaller. Modelling the cache without the ratio would overstate
    every modern serving deployment by that factor.

    Arithmetic, not measurement: the cache is a plain allocation of a known shape, unlike the
    activation term. What is *not* modelled is the allocator's behaviour around it --
    paged-attention runtimes (vLLM, TensorRT-LLM) manage the cache in blocks and will differ.
    """
    if not config.generation:
        return 0

    context = config.max_context or config.seq_len or shape.max_position
    if not context:
        return 0

    head_dim = shape.hidden // shape.heads if shape.heads else 0
    kv_heads = shape.kv_heads or shape.heads
    if not head_dim or not kv_heads:
        return 0

    per_token = 2 * shape.layers * kv_heads * head_dim
    # The cache holds whatever dtype the weights are in; it is not upcast.
    return int(per_token * context * config.batch_size * config.precision.activation_bytes)


def cnn_activation_bytes(profile: ModelProfile, config: RunConfig) -> Optional[int]:
    """Activation memory for a vision model, scaled from a reference resolution."""
    if profile.activation_bytes_per_sample is None:
        return None

    per_sample = profile.activation_bytes_per_sample
    # Feature maps scale with spatial area.
    if config.image_size and getattr(profile, "reference_image_size", None):
        reference = getattr(profile, "reference_image_size")
        per_sample = int(per_sample * (config.image_size / reference) ** 2)

    scale = config.precision.activation_bytes / ACT_REFERENCE_DTYPE_BYTES
    total = per_sample * config.batch_size * scale

    if config.gradient_checkpointing:
        total *= 0.3  # rough: only checkpoint boundaries survive
    if config.inference_only:
        total *= 0.2
    return int(total)


def lm_head_bytes(shape: TransformerShape, config: RunConfig, seq_len: int) -> int:
    """Logits and loss temporaries for a model with a vocabulary projection."""
    if not shape.has_lm_head or not shape.vocab:
        return 0

    elements = config.batch_size * seq_len * shape.vocab
    if config.inference_only:
        # No loss, so nothing is retained for backward; only the logits themselves exist.
        return elements * config.precision.activation_bytes
    return elements * (LM_HEAD_RETAINED_BYTES + LM_HEAD_BACKWARD_TRANSIENT_BYTES)


def _activation_bytes(profile: ModelProfile, config: RunConfig) -> Optional[int]:
    if profile.activation_bytes_per_sample is not None:
        return cnn_activation_bytes(profile, config)

    if config.generation and profile.shape is not None:
        return decode_step_activation_bytes(profile.shape, config)

    if profile.shape is not None and profile.shape.is_encoder_decoder:
        seq_len = config.seq_len or profile.shape.max_position
        if seq_len:
            stack = encoder_decoder_activation_bytes(profile.shape, config, seq_len)
            if stack is None:
                return None
            # The seq2seq head projects decoder output to the vocabulary, exactly as a
            # causal LM head does; it is validated by the same measurements.
            return stack + lm_head_bytes(profile.shape, config, seq_len)
        return None

    if profile.shape is not None:
        seq_len = config.seq_len or profile.shape.max_position
        if seq_len:
            return (
                transformer_activation_bytes(profile.shape, config, seq_len)
                + lm_head_bytes(profile.shape, config, seq_len)
            )

    return None


def estimate(
    profile: ModelProfile,
    config: RunConfig,
    gpu: Optional[object] = None,
    gpu_count: int = 1,
) -> VramReport:
    """Project peak VRAM for one device."""
    breakdown = MemoryBreakdown()
    notes: List[str] = []
    extra_uncertainty = 0.0

    if not profile.resolved:
        return VramReport(
            profile=profile,
            config=config,
            breakdown=breakdown,
            gpu=gpu,
            gpu_count=gpu_count,
            band=RiskBand.UNKNOWN,
            notes=[profile.reason or "model could not be resolved"],
        )

    params = profile.param_count
    trainable = profile.trainable_params or params
    if config.frozen_fraction > 0:
        trainable = int(params * (1.0 - config.frozen_fraction))

    precision = config.precision
    world = max(config.world_size, 1)

    # Sharding divides different terms depending on the ZeRO stage.
    param_div = world if config.sharding is Sharding.ZERO3 else 1
    grad_div = world if config.sharding in (Sharding.ZERO2, Sharding.ZERO3) else 1
    opt_div = world if config.sharding in (Sharding.ZERO1, Sharding.ZERO2, Sharding.ZERO3) else 1

    breakdown.weights = int(params * precision.param_bytes / param_div)

    if not config.inference_only:
        breakdown.gradients = int(trainable * precision.grad_bytes / grad_div)
        if config.offload_optimizer:
            # ZeRO-Offload moves optimizer state *and* the fp32 master copy into CPU
            # memory, and runs the optimizer step there. Neither occupies the device, which
            # is the entire point of the feature: for Adam that is 8 bytes per parameter
            # plus a 4-byte master copy, usually the largest term for a large model.
            breakdown.optimizer_state = 0
            breakdown.master_weights = 0
        else:
            breakdown.optimizer_state = int(
                trainable * config.optimizer.states * config.optimizer.bytes_per_state
                / opt_div
            )
            breakdown.master_weights = int(
                trainable * precision.master_copy_bytes / opt_div
            )

    # autocast holds its casted weight copies through the backward pass, in inference too.
    breakdown.autocast_cache = int(params * precision.cast_cache_bytes / param_div)

    if profile.shape is not None:
        breakdown.kv_cache = kv_cache_bytes(profile.shape, config)
        if breakdown.kv_cache and not config.max_context:
            # The cache is sized by prompt *plus* generated tokens. With only one of them
            # known the estimate is low, and low is the direction that lets a server OOM
            # mid-request, so say so rather than presenting it as complete.
            notes.append(
                "KV cache sized from the sequence length alone. It grows with prompt plus "
                "generated tokens, so pass --max-context with the real total if the prompt "
                "is long."
            )

    if config.offload_params:
        # `offload_param` streams parameters in per layer, so only a working set is
        # resident -- but how large that set is depends on prefetch depth and
        # `param_persistence_threshold`, and we have not measured it. Rather than invent a
        # fraction, the full weights term stands and the report says it is high. Over-
        # estimating is the safe direction for a tool whose job is predicting OOM.
        notes.append(
            "DeepSpeed `offload_param` is enabled, which keeps only a working set of "
            "parameters on the device. The weights term below is the full size, so the "
            "real peak will be lower than shown."
        )

    activations = _activation_bytes(profile, config)
    if activations is None:
        notes.append(
            "Activation memory could not be estimated (no sequence length or architecture "
            "dimensions available). The real peak will be higher than shown."
        )
        extra_uncertainty += 0.25
    else:
        breakdown.activations = activations

    breakdown.cuda_context = getattr(gpu, "context_bytes", None) or CUDA_CONTEXT_BYTES

    allocated = (
        breakdown.weights
        + breakdown.gradients
        + breakdown.optimizer_state
        + breakdown.master_weights
        + breakdown.autocast_cache
        + breakdown.activations
    )
    breakdown.fragmentation = int(allocated * FRAGMENTATION_FRACTION)

    notes.extend(_advisory_notes(profile, config, gpu))

    report = VramReport(
        profile=profile,
        config=config,
        breakdown=breakdown,
        gpu=gpu,
        gpu_count=gpu_count,
        notes=notes,
        extra_uncertainty=extra_uncertainty,
    )
    report.band = _band(report)
    return report


def _band(report: VramReport) -> RiskBand:
    """Bands from the error interval, never a fabricated probability (RFC 0001 §7)."""
    if report.gpu is None or not report.profile.resolved:
        return RiskBand.UNKNOWN

    usable = report.gpu.usable_bytes
    low, high = report.interval
    total = report.total

    if high < usable:
        return RiskBand.FITS
    if total < usable:
        return RiskBand.TIGHT
    if low < usable:
        return RiskBand.LIKELY_OOM
    return RiskBand.CERTAIN_OOM


def _advisory_notes(profile: ModelProfile, config: RunConfig, gpu) -> List[str]:
    notes: List[str] = []

    if config.accumulation_steps > 1:
        notes.append(
            f"Gradient accumulation ({config.accumulation_steps} steps) multiplies the "
            f"effective batch to {config.global_batch} but does not change peak memory — "
            f"activations scale with the micro-batch of {config.batch_size}."
        )

    if config.precision is PrecisionMode.AMP:
        notes.append(
            "torch.autocast keeps parameters and gradients in fp32; only activations move "
            "to low precision. Casting the model itself is what halves the weight term."
        )

    if gpu is not None and not gpu.supports_bf16 and config.precision in (
        PrecisionMode.PURE_BF16,
        PrecisionMode.AMP,
    ):
        notes.append(f"{gpu.name} has no bf16 support; fp16 will be used instead.")

    if config.sharding is Sharding.DDP and config.world_size > 1:
        notes.append(
            f"DDP keeps a full replica on every rank, so {config.world_size} GPUs do not "
            f"reduce per-device memory. FSDP/ZeRO-3 would shard it."
        )

    if (
        profile.shape is not None
        and config.seq_len
        and not config.flash_attention
        and config.seq_len >= 2048
    ):
        notes.append(
            f"The attention score matrix is O(seq²) and dominates at seq={config.seq_len}. "
            f"Flash attention / SDPA would remove that term entirely."
        )

    return notes


def params_from_transformer_shape(shape: TransformerShape) -> int:
    """Analytic parameter count for a standard transformer.

    Verified against known models: llama-2-7b (L32 H4096 I11008 V32000, gated, untied)
    gives 6.74B, and bert-base (L12 H768 I3072 V30522) gives ~109M.
    """
    h = shape.hidden
    per_layer = 0

    # Attention: q, k, v, o projections. Grouped-query attention shrinks k and v.
    kv_ratio = (shape.kv_heads or shape.heads) / shape.heads
    per_layer += h * h                    # q
    per_layer += 2 * h * h * kv_ratio     # k, v
    per_layer += h * h                    # o

    # MLP: two matrices, or three for gated variants (SwiGLU/GeGLU).
    mlp_matrices = 3 if shape.gated_mlp else 2
    per_layer += mlp_matrices * h * shape.intermediate

    # Layer norms are negligible but cheap to include.
    per_layer += 4 * h

    total = per_layer * shape.layers

    # An encoder-decoder stacks a decoder on top, and every decoder layer carries a second
    # attention block (cross-attention) on top of its own self-attention.
    if shape.is_encoder_decoder:
        cross_attention = 2 * h * h + 2 * h * h * kv_ratio + 2 * h
        total += (per_layer + cross_attention) * shape.decoder_layers

    total += shape.vocab * h                      # token embeddings
    if not shape.tied_embeddings:
        total += shape.vocab * h                  # separate output head
    if shape.learned_positions and shape.max_position:
        total += shape.max_position * h           # learned position table (not RoPE)

    return int(total)
