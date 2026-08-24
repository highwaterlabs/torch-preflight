"""Measure the two constants that need real NVIDIA hardware.

Everything else in the cost model is measured by ``measure_activations.py`` on the meta
device with no GPU. These two are CUDA-allocator properties with no MPS or CPU equivalent:

* ``CUDA_CONTEXT_BYTES``     — driver state, kernels and cuBLAS/cuDNN workspaces, which
                               exist before your first tensor and are invisible to
                               ``torch.cuda.memory_allocated()``.
* ``FRAGMENTATION_FRACTION`` — the caching allocator reserves more than it hands out.
                               Measured as ``max_memory_reserved / max_memory_allocated``.

It also records end-to-end peaks for ``measured_peaks.json``, which is what finally checks
the whole cost model against reality rather than against itself.

Running it on a free Colab / Kaggle T4
--------------------------------------
Runtime → Change runtime type → T4 GPU, then in a cell:

    !pip install -q torch-preflight transformers torchvision
    !wget -qO measure_cuda.py "https://raw.githubusercontent.com/highwaterlabs/torch-preflight/main/tests/calibration/measure_cuda.py?$(date +%s)"
    !python measure_cuda.py --models

``-O`` matters. Plain ``wget`` will not overwrite an existing file — it silently writes
``measure_cuda.py.1`` and leaves the stale copy in place, so a re-run in the same session
executes the old script. The cache-busting query string defeats raw.githubusercontent's
CDN, which holds a copy for a few minutes after a push.

Or just paste this file into a cell and run it. It prints a JSON block to copy into
``tests/calibration/measured_peaks.json``.

Locally:

    python tests/calibration/measure_cuda.py --models
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    sys.exit(f"this script needs torch ({exc})")

MIB = 1024 ** 2
GIB = 1024 ** 3


NO_CUDA_MESSAGE = (
    "No CUDA device found.\n\n"
    "This script measures CUDA-allocator behaviour specifically — MPS and CPU have no\n"
    "equivalent, so there is genuinely nothing to measure here. Everything else in the\n"
    "cost model is already covered by measure_activations.py, which needs no GPU.\n\n"
    "Use a free Colab or Kaggle T4:  Runtime -> Change runtime type -> T4 GPU"
)

# Catch the modern typed error where present, and the plain RuntimeError on older torch.
OOM_ERRORS = tuple(
    e for e in (getattr(torch.cuda, "OutOfMemoryError", None), RuntimeError) if e
)


# --------------------------------------------------------------------- CUDA context


def measure_cuda_context() -> Dict[str, int]:
    """Device memory held before any tensor of ours exists.

    ``mem_get_info`` reports what the *driver* sees, so the gap between total-free and
    what PyTorch has reserved is the context plus any library workspaces. cuBLAS and cuDNN
    allocate theirs lazily on first use, so this is sampled in three stages.
    """
    torch.cuda.init()
    torch.cuda.empty_cache()

    def overhead() -> int:
        free, total = torch.cuda.mem_get_info()
        return (total - free) - torch.cuda.memory_reserved()

    stages = {"after_init": overhead()}

    # First matmul pulls in cuBLAS.
    a = torch.randn(256, 256, device="cuda")
    (a @ a).sum().item()
    torch.cuda.synchronize()
    stages["after_cublas"] = overhead()

    # First convolution pulls in cuDNN, which has the larger workspace of the two.
    conv = nn.Conv2d(3, 16, 3, padding=1).cuda()
    conv(torch.randn(1, 3, 64, 64, device="cuda")).sum().backward()
    torch.cuda.synchronize()
    stages["after_cudnn"] = overhead()

    del a, conv
    torch.cuda.empty_cache()
    return stages


# ------------------------------------------------------------------- toy transformer


class Block(nn.Module):
    """Same architecture as measure_activations.py, so the two runs are comparable."""

    def __init__(self, h, a, i, dropout=0.0):
        super().__init__()
        self.a = a
        bias = dropout > 0
        self.ln1 = nn.LayerNorm(h)
        self.q = nn.Linear(h, h, bias=bias)
        self.k = nn.Linear(h, h, bias=bias)
        self.v = nn.Linear(h, h, bias=bias)
        self.o = nn.Linear(h, h, bias=bias)
        self.ln2 = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, i, bias=bias)
        self.fc2 = nn.Linear(i, h, bias=bias)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, flash=False):
        b, s, h = x.shape
        d = h // self.a
        y = self.ln1(x)
        q = self.q(y).view(b, s, self.a, d).transpose(1, 2)
        k = self.k(y).view(b, s, self.a, d).transpose(1, 2)
        v = self.v(y).view(b, s, self.a, d).transpose(1, 2)
        if flash:
            att = F.scaled_dot_product_attention(q, k, v)
        else:
            att = self.drop(((q @ k.transpose(-2, -1)) / d ** 0.5).softmax(-1)) @ v
        x = x + self.drop(self.o(att.transpose(1, 2).reshape(b, s, h)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class Stack(nn.Module):
    def __init__(self, layers, h, a, i, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([Block(h, a, i, dropout) for _ in range(layers)])

    def forward(self, x, flash=False):
        for block in self.blocks:
            x = block(x, flash)
        return x


# ------------------------------------------------------------------------ the harness


@contextlib.contextmanager
def clean_gpu():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def training_peak(step, *, warmup=True) -> Optional[Dict[str, int]]:
    """Peak allocated/reserved for one steady-state training step.

    The warmup step matters: Adam allocates its state lazily inside the first
    ``.step()``, so measuring step 1 would miss the optimizer term entirely.
    """
    try:
        if warmup:
            step()
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()

        return {
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }
    except OOM_ERRORS as exc:
        if "out of memory" not in str(exc).lower() and type(exc) is RuntimeError:
            raise
        return None


def synthetic_case(layers, h, a, i, batch, seq, dropout, amp_dtype, flash):
    with clean_gpu():
        model = Stack(layers, h, a, i, dropout).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(batch, seq, h, device="cuda")

        def step():
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(x, flash=flash).float().pow(2).mean()
            else:
                loss = model(x, flash=flash).pow(2).mean()
            loss.backward()
            optimizer.step()

        result = training_peak(step)
        if result is not None:
            result["params"] = sum(p.numel() for p in model.parameters())
        del model, optimizer, x
        return result


# ------------------------------------------------------------------- real HF models

#: (archdb key, hub id, loader kind). Chosen so the parameter count matches the entry in
#: torch-preflight's bundled snapshot — otherwise the fixture compares different things.
REAL_MODELS = [
    ("gpt2", "gpt2", "causal-lm"),
    ("distilbert-base-uncased", "distilbert-base-uncased", "encoder"),
    ("bert-base-uncased", "bert-base-uncased", "encoder"),
    # Vision activations were measured on the meta device but have never been checked
    # against a real peak. This is that check.
    ("resnet50", "resnet50", "vision"),
]


def vision_case(name, batch, size, amp_dtype):
    """Peak for a torchvision classifier, to validate the measured CNN activations."""
    try:
        import torchvision.models as tvm
    except ImportError:
        return None

    with clean_gpu():
        model = getattr(tvm, name)().cuda()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(batch, 3, size, size, device="cuda")
        target = torch.randint(0, 1000, (batch,), device="cuda")

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext():
                loss = torch.nn.functional.cross_entropy(model(x), target)
            loss.backward()
            optimizer.step()

        result = training_peak(step)
        if result is not None:
            result["params"] = sum(p.numel() for p in model.parameters())
        del model, optimizer, x, target
        return result


def lm_head_sweep(amp_dtype):
    """Isolate the per-logit cost by varying vocabulary with everything else fixed.

    The LM-head backward transient is currently fitted on two data points, which is not
    enough — sweeping it lowers error monotonically, the signature of absorbing a
    systematic bias rather than converging. Holding the body constant and moving only the
    vocabulary makes the b*s*vocab coefficient separable.
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        return []

    rows = []
    for vocab in (8000, 32000, 50257, 128256):
        for batch, seq in ((4, 256), (8, 256)):
            with clean_gpu():
                cfg = GPT2Config(n_layer=4, n_embd=256, n_head=4,
                                 vocab_size=vocab, n_positions=seq)
                model = GPT2LMHeadModel(cfg).cuda()
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                ids = torch.randint(0, vocab - 1, (batch, seq), device="cuda")

                def step():
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext():
                        loss = model(input_ids=ids, labels=ids).loss
                    loss.backward()
                    optimizer.step()

                result = training_peak(step)
                if result is not None:
                    rows.append({
                        "vocab": vocab, "batch": batch, "seq": seq,
                        "logit_elements": batch * seq * vocab,
                        "params": sum(p.numel() for p in model.parameters()),
                        **result,
                    })
                del model, optimizer, ids
    return rows


def guard_accuracy_case(amp_dtype):
    """Check VRAMGuard's projection against what the run actually used."""
    try:
        from torch_preflight import VRAMGuard
        import torchvision.models as tvm
    except ImportError:
        return None

    with clean_gpu():
        model = tvm.resnet50().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(32, 3, 224, 224, device="cuda")
        target = torch.randint(0, 1000, (32,), device="cuda")
        try:
            with VRAMGuard(model, optimizer=optimizer, batch_size=32,
                           image_size=224, precision="amp") as guard:
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=amp_dtype):
                        loss = torch.nn.functional.cross_entropy(model(x), target)
                    loss.backward()
                    optimizer.step()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        out = {
            "projected_bytes": guard.report.total if guard.report else None,
            "measured_peak_bytes": guard.measured_peak,
            "accuracy": guard.accuracy,
        }
        del model, optimizer, x, target
        return out


def real_model_case(hub_id, kind, batch, seq, amp_dtype):
    if kind == "vision":
        return vision_case(hub_id, batch, seq, amp_dtype)
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except ImportError:
        return None

    with clean_gpu():
        if kind == "causal-lm":
            model = AutoModelForCausalLM.from_pretrained(hub_id).cuda()
        else:
            model = AutoModel.from_pretrained(hub_id).cuda()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        vocab = getattr(model.config, "vocab_size", 30000)
        ids = torch.randint(0, vocab - 1, (batch, seq), device="cuda")

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext():
                out = model(input_ids=ids, labels=ids) if kind == "causal-lm" else model(input_ids=ids)
                loss = out.loss if kind == "causal-lm" else out.last_hidden_state.float().pow(2).mean()
            loss.backward()
            optimizer.step()

        result = training_peak(step)
        if result is not None:
            result["params"] = sum(p.numel() for p in model.parameters())
        del model, optimizer, ids
        return result


# ------------------------------------------------------------------------------ main


def _unsupported_architecture() -> Optional[str]:
    """Can this PyTorch build actually launch a kernel on this card?

    `torch.cuda.is_available()` returns True for a GPU whose compute capability the build
    has no kernels for, and the failure then surfaces as `no kernel image is available for
    execution on the device` from the first real op — a CUDA error several frames away from
    the cause.

    Found on Kaggle's free P100: torch 2.10+cu128 ships sm_70 upward, and Pascal is sm_60.
    PyTorch has dropped Pascal, so the free-P100 route that this project's own calibration
    issue recommended cannot work at all.
    """
    try:
        major, minor = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
    except Exception:  # pragma: no cover - older torch, or a driver that will not answer
        return None

    architectures = {
        int(arch[3:].rstrip("a+")) for arch in supported
        if arch.startswith("sm_") and arch[3:].rstrip("a+").isdigit()
    }
    if not architectures:
        return None

    capability = major * 10 + minor
    if capability in architectures:
        return None

    name = torch.cuda.get_device_name(0)
    return (
        f"{name} is compute capability sm_{capability}, and this PyTorch build only ships "
        f"kernels for {', '.join('sm_%d' % a for a in sorted(architectures))}.\n\n"
        f"No kernel can launch, so there is nothing to measure — every op would fail with\n"
        f"'no kernel image is available for execution on the device'.\n\n"
        f"This is not a driver problem and reinstalling torch on the same card will not fix\n"
        f"it: PyTorch has dropped support for this architecture. Use a newer card.\n\n"
        f"  Colab   Runtime -> Change runtime type -> L4 (Ada) or A100 (Ampere, Pro)\n"
        f"  Kaggle  Settings -> Accelerator -> GPU T4 x2 (Turing)"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true",
                        help="also measure real HF models (downloads weights)")
    parser.add_argument("--out", default="measured_cuda.json")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print(NO_CUDA_MESSAGE, file=sys.stderr)
        return 1

    unsupported = _unsupported_architecture()
    if unsupported is not None:
        print(unsupported, file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0)
    _, total = torch.cuda.mem_get_info()
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    print(f"GPU          {name}")
    print(f"total memory {total / GIB:.2f} GiB")
    print(f"torch        {torch.__version__}  cuda {torch.version.cuda}")
    print(f"bf16         {'yes' if bf16 else 'no (using fp16 for AMP)'}")

    # ---------------------------------------------------------------- context

    print("\n" + "=" * 74)
    print("CUDA context — memory held before any tensor of ours exists")
    print("=" * 74)
    context = measure_cuda_context()
    for stage, value in context.items():
        print(f"  {stage:<16} {value / MIB:8.0f} MiB")
    context_bytes = max(context.values())
    print(f"\n  -> CUDA_CONTEXT_BYTES = {context_bytes / MIB:.0f} MiB "
          f"({context_bytes} bytes)")

    # ----------------------------------------------------------- fragmentation

    print("\n" + "=" * 74)
    print("Fragmentation — reserved / allocated across a range of shapes")
    print("=" * 74)

    configs = [
        # layers, hidden, heads, intermediate, batch, seq, dropout, flash
        (4, 512, 8, 2048, 4, 256, 0.0, False),
        (6, 768, 12, 3072, 8, 512, 0.1, False),
        (6, 768, 12, 3072, 4, 1024, 0.0, True),
        (12, 768, 12, 3072, 2, 512, 0.0, False),
        (4, 1024, 16, 4096, 2, 1024, 0.0, True),
    ]

    fragmentation: List[float] = []
    synthetic_rows = []
    for layers, h, a, i, batch, seq, dropout, flash in configs:
        result = synthetic_case(layers, h, a, i, batch, seq, dropout, amp_dtype, flash)
        label = f"L{layers} h{h} b{batch} s{seq}{' flash' if flash else ''}"
        if result is None:
            print(f"  {label:<32} OOM, skipped")
            continue
        ratio = result["max_reserved"] / result["max_allocated"]
        fragmentation.append(ratio - 1)
        synthetic_rows.append({
            "label": label, "layers": layers, "hidden": h, "heads": a,
            "intermediate": i, "batch": batch, "seq": seq,
            "dropout": dropout, "flash": flash, **result,
        })
        print(f"  {label:<32} allocated {result['max_allocated'] / GIB:6.2f} GiB   "
              f"reserved {result['max_reserved'] / GIB:6.2f} GiB   +{(ratio - 1) * 100:5.1f}%")

    if fragmentation:
        mean = sum(fragmentation) / len(fragmentation)
        print(f"\n  -> FRAGMENTATION_FRACTION = {mean:.3f} "
              f"(range {min(fragmentation):.3f} – {max(fragmentation):.3f})")
    else:
        mean = None
        print("\n  -> no configs fit; cannot measure fragmentation")

    # ------------------------------------------------------------- real models

    peaks = []
    if args.models:
        print("\n" + "=" * 74)
        print("End-to-end peaks for measured_peaks.json")
        print("=" * 74)
        for key, hub_id, kind in REAL_MODELS:
            shapes = ((16, 224), (32, 224)) if kind == "vision" else ((4, 128), (8, 256))
            for batch, seq in shapes:
                result = real_model_case(hub_id, kind, batch, seq, amp_dtype)
                if result is None:
                    print(f"  {key:<26} b{batch} s{seq}   OOM or transformers missing")
                    continue
                # What the cost model predicts is total device occupancy: the allocator's
                # reservation (which already includes fragmentation) plus the context.
                peak = result["max_reserved"] + context_bytes
                peaks.append({
                    "model": key,
                    "gpu": _gpu_key(name, total),
                    "config": {
                        "batch_size": batch,
                        ("image_size" if kind == "vision" else "seq_len"): seq,
                        "precision": "amp",
                        "optimizer": "adamw",
                        "gradient_checkpointing": False,
                        "flash_attention": False,
                    },
                    "measured_peak_bytes": peak,
                    "measured_allocated_bytes": result["max_allocated"],
                    "measured_reserved_bytes": result["max_reserved"],
                    "cuda_context_bytes": context_bytes,
                    "params": result["params"],
                    "torch_version": torch.__version__,
                    "note": f"{name}, reserved + context, steady-state step",
                })
                print(f"  {key:<26} b{batch} s{seq}   peak {peak / GIB:6.2f} GiB "
                      f"(allocated {result['max_allocated'] / GIB:.2f})")

    # ------------------------------------------------------------- LM head sweep

    lm_rows = []
    if args.models:
        print("\n" + "=" * 74)
        print("LM-head cost per logit element (vocabulary sweep)")
        print("=" * 74)
        lm_rows = lm_head_sweep(amp_dtype)
        if lm_rows:
            print(f"  {'vocab':>8}{'batch':>7}{'seq':>6}{'logits':>14}"
                  f"{'peak GiB':>11}{'d(peak)/d(logit)':>18}")
            base = None
            for r in sorted(lm_rows, key=lambda x: (x["batch"], x["vocab"])):
                peak = r["max_reserved"] + context_bytes
                slope = ""
                if base and base["batch"] == r["batch"]:
                    d_peak = peak - base["peak"]
                    d_elem = r["logit_elements"] - base["logit_elements"]
                    if d_elem:
                        slope = f"{d_peak / d_elem:.2f} B"
                print(f"  {r['vocab']:>8}{r['batch']:>7}{r['seq']:>6}"
                      f"{r['logit_elements']:>14,}{peak / GIB:>11.2f}{slope:>18}")
                base = {**r, "peak": peak}
            print("\n  The slope is the per-logit cost, transient included. Compare with"
                  "\n  LM_HEAD_RETAINED_BYTES + LM_HEAD_BACKWARD_TRANSIENT_BYTES.")
        else:
            print("  transformers not installed — skipped")

    # ------------------------------------------------------------- VRAMGuard

    guard = None
    if args.models:
        print("\n" + "=" * 74)
        print("VRAMGuard: projection vs what the run actually used")
        print("=" * 74)
        guard = guard_accuracy_case(amp_dtype)
        if not guard:
            print("  torch-preflight or torchvision not installed — skipped")
        elif guard.get("error"):
            print(f"  {guard['error']}")
        else:
            print(f"  projected {guard['projected_bytes'] / GIB:.2f} GiB   "
                  f"measured {guard['measured_peak_bytes'] / GIB:.2f} GiB   "
                  f"error {guard['accuracy'] * 100:+.1f}%")

    # -------------------------------------------------- compare with torch-preflight

    try:
        from torch_preflight.vram import costmodel

        print("\n" + "=" * 74)
        print("Compared with the constants torch-preflight ships")
        print("=" * 74)
        print(f"  CUDA_CONTEXT_BYTES      shipped {costmodel.CUDA_CONTEXT_BYTES / MIB:6.0f} MiB"
              f"   measured {context_bytes / MIB:6.0f} MiB")

        # Two fragmentation numbers, and only one of them is the right comparison.
        # Toy stacks allocate a handful of uniform tensors and under-fragment; real models
        # churn through embeddings, masks and head projections of many sizes. The shipped
        # constant is calibrated against the real ones, so show both and say which counts.
        real = [r["measured_reserved_bytes"] / r["measured_allocated_bytes"] - 1
                for r in peaks if r.get("measured_allocated_bytes")]
        if mean is not None:
            print(f"  FRAGMENTATION_FRACTION  shipped {costmodel.FRAGMENTATION_FRACTION:6.3f}"
                  f"       synthetic {mean:6.3f}  (expected lower — see below)")
        if real:
            print(f"  {'':<24}{'':<14} real models {sum(real)/len(real):6.3f}"
                  f"  <- the one the constant is fitted to")
        elif args.models:
            print("  (no real-model runs completed, so no comparison for the constant)")
    except ImportError:
        print("\n(torch-preflight not installed here — skipping the comparison)")

    # ------------------------------------------------------------------ output

    payload = {
        "gpu": name,
        "gpu_key": _gpu_key(name, total),
        "total_memory_bytes": total,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_context": context,
        "cuda_context_bytes": context_bytes,
        "fragmentation_samples": fragmentation,
        "fragmentation_fraction": mean,
        "synthetic_runs": synthetic_rows,
        "lm_head_sweep": lm_rows,
        "guard_accuracy": guard,
        "runs": peaks,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n" + "=" * 74)
    print(f"Wrote {args.out}")
    if peaks:
        print("Paste the block below into tests/calibration/measured_peaks.json "
              'under "runs":')
        print("=" * 74)
        print(json.dumps(peaks, indent=2))
    else:
        print("Re-run with --models to also produce measured_peaks.json fixtures.")
    return 0


def _gpu_key(name: str, total_bytes: Optional[int] = None) -> str:
    """Best-effort mapping from the device name to a torch-preflight hardware key.

    Board capacity disambiguates the parts that ship in two sizes — the device name alone
    does not distinguish a 40GB A100 from an 80GB one.
    """
    lowered = name.lower()
    gib = (total_bytes or 0) / GIB

    if "a100" in lowered:
        return "a100-80gb" if gib > 60 else "a100-40gb"
    if "v100" in lowered:
        return "v100-32gb" if gib > 24 else "v100-16gb"
    if "h100" in lowered:
        return "h100-94gb" if gib > 88 else "h100-80gb"

    for needle, key in (
        ("h200", "h200"), ("t4", "t4"), ("l40s", "l40s"), ("l4", "l4"),
        ("a10g", "a10g"), ("a6000", "a6000"), ("a40", "a40"),
        ("5090", "rtx5090"), ("4090", "rtx4090"), ("4080", "rtx4080"),
        ("3090", "rtx3090"), ("3080", "rtx3080"), ("2080", "rtx2080ti"),
    ):
        if needle in lowered:
            return key
    return "unknown"


if __name__ == "__main__":
    # In a notebook, ``__name__`` is also "__main__" but ``sys.argv`` belongs to the
    # kernel ("-f /root/.../kernel.json"), which argparse would reject. And a bare
    # SystemExit renders as an ugly traceback in a cell.
    if "ipykernel" in sys.modules:
        print("Notebook detected — running context + fragmentation only.")
        print("For end-to-end peaks as well (downloads model weights), run:")
        print("    main(['--models'])\n")
        main([])
    else:
        raise SystemExit(main())
