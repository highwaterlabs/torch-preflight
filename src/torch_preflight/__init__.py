"""torch-preflight: a static analyzer for PyTorch training code.

Catches VRAM leaks, silent convergence bugs and GPU pipeline stalls before a training
run is launched, rather than three hours into it.
"""

from .config import Config, load_config
from .diagnostics import Category, Diagnostic, Severity
from .engine import Result, check_paths, check_source

__version__ = "0.5.0"

#: Exported lazily so that ``import torch_preflight`` never pulls in torch. ``VRAMGuard`` is a
#: runtime tool and needs the ``[vram]`` extra; everything else here is dependency-free.
_LAZY_EXPORTS = {
    "VRAMGuard": ".vram.guard",
    "VramRiskError": ".vram.guard",
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(target, __name__), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY_EXPORTS))

__all__ = [
    "Category",
    "VRAMGuard",
    "VramRiskError",
    "Config",
    "Diagnostic",
    "Result",
    "Severity",
    "__version__",
    "check_paths",
    "check_source",
    "load_config",
]
