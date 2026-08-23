"""TG002 - evaluation/inference code running with autograd still enabled."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import libcst as cst

from ..analysis.helpers import contains_call_to, dotted_name
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Exact function names that mean "this routine evaluates, it does not train".
#: Kept as exact matches (plus the prefixes below) so pytest's ``test_*`` functions,
#: which legitimately exercise autograd, are never caught.
EVAL_NAMES = frozenset(
    {
        "validate", "validation", "val", "evaluate", "eval", "evaluation",
        "test", "test_epoch", "test_loop", "inference", "infer", "predict",
        "prediction", "sample", "generate", "score", "benchmark",
        "validate_epoch", "val_epoch", "eval_epoch", "run_eval", "run_validation",
    }
)

EVAL_PREFIXES = (
    "validate_", "validation_", "val_", "evaluate_", "eval_", "predict_", "inference_",
)

#: Lightning runs these hooks under ``torch.no_grad()`` itself.
LIGHTNING_EVAL_HOOKS = frozenset({"validation_step", "test_step", "predict_step"})

VAL_LOADER_HINTS = ("val", "valid", "test", "eval", "dev", "holdout")

#: An eval hint alone is not enough. In a test suite half the identifiers contain "test"
#: — `models_to_test` in torch's own distributed tests was being read as a validation
#: dataloader — so the name must also look like something you iterate batches from.
LOADER_TOKENS = ("loader", "dataset", "batches", "_data", "data_", "iterator")


def _is_pytest_module(path: str) -> bool:
    """Is this file part of a test suite?

    The rule already claims to exempt pytest's ``test_*`` functions, which legitimately
    exercise autograd — but only through the *name* path. A test that calls ``model.eval()``
    and then does a forward was still reported, which is 29 of `peft`'s findings and most of
    `composer`'s.

    Matched on the module rather than the function name because six of those 29 sit in
    `_check_inference_finite`, a helper that no name-based rule would catch. What makes them
    exempt is that the file is a test suite, not what the function happens to be called.
    """
    parts = path.replace("\\", "/").split("/")
    if any(part in ("tests", "test", "testing") for part in parts[:-1]):
        return True
    name = parts[-1]
    return name.startswith("test_") or name.endswith(("_test.py", "_tests.py"))


def _is_eval_name(name: str) -> bool:
    lowered = name.lower().lstrip("_")
    return lowered in EVAL_NAMES or lowered.startswith(EVAL_PREFIXES)


def _looks_like_eval_loader(name: Optional[str]) -> bool:
    if not name:
        return False
    lowered = name.lower()
    if "train" in lowered:
        return False
    if not any(token in lowered for token in LOADER_TOKENS):
        return False
    return any(hint in lowered for hint in VAL_LOADER_HINTS)


@dataclass
class _Unit:
    """One function (or the module top level) considered as a whole."""

    node: Optional[cst.CSTNode]
    name: str
    eval_name: bool = False
    has_backward: bool = False
    #: Flipped by ``model.eval()`` / ``model.train()`` as we walk the body.
    in_eval_mode: bool = False
    #: ``(call node, came from a val/test loop)`` for each unguarded forward pass.
    candidates: List[Tuple[cst.CSTNode, bool]] = field(default_factory=list)


@register
class MissingNoGrad(Rule):
    code = "TG002"
    name = "missing-no-grad"
    summary = "Evaluation runs a model without torch.no_grad()/inference_mode()"
    severity = Severity.ERROR
    category = Category.CRITICAL_OOM
    explanation = """
Running a forward pass with autograd enabled builds a computational graph and keeps every
intermediate activation alive so gradients *could* be computed later. During validation or
inference you never call ``.backward()``, so that graph is pure waste - it commonly doubles
or triples peak VRAM and slows the pass down.

Wrap the loop in ``with torch.no_grad():``, or decorate the routine with
``@torch.inference_mode()``, which additionally disables version counter bookkeeping.
""".strip()

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._units: List[_Unit] = []

    # ------------------------------------------------------------------- units

    def visit_Module(self, node: cst.Module) -> bool:
        self._units.append(_Unit(node=None, name="<module>"))
        return True

    def leave_Module(self, original_node: cst.Module) -> None:
        self._finish(self._units.pop())

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name = node.name.value
        unit = _Unit(
            node=node,
            name=name,
            eval_name=_is_eval_name(name) and name != "forward",
            has_backward=contains_call_to(node, ["backward"]),
        )
        if self.ctx.is_lightning and name in LIGHTNING_EVAL_HOOKS:
            unit.has_backward = True  # suppress: Lightning guards these hooks itself
        self._units.append(unit)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._finish(self._units.pop())

    # ------------------------------------------------------------------ signals

    def visit_Call(self, node: cst.Call) -> bool:
        if not self._units:
            return True
        unit = self._units[-1]
        func = node.func

        # ``model.eval()`` / ``model.train()`` switch the surrounding intent.
        if isinstance(func, cst.Attribute) and func.attr.value in ("eval", "train"):
            if self.prov.is_model(dotted_name(func.value), self.scope_path):
                unit.in_eval_mode = func.attr.value == "eval"
            return True

        # A direct model invocation: ``model(x)`` or ``model.forward(x)``.
        target: cst.BaseExpression = func
        if isinstance(func, cst.Attribute):
            if func.attr.value != "forward":
                return True
            target = func.value

        if self.in_no_grad or not self.prov.is_model(dotted_name(target), self.scope_path):
            return True

        from_eval_loop = any(_looks_like_eval_loader(f.iterable) for f in self.loops)
        # ...unless that very loop does the backward pass. `fgsm_tutorial.py` iterates
        # `test_loader` and backwards through it on purpose, because an adversarial attack
        # needs gradients with respect to the input. Reporting it said the function "never
        # calls `.backward()`" while the call sat nine lines below.
        if from_eval_loop and any(
            _looks_like_eval_loader(f.iterable) and contains_call_to(f.node, ["backward"])
            for f in self.loops
        ):
            from_eval_loop = False
        if unit.eval_name or unit.in_eval_mode or from_eval_loop:
            unit.candidates.append((node, from_eval_loop))
        return True

    # ------------------------------------------------------------------ verdict

    def _finish(self, unit: _Unit) -> None:
        # A test suite is not a training run. Retaining a graph for a forward that a test
        # never backwards costs nothing worth reporting, and the rule already promised this
        # exemption for `test_*` functions -- it just never applied it past the name check.
        if _is_pytest_module(self.ctx.path):
            return

        # A routine that calls ``.backward()`` is training, not evaluating - unless the
        # forward pass we found sits in an explicit validation loop inside it.
        candidates = [
            node for node, from_loop in unit.candidates if from_loop or not unit.has_backward
        ]
        if not candidates:
            return

        where = f"`{unit.name}()`" if unit.node is not None else "this module"
        # Only decorate a function whose *name* says it evaluates. Adding
        # ``@torch.no_grad()`` to a training routine that happens to contain a
        # validation loop would silently break it.
        fixable = unit.eval_name and unit.node is not None and self.ctx.has_torch_import

        self.report(
            candidates[0],
            f"{where} runs a forward pass with autograd enabled but never calls "
            f"`.backward()`; the graph and all activations are retained for nothing.",
            hint="Wrap the loop in `with torch.no_grad():` or decorate the function with "
            "`@torch.inference_mode()`.",
            fix_node=unit.node if fixable else None,
            fix_build=_add_no_grad_decorator if fixable else None,
            fix_description="add @torch.no_grad()",
        )


def _add_no_grad_decorator(updated: cst.CSTNode) -> cst.CSTNode:
    assert isinstance(updated, cst.FunctionDef)
    decorator = cst.Decorator(decorator=cst.parse_expression("torch.no_grad()"))
    return updated.with_changes(decorators=[decorator, *updated.decorators])
