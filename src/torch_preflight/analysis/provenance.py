"""Grad-provenance analysis: which expressions carry a live autograd graph?

A plain AST linter sees ``losses.append(loss)`` and knows nothing. To decide whether
that leaks VRAM we need to know what ``loss`` *is*. This module answers that with a
small, deliberately coarse dataflow pass:

1. **Collect** every assignment, plus seeds: names ``.backward()`` was called on,
   results of criterion/model calls, and tensors built with ``requires_grad=True``.
2. **Propagate** to a fixpoint across assignment edges, refusing to propagate through
   graph-severing operations (``.detach()``, ``.item()``, ``float(...)``).

The analysis is flow-insensitive within a scope (a name that is ever grad-bearing is
treated as grad-bearing) and scope-sensitive across functions. That trade lets us keep
the whole thing linear and dependency-free while catching the patterns that actually
cost money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import libcst as cst

from .helpers import (
    DETACHING_BUILTINS,
    DETACHING_METHODS,
    NON_DIFFERENTIABLE_METHODS,
    base_name,
    dotted_name,
    final_attr,
    is_literal_true,
    keyword_arg,
)
from .scope import ScopePath, ScopeTrackingVisitor, target_names

# Variable names ML code conventionally uses for the model under training.
MODEL_NAME_HINTS = frozenset(
    {
        "model", "net", "network", "module", "backbone", "encoder", "decoder",
        "generator", "discriminator", "policy", "critic", "actor", "student",
        "teacher", "classifier", "transformer", "gen", "disc", "ema_model",
    }
)

#: `from_pretrained` classes that load a *preprocessor* rather than a model. Calling one is
#: tokenization or feature extraction, not a forward pass, and no autograd is involved.
NOT_A_MODEL_HINTS = (
    "tokenizer", "featureextractor", "imageprocessor", "processor", "extractor",
    "config", "scheduler",
    # A diffusers/transformers *pipeline* runs inference and hands back PIL images, strings
    # or dicts -- not a tensor with a graph. `DiffusionPipeline.from_pretrained(...)` then
    # `images.append(pipeline(prompt, num_inference_steps=25))` was reported as a retained
    # autograd graph in six of peft's dreambooth examples.
    "pipeline",
)

# Variable names conventionally holding a loss function.
CRITERION_NAME_HINTS = frozenset(
    {"criterion", "loss_fn", "loss_func", "lossfn", "loss_function", "objective", "crit"}
)

# Variable names that are almost always a live loss tensor.
LOSS_NAME_HINTS = frozenset(
    {
        "loss", "losses", "total_loss", "train_loss", "val_loss", "valid_loss",
        "test_loss", "batch_loss", "running_loss", "epoch_loss", "cost",
        "loss_val", "l", "total", "sum_loss", "loss_sum", "accum_loss",
    }
)

# ``torch.nn.functional`` losses. Anything ending in ``_loss`` is also treated as one.
FUNCTIONAL_LOSSES = frozenset(
    {
        "cross_entropy", "binary_cross_entropy", "binary_cross_entropy_with_logits",
        "nll_loss", "kl_div", "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
        "ctc_loss", "cosine_embedding_loss", "hinge_embedding_loss", "margin_ranking_loss",
        "multi_margin_loss", "soft_margin_loss", "triplet_margin_loss", "poisson_nll_loss",
        "gaussian_nll_loss", "kl_div_loss",
    }
)

# Factory functions whose output is grad-bearing only with ``requires_grad=True``.
TENSOR_FACTORIES = frozenset(
    {
        "tensor", "zeros", "ones", "randn", "rand", "randint", "empty", "full",
        "arange", "linspace", "eye", "zeros_like", "ones_like", "randn_like",
        "empty_like", "full_like", "as_tensor", "from_numpy",
    }
)

# Methods that belong to ``nn.Module``, not to a tensor. Calling them yields a module.
MODULE_METHODS = frozenset(
    {
        "eval", "train", "parameters", "named_parameters", "buffers", "named_buffers",
        "state_dict", "load_state_dict", "zero_grad", "requires_grad_", "apply",
        "modules", "named_modules", "children", "named_children", "register_buffer",
        "register_parameter", "add_module", "compile", "share_memory",
    }
)

# Calls that wrap a model and return a model.
MODEL_WRAPPERS = frozenset(
    {
        "DistributedDataParallel", "DataParallel", "DDP", "FSDP",
        "FullyShardedDataParallel", "compile",
        "from_pretrained", "create_model", "to_empty",
    }
)

# ``accelerator.prepare(model)`` returns a model; a bare ``prepare(...)`` is far too
# common a name to claim (torch's quantization utilities call theirs exactly that).
QUALIFIED_MODEL_WRAPPERS = frozenset({"prepare", "prepare_model"})
QUALIFIED_WRAPPER_RECEIVERS = ("accel", "fabric")

#: Receivers whose ``.backward(loss)`` takes the tensor as an argument rather than being
#: one. Accelerate, Lightning Fabric and DeepSpeed engines all use this shape.
BACKWARD_WRAPPER_RECEIVERS = ("accel", "fabric", "scaler", "engine", "strategy")

# Aggregations that preserve the graph, so grad flows from their arguments.
PROPAGATING_BUILTINS = frozenset({"sum", "min", "max", "abs", "sorted", "next", "iter"})


@dataclass
class _Assignment:
    scope: ScopePath
    targets: List[str]
    value: cst.BaseExpression
    #: ``loss += f(x)`` rather than ``loss = f(x)``. The name-based heuristic does not
    #: apply to accumulators: whether they carry a graph depends on the right-hand side.
    augmented: bool = False
    #: Assigned inside ``torch.no_grad()`` / ``inference_mode()``. The *value* has no
    #: ``grad_fn``, whatever is done with it later — and the common idiom wraps only the
    #: forward, then uses the result after the block::
    #:
    #:     with torch.no_grad():
    #:         outputs = model(**batch)
    #:     losses.append(outputs.loss)      # outside the block, but still detached
    #:
    #: A positional "am I inside a no-grad block" check misses that, which is what made
    #: TG001 fire on the standard Hugging Face evaluation loop.
    no_grad: bool = False
    #: Ordered names when the target is a plain tuple of names, so ``output, loss =
    #: train(...)`` can be matched element-wise against the callee's ``return`` tuple.
    #: ``targets`` is flattened and loses the positions.
    unpacked: Tuple[str, ...] = ()
    #: Identities of the loops enclosing this assignment, outermost first. Used to tell two
    #: assignments to the same name apart when they live in sibling loops -- a training loop
    #: and an evaluation loop in one function, which is the shape that defeated `no_grad`.
    loops: Tuple[int, ...] = ()


@dataclass
class Provenance:
    """Query interface produced by :func:`analyze`."""

    grad: Set[Tuple[ScopePath, str]] = field(default_factory=set)
    #: Names bound in each scope, so an inner binding shadows an outer one.
    bindings: Dict[ScopePath, Set[str]] = field(default_factory=dict)
    #: Scope-qualified, like :attr:`grad`. A flat name set let `prepared = DDP(...)` in
    #: one function make an unrelated `prepared` a thousand lines away look like a model.
    models: Set[Tuple[ScopePath, str]] = field(default_factory=set)
    #: Scope-qualified, like :attr:`grad` and :attr:`models`. A flat name->class map let
    #: two functions that each bind ``criterion`` collide, so whichever was seen last
    #: decided the class for both — reporting a ``BCELoss`` call as ``BCEWithLogitsLoss``.
    criteria: Dict[Tuple[ScopePath, str], str] = field(default_factory=dict)
    module_classes: Set[str] = field(default_factory=set)
    #: Names that were explicitly detached somewhere (used only for hint wording).
    detached: Set[str] = field(default_factory=set)
    #: ``(scope, name) -> loop identities inside which the binding carried no graph``.
    #:
    #: Python has function scope, not block scope, so this cannot simply shadow the name.
    #: It says "within *this* loop, that binding held a detached value", which is what
    #: separates an evaluation loop from a training loop in the same function.
    #:
    #: Deliberately broader than ``no_grad``: any binding that does not evaluate as
    #: grad-bearing *in its own loop* is recorded, so a name is judged by the assignment
    #: that reaches it rather than by every assignment in the function. That also means an
    #: analysis gap can now actively silence a name rather than merely fail to seed it --
    #: the trade is fewer false positives against the risk of a quiet false negative, and
    #: it is the reason the rule suite and a seven-repo scan are both checked for movement.
    not_grad_in_loops: Dict[Tuple[ScopePath, str], Set[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ queries

    def is_grad_name(self, name: str, scope: ScopePath) -> bool:
        """Scope-aware lookup: walk from the innermost scope outward to module level.

        The walk stops at the first scope that *binds* the name. Without that, a nested
        helper with its own ``loss = 0.0`` would inherit grad-ness from an unrelated
        ``loss`` in the enclosing function — which is exactly what happened in torch's
        distributed tests.
        """
        for depth in range(len(scope), -1, -1):
            prefix = scope[:depth]
            if (prefix, name) in self.grad:
                return True
            if name in self.bindings.get(prefix, ()):
                return False  # shadowed here, and not grad-bearing at this level
        return False

    def is_model(self, name: Optional[str], scope: Optional[ScopePath] = None) -> bool:
        """Is this name a model? Scope-aware when a scope is supplied.

        Falls back to the naming convention, which is scope-free by nature: a parameter
        called ``model`` is a model wherever it appears.
        """
        if not name:
            return False
        by_convention = name.rsplit(".", 1)[-1] in MODEL_NAME_HINTS

        if scope is None:
            return any(known == name for _, known in self.models) or by_convention

        for depth in range(len(scope), -1, -1):
            prefix = scope[:depth]
            if (prefix, name) in self.models:
                return True
            if name in self.bindings.get(prefix, ()):
                return by_convention  # bound here as something else; stop walking out
        return by_convention

    def is_criterion(self, name: Optional[str], scope: Optional[ScopePath] = None) -> bool:
        if not name:
            return False
        if self.criterion_class(name, scope) is not None:
            return True
        return name.rsplit(".", 1)[-1] in CRITERION_NAME_HINTS

    def criterion_class(
        self, name: Optional[str], scope: Optional[ScopePath] = None
    ) -> Optional[str]:
        """Which loss class this name was bound to, scope-aware when a scope is given.

        Walks innermost-outward and stops at the first scope that binds the name, exactly
        as :meth:`is_grad_name` does. Without the walk, a file with ``crit =
        nn.BCELoss()`` in one function and ``crit = nn.BCEWithLogitsLoss()`` in another
        resolved both to whichever was visited last.
        """
        if not name:
            return None
        if scope is None:
            for (_, known), cls in self.criteria.items():
                if known == name:
                    return cls
            return None

        for depth in range(len(scope), -1, -1):
            prefix = scope[:depth]
            found = self.criteria.get((prefix, name))
            if found is not None:
                return found
            if name in self.bindings.get(prefix, ()):
                return None  # bound here as something else; stop walking out
        return None

    def is_grad_bearing(
        self,
        node: cst.BaseExpression,
        scope: ScopePath,
        loops: Sequence[int] = (),
    ) -> bool:
        """Does evaluating ``node`` here yield a tensor attached to an autograd graph?

        ``loops`` identifies the loops enclosing the *use*, so a name bound under
        ``no_grad`` in one loop is not read as grad-bearing there merely because a sibling
        loop binds the same name normally.
        """
        return _ExprGrad(self, scope, tuple(loops)).check(node)

    def detached_here(self, name: str, scope: ScopePath, loops: Sequence[int]) -> bool:
        """Was ``name`` bound under ``no_grad`` inside one of the loops we are in?

        Scoped the same way as :meth:`is_grad_name`: walk innermost-outward. Requires an
        enclosing loop in common, so a `no_grad` block elsewhere in the function does not
        silence a name that a training loop legitimately rebinds.
        """
        if not loops:
            return False
        for depth in range(len(scope), -1, -1):
            found = self.not_grad_in_loops.get((scope[:depth], name))
            if found and found.intersection(loops):
                return True
        return False

    def is_explicitly_detached(self, node: cst.BaseExpression) -> bool:
        """True if the outermost operation of ``node`` severs the graph."""
        if isinstance(node, cst.Call):
            attr = node.func
            if isinstance(attr, cst.Attribute) and attr.attr.value in DETACHING_METHODS:
                return True
            if isinstance(attr, cst.Name) and attr.value in DETACHING_BUILTINS:
                return True
        return False


class _ExprGrad:
    """Recursive expression evaluator for grad-bearing-ness."""

    def __init__(
        self, prov: Provenance, scope: ScopePath, loops: Tuple[int, ...] = ()
    ) -> None:
        self.prov = prov
        self.scope = scope
        self.loops = loops

    def check(self, node: Optional[cst.CSTNode], depth: int = 0) -> bool:
        if node is None or depth > 24:
            return False

        if isinstance(node, cst.Name):
            if self.prov.detached_here(node.value, self.scope, self.loops):
                return False
            return self.prov.is_grad_name(node.value, self.scope)

        if isinstance(node, cst.Attribute):
            dotted = dotted_name(node)
            base = base_name(node)
            if base and self.prov.detached_here(base, self.scope, self.loops):
                return False  # ``outputs.loss`` where ``outputs`` came from a no_grad forward
            if dotted and self.prov.is_grad_name(dotted, self.scope):
                return True
            # ``x.grad``/``x.data`` are detached views; everything else inherits.
            if node.attr.value in {"grad", "data", "shape", "dtype", "device", "ndim"}:
                return False
            return self.check(node.value, depth + 1)

        if isinstance(node, cst.Call):
            return self._check_call(node, depth)

        if isinstance(node, cst.BinaryOperation):
            return self.check(node.left, depth + 1) or self.check(node.right, depth + 1)

        if isinstance(node, cst.UnaryOperation):
            return self.check(node.expression, depth + 1)

        if isinstance(node, (cst.Comparison, cst.BooleanOperation)):
            return False

        if isinstance(node, (cst.Tuple, cst.List, cst.Set)):
            return any(self.check(e.value, depth + 1) for e in node.elements)

        if isinstance(node, cst.Dict):
            return any(
                isinstance(e, cst.DictElement) and self.check(e.value, depth + 1)
                for e in node.elements
            )

        if isinstance(node, cst.Subscript):
            return self.check(node.value, depth + 1)

        if isinstance(node, cst.IfExp):
            return self.check(node.body, depth + 1) or self.check(node.orelse, depth + 1)

        if isinstance(node, cst.Await):
            return self.check(node.expression, depth + 1)

        if isinstance(node, (cst.ListComp, cst.SetComp, cst.GeneratorExp)):
            return self.check(node.elt, depth + 1)

        return False

    def _check_call(self, node: cst.Call, depth: int) -> bool:
        func = node.func
        dotted = dotted_name(func)

        if isinstance(func, cst.Attribute):
            receiver = func.value
            method = func.attr.value

            if method in DETACHING_METHODS or method == "backward":
                return False

            # ``torch.autograd.grad(...)`` returns detached gradients unless it is asked
            # to build a graph over them. Storing the result retains nothing.
            if method == "grad":
                arg = keyword_arg(node, "create_graph")
                return arg is not None and is_literal_true(arg.value)

            # ``logits.argmax(-1)`` and friends return indices/masks, not graph nodes.
            if method in NON_DIFFERENTIABLE_METHODS:
                return False

            # A loss functional: ``F.cross_entropy(...)``, ``F.mse_loss(...)``.
            if method in FUNCTIONAL_LOSSES or method.endswith("_loss"):
                return True

            # Tensor factories are grad-bearing only when asked to be.
            if method in TENSOR_FACTORIES:
                return _has_requires_grad(node)

            # Module bookkeeping methods return the module, never a tensor.
            if method in MODULE_METHODS:
                return False

            # Calling the model itself: ``model(x)`` handled below; ``model.forward(x)``.
            if self.prov.is_model(dotted_name(receiver), self.scope):
                return method == "forward" or method == "__call__"

            if self.prov.is_criterion(dotted):
                return True

            # Generic tensor method (``out.view()``, ``torch.stack(losses)``):
            # the graph flows from the receiver or from any argument.
            if self.check(receiver, depth + 1):
                return True
            return self._any_arg_grad(node, depth)

        if isinstance(func, cst.Name):
            name = func.value

            if name in DETACHING_BUILTINS or name in NON_DIFFERENTIABLE_METHODS:
                return False

            if self.prov.is_model(name, self.scope) or self.prov.is_criterion(name):
                return True

            # Only the known torch functionals count when called by a bare name. A
            # `*_loss` suffix alone is not enough: `get_loss(x)` is a user helper, and
            # treating it as a loss producer regardless of its arguments made TG001 fire
            # on a plain float accumulator in torch's own tests. Unknown names ending in
            # "loss" still fall through to argument propagation below.
            if name in FUNCTIONAL_LOSSES:
                return True

            if name in TENSOR_FACTORIES:
                return _has_requires_grad(node)

            # Locally defined helpers named like loss producers. Requiring grad to flow
            # in from an argument matters: ``_multilabelmarginloss_reference(input, target)``
            # in torch's own test utilities matches "loss" but is handed plain tensors.
            # Loss variables assigned from an unknown call are still seeded by name in
            # ``analyze()``, so this does not lose the common case.
            lowered = name.lower()
            if "loss" in lowered or lowered.startswith(("compute_", "forward")):
                return self._any_arg_grad(node, depth)

            if name in PROPAGATING_BUILTINS:
                return self._any_arg_grad(node, depth)

            # Unknown callable: assume it forwards its arguments' graph.
            return self._any_arg_grad(node, depth)

        return False

    def _any_arg_grad(self, node: cst.Call, depth: int) -> bool:
        return any(self.check(arg.value, depth + 1) for arg in node.args)


def _has_requires_grad(node: cst.Call) -> bool:
    arg = keyword_arg(node, "requires_grad")
    return arg is not None and is_literal_true(arg.value)


class _Collector(ScopeTrackingVisitor):
    """Pass 1: gather assignments, model/criterion bindings and grad seeds."""

    def __init__(self) -> None:
        super().__init__()
        self.assignments: List[_Assignment] = []
        self.prov = Provenance()
        self.seeds: Set[Tuple[ScopePath, str]] = set()
        #: ``function name -> [(scope it returns in, returned expression)]``. Enough to tell
        #: a caller that ``train()`` hands back ``loss.item() / n`` -- a float -- rather than
        #: guessing from the caller's variable being named ``loss``.
        self.returns: Dict[str, List[Tuple[ScopePath, cst.BaseExpression]]] = {}

    # -- class definitions: which local classes are nn.Modules? ----------------

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        for base in node.bases:
            name = dotted_name(base.value) or ""
            if name.endswith(("Module", "LightningModule", "Sequential")):
                self.prov.module_classes.add(node.name.value)
        return True

    # -- assignments ------------------------------------------------------------

    def visit_Assign(self, node: cst.Assign) -> bool:
        self._pending_value = node.value
        targets: List[str] = []
        unpacked: Tuple[str, ...] = ()
        for target in node.targets:
            targets.extend(target_names(target.target))
            if isinstance(target.target, cst.Tuple):
                names = [
                    element.value.value
                    for element in target.target.elements
                    if isinstance(element.value, cst.Name)
                ]
                if len(names) == len(target.target.elements):
                    unpacked = tuple(names)
        self._record(targets, node.value, unpacked=unpacked)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if node.value is not None:
            self._record(target_names(node.target), node.value)
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        # ``total += loss`` makes ``total`` inherit whatever ``loss`` carries.
        self._record(target_names(node.target), node.value, augmented=True)
        return True

    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
        self._record(target_names(node.target), node.value)
        return True

    _pending_value: Optional[cst.BaseExpression] = None

    def _record(
        self,
        targets: List[str],
        value: cst.BaseExpression,
        augmented: bool = False,
        unpacked: Tuple[str, ...] = (),
    ) -> None:
        if not targets:
            return
        scope = self.scope_path
        loops = tuple(id(frame.node) for frame in self.loops)
        self.assignments.append(
            _Assignment(
                scope, targets, value, augmented, self.in_no_grad, unpacked, loops
            )
        )
        if self.in_no_grad and loops:
            for name in targets:
                self.prov.not_grad_in_loops.setdefault((scope, name), set()).update(loops)
        self.prov.bindings.setdefault(scope, set()).update(targets)

        for name in targets:
            if self._is_model_expr(value):
                self.prov.models.add((scope, name))
            loss_cls = self._criterion_class(value)
            if loss_cls:
                self.prov.criteria[(scope, name)] = loss_cls
            if isinstance(value, cst.Call) and _has_requires_grad(value):
                self.seeds.add((scope, name))
            if self.prov.is_explicitly_detached(value):
                self.prov.detached.add(name)

    def _is_model_expr(self, value: cst.BaseExpression) -> bool:
        if not isinstance(value, cst.Call):
            return False
        leaf = final_attr(value.func)
        dotted = dotted_name(value.func) or ""

        # `from_pretrained` loads preprocessors as well as models, and it is in
        # MODEL_WRAPPERS, so it matches before anything else can object. Reading
        # `AutoTokenizer.from_pretrained(...)` as a model made `tokenizer(x["question"])`
        # look like a forward pass, and TG002 duly reported a missing `no_grad` around
        # tokenization. Same for feature extractors and image processors.
        if any(hint in dotted.lower() for hint in NOT_A_MODEL_HINTS):
            return False

        if leaf in self.prov.module_classes or leaf in MODEL_WRAPPERS:
            return True
        if leaf in QUALIFIED_MODEL_WRAPPERS:
            receiver = dotted.rsplit(".", 1)[0].lower() if "." in dotted else ""
            return any(hint in receiver for hint in QUALIFIED_WRAPPER_RECEIVERS)
        if dotted.startswith(("nn.", "torch.nn.")) and leaf not in FUNCTIONAL_LOSSES:
            # ``nn.Linear(...)``, ``nn.Sequential(...)`` — but not ``nn.CrossEntropyLoss``.
            return not (leaf or "").endswith("Loss")
        if "models." in dotted or dotted.endswith(".from_pretrained"):
            return True
        # ``DDP(model)``, ``torch.compile(model)`` keep the wrapped object a model.
        if leaf in MODEL_WRAPPERS:
            return True
        return False

    def _criterion_class(self, value: cst.BaseExpression) -> Optional[str]:
        if not isinstance(value, cst.Call):
            return None
        leaf = final_attr(value.func)
        if leaf and leaf.endswith("Loss"):
            return leaf
        return None

    # -- return expressions, so a caller can be told what a local helper hands back

    def visit_Return(self, node: cst.Return) -> bool:
        function = self.current_function
        if function is not None and node.value is not None:
            self.returns.setdefault(function.name, []).append(
                (self.scope_path, node.value)
            )
        return True

    # -- seeds: anything ``.backward()`` is called on is definitively grad-bearing

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "backward":
            return True

        name = dotted_name(func.value)
        if not name:
            return True

        # `accelerator.backward(loss)` and `fabric.backward(loss)` invert the usual shape:
        # the tensor is the *argument*, and the receiver is a framework object. Seeding the
        # receiver made `accelerator` itself read as a live tensor, so every later
        # `accelerator.gather_for_metrics(...)` looked grad-bearing -- which is how a
        # detached evaluation loop still produced a TG001 error.
        leaf = name.rsplit(".", 1)[-1].lower()
        if any(hint in leaf for hint in BACKWARD_WRAPPER_RECEIVERS):
            args = [a for a in node.args if a.keyword is None]
            if args:
                argument = dotted_name(args[0].value)
                if argument:
                    self.seeds.add((self.scope_path, argument))
            return True

        self.seeds.add((self.scope_path, name))
        return True

    # -- ``x.requires_grad = True`` --------------------------------------------

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool:
        target = node.target
        if isinstance(target, cst.Attribute) and target.attr.value == "requires_grad":
            name = dotted_name(target.value)
            # Only ``x.requires_grad = True`` seeds. ``x.requires_grad = inp.requires_grad``
            # copies a flag whose value we do not know, and marking it grad-bearing flags
            # perfectly safe code (torch.utils.checkpoint does exactly this).
            if name and isinstance(self._pending_value, cst.Name):
                if self._pending_value.value == "True":
                    self.seeds.add((self.scope_path, name))
        return True


def _reads_detached_name(
    prov: Provenance,
    value: cst.BaseExpression,
    scope: ScopePath,
    loops: Tuple[int, ...],
) -> bool:
    """Does ``value`` read a name already known to hold no graph inside ``loops``?"""

    class _Probe(cst.CSTVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_Name(self, node: cst.Name) -> bool:
            if prov.detached_here(node.value, scope, loops):
                self.found = True
            return True

    probe = _Probe()
    value.visit(probe)
    return probe.found


def _contains_detaching_call(tree: cst.CSTNode) -> bool:
    """Does this expression visibly sever the graph anywhere inside it?

    Positive evidence, deliberately. `#46` established that "we could not prove it carries a
    graph" must never be read as "it does not" -- that silenced a real leak even when
    ``loss.backward()`` was called on the name. So a callee's return only counts as detached
    when a ``.item()`` / ``.detach()`` / ``float(...)`` is actually visible in it.
    """

    class _Probe(cst.CSTVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_Call(self, node: cst.Call) -> bool:
            func = node.func
            if isinstance(func, cst.Attribute) and func.attr.value in DETACHING_METHODS:
                self.found = True
            elif isinstance(func, cst.Name) and func.value in DETACHING_BUILTINS:
                self.found = True
            return True

    probe = _Probe()
    tree.visit(probe)
    return probe.found


def _resolved_returns(
    collector: "_Collector", assignment: _Assignment
) -> List[Tuple[str, ScopePath, cst.BaseExpression]]:
    """What a local helper hands back, matched to the names it is assigned to.

    ``output, loss = train(...)`` against ``return output, loss.item() / n`` maps ``loss`` to
    the second element. The flattened ``targets`` list cannot express that, which is why the
    positional names are kept separately.

    Only bare-name calls to functions defined in this file are resolved; anything imported,
    dotted or shadowed is left alone.
    """
    value = assignment.value
    if not isinstance(value, cst.Call) or not isinstance(value.func, cst.Name):
        return []
    returns = collector.returns.get(value.func.value)
    if not returns or len(returns) != 1:
        return []  # one unambiguous return only; branching returns are not worth guessing at

    scope, expression = returns[0]
    if assignment.unpacked and isinstance(expression, cst.Tuple):
        elements = [e.value for e in expression.elements]
        if len(elements) != len(assignment.unpacked):
            return []
        return [
            (name, scope, element)
            for name, element in zip(assignment.unpacked, elements)
        ]
    if len(assignment.targets) == 1 and not isinstance(expression, cst.Tuple):
        return [(assignment.targets[0], scope, expression)]
    return []


def analyze(module: cst.Module) -> Provenance:
    """Run the two-phase provenance analysis over a parsed module."""
    collector = _Collector()
    module.visit(collector)

    prov = collector.prov
    prov.grad = set(collector.seeds)

    # Name-hint seeding: a variable literally called ``loss`` that was assigned from a
    # call is a live loss tensor unless the call visibly detached it. This is the single
    # highest-value heuristic in the analysis, so it runs before the fixpoint.
    for assignment in collector.assignments:
        if not isinstance(assignment.value, cst.Call) or assignment.augmented:
            continue
        if prov.is_explicitly_detached(assignment.value) or assignment.no_grad:
            continue
        for name in assignment.targets:
            if name.rsplit(".", 1)[-1] in LOSS_NAME_HINTS:
                prov.grad.add((assignment.scope, name))

    # A local helper that visibly detaches what it returns settles what its caller holds.
    # `char_rnn_generation_tutorial.py` has `train()` return `loss.item() / n`, so the
    # caller's `total_loss += loss` accumulates a float -- but the caller's variable is
    # named `loss`, and the name hint is the strongest heuristic here, so without reading
    # the callee it wins and reports a retained graph.
    #
    # This has to run *before* the fixpoint. Discarding afterwards is too late: the fixpoint
    # will already have carried the name into everything derived from it, and removing the
    # source does not retract what it fed.
    visibly_detached: Set[Tuple[ScopePath, str]] = set()
    for assignment in collector.assignments:
        for name, callee_scope, expression in _resolved_returns(collector, assignment):
            if not _contains_detaching_call(expression):
                continue
            if prov.is_grad_bearing(expression, callee_scope):
                continue
            visibly_detached.add((assignment.scope, name))
    prov.grad -= visibly_detached

    # Fixpoint over assignment edges.
    for _ in range(12):
        changed = False
        for assignment in collector.assignments:
            # Autograd was off when this ran, so the value has no graph to propagate.
            if assignment.no_grad:
                continue
            if not prov.is_grad_bearing(
                assignment.value, assignment.scope, assignment.loops
            ):
                # Propagate the *detachment* one hop, so `loss = outputs.loss` inherits it
                # from an `outputs` that a no_grad forward produced. This requires positive
                # evidence -- the value must read a name already known detached in these
                # loops. "We could not prove it carries a graph" is not evidence: treating
                # absence of proof as detachment silenced `loss = compute_loss(...)` and,
                # worse, silenced it even when `loss.backward()` was called on it.
                if assignment.loops and _reads_detached_name(
                    prov, assignment.value, assignment.scope, assignment.loops
                ):
                    for name in assignment.targets:
                        key = (assignment.scope, name)
                        known = prov.not_grad_in_loops.setdefault(key, set())
                        if not known.issuperset(assignment.loops):
                            known.update(assignment.loops)
                            changed = True
                continue
            for name in assignment.targets:
                key = (assignment.scope, name)
                if key in visibly_detached:
                    continue  # the callee's own return says otherwise
                if key not in prov.grad:
                    prov.grad.add(key)
                    changed = True
        if not changed:
            break

    # A detached leaf has no graph behind it, so any seed we picked up for such a name
    # (e.g. a later ``requires_grad = True``) is wrong.
    for assignment in collector.assignments:
        if not prov.is_explicitly_detached(assignment.value):
            continue
        for name in assignment.targets:
            prov.grad.discard((assignment.scope, name))

    # An explicitly detached assignment wins: drop names whose every binding detaches.
    for assignment in collector.assignments:
        if not prov.is_explicitly_detached(assignment.value):
            continue
        for name in assignment.targets:
            if name.rsplit(".", 1)[-1] in LOSS_NAME_HINTS:
                prov.grad.discard((assignment.scope, name))

    return prov
