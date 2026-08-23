"""TG001 - storing a graph-attached tensor in a container that outlives the iteration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import libcst as cst

from ..analysis.helpers import attach_method, dotted_name, final_attr
from ..analysis.scope import ScopePath, target_names
from ..diagnostics import Category, Severity
from .base import Rule, register

ACCUMULATING_METHODS = {"append", "add", "extend", "insert", "put", "push"}

#: Containers that hold checkpoint or configuration state rather than per-iteration values.
#: `state_dict[f"{name}.lora_A.weight"] = lora_A` is how you *assemble* a checkpoint, and
#: `model_args[k] = checkpoint_model_args[k]` moves config integers between dicts -- neither
#: runs per training step. Both were reported as retained graphs, in `peft` and in nanoGPT.
STATE_CONTAINER_HINTS = (
    "state_dict", "statedict", "checkpoint", "ckpt", "config", "args", "pattern",
    "metadata", "kwargs", "params_dict",
)

#: Factories that allocate a tensor buffer. Writing into one is not accumulation: the buffer
#: has a fixed size, and filling it chunk by chunk is the memory-efficient idiom -- axolotl
#: builds `loss_per_sample[i] = ...` exactly this way, then reduces and backwards it.
TENSOR_FACTORIES = {
    "zeros", "empty", "ones", "full", "zeros_like", "empty_like", "ones_like",
    "full_like", "new_zeros", "new_empty", "new_full",
}

#: A name, keyed to the scope that owns it. ``self.losses`` is instance state and keys to
#: ``None`` so it matches file-wide; a bare local keys to its enclosing function.
Key = Tuple[Optional[ScopePath], str]


@dataclass
class _Candidate:
    """A finding held back until the whole module has been read."""

    node: cst.CSTNode
    holder: Key
    #: Names read in the stored expression, for deciding whether its graph was freed.
    stored: Set[Key]
    subject: str
    hint: str


@register
class GraphRetention(Rule):
    code = "TG001"
    name = "graph-retention"
    summary = "Tensor stored with its autograd graph still attached"
    severity = Severity.ERROR
    category = Category.CRITICAL_OOM
    explanation = """
Storing a tensor that still carries ``grad_fn`` keeps its computational graph reachable. How
much that costs depends on one thing: whether the graph has already been backwarded.

**Never backwarded** — the graph still holds every intermediate activation produced on the way
to that tensor. A validation loop appending outputs, or a container built up for a single
backward later, retains one full graph per iteration. Memory grows linearly and the run dies
with CUDA OOM partway through, usually after hours of GPU time. Reported as an error.

**Already backwarded** — ``backward()`` releases the saved tensors as it traverses, so a
tensor stored *after* its backward retains the graph nodes but none of the activations.
Measured on a 13x256 MLP (``tests/calibration/measure_retention.py``): **560 KiB of
activations per iteration** still reachable when nothing backwards them, and **0 KiB** when
something does — with the same ~30 graph nodes per iteration held either way. Those nodes are
host-side bookkeeping, not VRAM. The retention is real and still grows linearly, but it will
not OOM the GPU, so it is reported as a warning rather than an error.

Either way the fix is the same. Call ``.item()`` for scalars you only want to log, or
``.detach()`` for tensors you need to keep as tensors.

**Retention a later backward depends on is not a leak, and this rule does not flag it.** Some
code stores graph-attached tensors precisely so a backward pass can run over them, and
detaching would break it rather than save memory::

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        loss = self._compute_loss(output, target_mbs[mb_index])
        self._internal_losses.append(loss)      # required, not a leak

    def _maybe_get_loss(self, stage, mb_index):
        return self._internal_losses[mb_index]  # the graph is still needed here

That is how pipeline parallelism schedules microbatches. The same shape appears in chunked
loss modules that accumulate per-chunk losses and return the total, in RL objectives that
stack per-step losses and backward the sum, and in ``torch.distributed.autograd``.

So a holder is exempt when its value reaches a backward pass, or is returned to a caller we
cannot follow. That is a reachability question, not a syntactic one:
``torch.stack(losses).mean()`` is a throwaway reduction when it is logged and load-bearing
when it becomes the training loss, and the two are written identically.
""".strip()

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._candidates: List[_Candidate] = []
        #: ``target -> names read in whatever was assigned to it``. Reachability runs
        #: backwards along these edges: if a value is needed, so is everything it came from.
        self._flows_from: Dict[Key, Set[Key]] = {}
        #: Names bound to a tensor buffer from a factory, so a subscript write into one is
        #: not read as a container growing.
        self._tensor_buffers: Set[Key] = set()
        self._backward_seeds: Set[Key] = set()
        self._return_seeds: Set[Key] = set()
        #: Bare names handed back, resolved at verdict time. A bare name is evidence only
        #: when it is not itself a holder -- see :meth:`_return_keys`.
        self._returned_bare: Set[Key] = set()

    # ------------------------------------------------------------------ collection

    def visit_Call(self, node: cst.Call) -> bool:
        if self._in_triton_kernel():
            return True
        # A backward pass reads whatever it is handed. Record that before considering any
        # candidate, since one call can be both -- `losses[i].backward()`.
        if self._is_backward_call(node):
            for arg in node.args:
                self._backward_seeds |= self._read_keys(arg.value)
            if isinstance(node.func, cst.Attribute):
                self._backward_seeds |= self._read_keys(node.func.value)
            return True

        func = node.func
        if not isinstance(func, cst.Attribute):
            return True
        method = func.attr.value
        if method not in ACCUMULATING_METHODS:
            return True

        container = dotted_name(func.value)
        if container is None or not self._container_accumulates(container):
            return True

        # ``insert(index, value)`` stores its second argument.
        args = [a for a in node.args if a.keyword is None]
        if not args:
            return True
        stored = args[1].value if method == "insert" and len(args) > 1 else args[0].value

        if self.in_no_grad or not self.is_grad(stored):
            return True

        self._defer(
            stored,
            container,
            f"`{container}.{method}(...)`",
            "Use `.item()` to keep just the scalar value, or `.detach()` to keep the "
            "tensor without its graph.",
        )
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        if self._in_triton_kernel():
            return True
        target = dotted_name(node.target)
        if target is not None:
            self._note_flow(target, node.value)

        # ``total_loss += loss`` accumulates graphs just as surely as ``.append``.
        if not self.in_loop or self.in_no_grad:
            return True
        if not isinstance(node.operator, (cst.AddAssign, cst.SubtractAssign)):
            return True
        if target is None or self.bound_in_innermost_loop(target):
            return True
        if not self.is_grad(node.value):
            return True

        self._defer(
            node.value,
            target,
            f"`{target} += ...`",
            "Accumulate the scalar instead: `{0} += loss.item()`.".format(target),
        )
        return True

    def visit_Assign(self, node: cst.Assign) -> bool:
        if self._in_triton_kernel():
            return True
        for target in node.targets:
            # ``x = expr`` and ``cache[k] = expr`` both mean the graph of ``expr`` is now
            # reachable through the target, which is what reachability needs to follow.
            holder = target.target
            if isinstance(holder, cst.Subscript):
                holder = holder.value
            name = dotted_name(holder)
            if name is not None:
                self._note_flow(name, node.value)

        # A tensor buffer is not a container. Record it before the candidate check below.
        if isinstance(node.value, cst.Call) and final_attr(node.value.func) in TENSOR_FACTORIES:
            for target in node.targets:
                name = dotted_name(target.target)
                if name is not None:
                    self._tensor_buffers.add(self._key(name))

        if self.in_no_grad or not self.is_grad(node.value):
            return True

        for target in node.targets:
            if not isinstance(target.target, cst.Subscript):
                continue
            container = dotted_name(target.target.value)
            if container is None or not self._container_accumulates(container):
                continue
            if self._is_state_container(container, target.target):
                continue
            if self._key(container) in self._tensor_buffers:
                continue
            self._defer(
                node.value,
                container,
                f"`{container}[...] = ...`",
                "Store `.item()` or `.detach()` instead.",
            )
            break
        return True

    def visit_For(self, node: cst.For) -> bool:
        # A loop variable is an element of what it iterates, so `for l in self.losses:
        # l.backward()` has to reach the container and not the throwaway name.
        for name in target_names(node.target):
            self._flows_from.setdefault(self._key(name), set()).update(
                self._read_keys(node.iter)
            )
        return True

    def visit_Return(self, node: cst.Return) -> bool:
        if node.value is not None:
            self._return_seeds |= self._return_keys(node.value)
        return True

    def _return_keys(self, expr: cst.BaseExpression) -> Set[Key]:
        """Names whose *graph* the caller is being handed.

        A bare name says nothing. ``return losses`` hands back a container, and a caller is
        as likely to read values out of it as to backward through it — so that on its own
        must not excuse the retention, or the rule would fall silent on every helper that
        collects and returns.

        A value *computed* from the accumulation is different: ``return total_loss / n`` and
        ``return self._internal_losses[i]`` are only meaningful with the graph attached, so
        holding it is the point rather than an oversight.
        """
        if isinstance(expr, (cst.Tuple, cst.List)):
            keys: Set[Key] = set()
            for element in expr.elements:
                keys |= self._return_keys(element.value)
            return keys
        if isinstance(expr, (cst.Name, cst.Attribute)) and dotted_name(expr) is not None:
            # Held back rather than discarded. Returning the *container* says nothing, but
            # returning something *derived* from it hands over the graph -- and both are
            # written as a bare name. `trl`'s GRPO trainer does
            # `logps = torch.cat(all_logps, dim=0)` then `return logps, entropies, aux_loss`,
            # where `logps` is not the container. Which case this is cannot be known until
            # every holder has been seen, so the decision moves to `leave_Module`.
            self._returned_bare |= self._read_keys(expr)
            return set()
        return self._read_keys(expr)

    # ------------------------------------------------------------------ verdict

    def leave_Module(self, original_node: cst.Module) -> None:
        # A returned bare name counts as evidence unless it is a holder itself. `return
        # losses` must not excuse `losses`, but `return logps` may excuse the container
        # `logps` was concatenated from.
        holders = {candidate.holder for candidate in self._candidates}
        derived_returns = self._returned_bare - holders
        load_bearing = self._reachable(
            self._backward_seeds | self._return_seeds | derived_returns
        )
        already_backwarded = self._reachable(self._backward_seeds)

        for candidate in self._candidates:
            if candidate.holder in load_bearing:
                continue

            # `backward()` frees the saved tensors as it traverses, so a tensor stored after
            # its own backward retains graph nodes and no activations. Same fix, much smaller
            # consequence, and claiming OOM for it would be false.
            freed = bool(candidate.stored & already_backwarded)
            if freed:
                message = (
                    f"{candidate.subject} keeps a graph-attached tensor whose backward has "
                    f"already run. Its activations are freed, so what accumulates is the "
                    f"graph nodes — host memory rather than VRAM, but still growing every "
                    f"iteration, and the stored value stays silently differentiable."
                )
            else:
                message = (
                    f"{candidate.subject} keeps a tensor attached to the autograd graph and "
                    f"nothing backwards it, so every activation on the way to it stays in "
                    f"VRAM — one full graph per iteration, until the run OOMs."
                )

            self.report(
                candidate.node,
                message,
                hint=candidate.hint,
                severity=Severity.WARNING if freed else Severity.ERROR,
                category=Category.PERFORMANCE_WARN if freed else Category.CRITICAL_OOM,
                fix_build=lambda updated: attach_method(updated, "detach"),
                fix_description="add .detach()",
            )

    # ------------------------------------------------------------------ helpers

    def _defer(self, node: cst.BaseExpression, holder: str, subject: str, hint: str) -> None:
        """Hold a finding until ``leave_Module``.

        Both the exemption and the severity depend on code that may appear anywhere in the
        file — a getter defined below the append, a backward in a sibling method — so
        neither verdict can be reached at the point of the write.
        """
        self._candidates.append(
            _Candidate(
                node=node,
                holder=self._key(holder),
                stored=self._read_keys(node),
                subject=subject,
                hint=hint,
            )
        )

    def _key(self, name: str) -> Key:
        """Scope a name so file-level facts cannot leak between functions.

        ``self.losses`` is instance state: appended in one method and backwarded in another
        is the normal shape, so it matches file-wide. A bare local matches only within its
        own function — the leakage that has already bitten ``models``, ``criteria``,
        ``uses_distributed`` and TG008.
        """
        if name.startswith("self."):
            return (None, name)
        return (self._function_key(), name)

    def _function_key(self) -> ScopePath:
        """Scope path down to the innermost enclosing function."""
        last = 0
        for index, scope in enumerate(self.scopes):
            if scope.kind == "function":
                last = index
        return tuple(s.name for s in self.scopes[: last + 1])

    def _note_flow(self, target: str, value: cst.BaseExpression) -> None:
        self._flows_from.setdefault(self._key(target), set()).update(self._read_keys(value))

    def _read_keys(self, node: cst.BaseExpression) -> Set[Key]:
        return {self._key(name) for name in _reads(node)}

    def _reachable(self, seeds: Set[Key]) -> Set[Key]:
        """Everything the seeds' values were built from, transitively.

        Flow-insensitive and within-scope, which is all the surrounding analysis claims.
        ``loss = stack(losses).sum(); loss.backward()`` reaches ``losses`` in two hops.
        """
        seen = set(seeds)
        stack = list(seeds)
        while stack:
            for source in self._flows_from.get(stack.pop(), ()):
                if source not in seen:
                    seen.add(source)
                    stack.append(source)
        return seen

    def _is_backward_call(self, node: cst.Call) -> bool:
        """A call that runs a backward pass over whatever it is given.

        Matched on the callee's last segment containing ``backward``, which covers
        ``loss.backward()``, ``torch.autograd.backward(...)``,
        ``dist_autograd.backward(ctx, [...])`` and pipelining's ``backward_one_chunk(...)``.
        """
        leaf = final_attr(node.func)
        return leaf is not None and "backward" in leaf

    def _is_state_container(self, container: str, subscript: cst.Subscript) -> bool:
        """Checkpoint or config plumbing rather than a per-iteration accumulator.

        Two signals, either sufficient: the name says so, or the key is a string. A slot
        addressed by a parameter name (`state_dict[f"{name}.lora_A.weight"]`) is a mapping
        being assembled; a per-step accumulator is addressed by an index.
        """
        leaf = container.rsplit(".", 1)[-1].lower()
        if any(hint in leaf for hint in STATE_CONTAINER_HINTS):
            return True
        for element in subscript.slice:
            index = getattr(element.slice, "value", None)
            if isinstance(index, (cst.SimpleString, cst.FormattedString)):
                return True
        return False

    def _in_triton_kernel(self) -> bool:
        """Triton kernels contain no autograd at all.

        `@triton.jit` functions are compiled to GPU code; `dv += tl.dot(...)` in a
        hand-written flash-attention backward is arithmetic, not graph accumulation.
        axolotl's `flash_attn_d512.py` produced three errors this way, in a file that
        never touches `torch.autograd`.
        """
        for scope in self.scopes:
            node = scope.node
            if not isinstance(node, cst.FunctionDef):
                continue
            for decorator in node.decorators:
                if "triton" in (dotted_name(decorator.decorator) or ""):
                    return True
                call = decorator.decorator
                if isinstance(call, cst.Call) and "triton" in (dotted_name(call.func) or ""):
                    return True
        return False

    def _container_accumulates(self, container: str) -> bool:
        """True if writes to ``container`` survive past the current loop iteration."""
        if self.in_loop:
            # A list built fresh each iteration cannot accumulate across iterations.
            return not self.bound_in_innermost_loop(container)
        # Outside a loop, only instance state persists across repeated calls
        # (``self.outputs.append(...)`` in a step method is the classic Lightning leak).
        return container.startswith("self.")


def _reads(node: cst.CSTNode) -> List[str]:
    """Dotted names read within an expression.

    A subscript contributes its base, so ``losses[i]`` reads ``losses`` — indexing a held
    container is how a deferred backward gets at what it needs.
    """
    collector = _ReadCollector()
    node.visit(collector)
    return collector.names


class _ReadCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: List[str] = []

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        name = dotted_name(node.value)
        if name is not None:
            self.names.append(name)
        return True

    def visit_Name(self, node: cst.Name) -> bool:
        self.names.append(node.value)
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        name = dotted_name(node)
        if name is None:
            return True  # ``f().x`` -- descend, the reads are inside the call
        self.names.append(name)
        # A pure name path holds no further reads. Descending would collect its own
        # segments, so ``self.losses`` would also mark a local ``losses`` elsewhere.
        return False
