"""Rule-by-rule behaviour: what must fire, and what must stay quiet."""

from conftest import analyze, codes

from torch_preflight.diagnostics import Severity


# --------------------------------------------------------------------- TG001


def test_tg001_flags_appending_attached_loss():
    assert "TG001" in codes(
        """
        def train(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
        """
    )


def test_tg001_flags_augmented_accumulation():
    """Accumulating after backward is a warning: the activations are already freed.

    Measured by `tests/calibration/measure_retention.py`: on a 13x256 MLP, 560 KiB of
    activations per iteration stay reachable when nothing backwards them and 0 KiB when
    something does. Same fix either way, but only the first is the CUDA OOM the rule used to
    claim for both.
    """
    diagnostics = analyze(
        """
        def train(model, loader, criterion, optimizer):
            total = 0.0
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                total += loss
        """
    )
    assert [d.code for d in diagnostics] == ["TG001"]
    assert diagnostics[0].severity is Severity.WARNING
    assert "whose backward has already run" in diagnostics[0].message


def test_tg001_errors_when_nothing_backwards_the_stored_tensor():
    """No backward, so the activations really are retained and the run really will OOM."""
    diagnostics = analyze(
        """
        def train(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                losses.append(loss)
        """
    )
    assert [d.code for d in diagnostics] == ["TG001"]
    assert diagnostics[0].severity is Severity.ERROR
    assert "stays in VRAM" in diagnostics[0].message


def test_tg001_quiet_when_a_derived_bare_name_is_returned():
    """Reduced from `trl/trainer/grpo_trainer.py` and torchtune's `linear_grpo_loss.py`.

    `return logps, ...` is a bare name, which the rule above deliberately treats as no
    evidence — but `logps` is not the container, it is `torch.cat(all_logps)`. Returning it
    hands the caller a graph that the GRPO objective then backwards.

    So a returned bare name counts unless it is *itself* a holder. That keeps `return losses`
    from excusing `losses` while letting a reduction of it through, and the two cannot be
    told apart until every holder has been seen.
    """
    assert codes(
        """
        import torch

        def _get_per_token_logps(self, model, input_ids, batches):
            all_logps = []
            for input_ids_batch in batches:
                logits = model(input_ids_batch).logits
                all_logps.append(selective_log_softmax(logits, input_ids_batch))
            logps = torch.cat(all_logps, dim=0)
            return logps
        """
    ) == []


def test_tg001_returning_a_bare_container_is_not_evidence_of_a_deferred_backward():
    """`return losses` must not excuse the retention.

    A caller is as likely to read values out of a returned container as to backward through
    it, and treating the bare return as proof would silence the rule on every helper that
    collects and hands back. A value *computed* from the accumulation is different — that is
    the chunked-loss shape, covered below.
    """
    assert "TG001" in codes(
        """
        def collect(model, loader, criterion):
            losses = []
            for batch, y in loader:
                losses.append(criterion(model(batch), y))
            return losses
        """
    )


def test_tg001_quiet_when_an_accumulated_total_is_returned_for_backward():
    """Reduced from `torchtune/modules/loss/kd_losses.py` and `cross_entropy_loss.py`.

    A chunked loss module accumulates per-chunk losses and returns the total for the caller
    to backward. Detaching would make the loss non-differentiable and the model would not
    train at all.
    """
    assert codes(
        """
        def forward(self, student_logits, teacher_logits, labels, mask):
            total_loss = 0.0
            for student_chunk, label_chunk in zip(student_logits, labels):
                total_loss += self.loss_fn(student_chunk, label_chunk)
            return total_loss / torch.sum(mask.view(-1), dim=0)
        """
    ) == []


def test_tg001_quiet_when_a_stacked_container_becomes_the_training_loss():
    """Reduced from `examples/reinforcement_learning/actor_critic.py`.

    `torch.stack(container).sum()` is a throwaway reduction when it is logged and
    load-bearing when it becomes the loss. The two are written identically, so only
    reachability separates them — see the logging case above, which still fires.
    """
    assert codes(
        """
        def finish_episode(model, optimizer, saved_actions, returns):
            policy_losses = []
            value_losses = []
            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            loss.backward()
            optimizer.step()
        """
    ) == []


def test_tg001_quiet_for_checkpoint_and_config_plumbing():
    """Reduced from `karpathy/nanoGPT` and `peft/src/peft/tuners/lora/conversion.py`.

    Assembling a `state_dict` or moving config integers between dicts is startup plumbing,
    not a training loop. We reported `CRITICAL_OOM` on nanoGPT — the most widely read minimal
    training script there is — for renaming checkpoint keys.

    Two signals, either sufficient: the container's name says it holds state, or the key is a
    string. A slot addressed by a parameter name is a mapping being assembled; a per-step
    accumulator is addressed by an index.
    """
    assert "TG001" not in codes(
        """
        import torch

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model"]
        for k, v in list(state_dict.items()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
        loss = criterion(model(x), y)
        loss.backward()
        """
    )
    assert "TG001" not in codes(
        """
        def convert(model, lora_A, lora_B, state_dict, names):
            for name in names:
                state_dict[f"{name}.lora_A.weight"] = lora_A
                state_dict[f"{name}.lora_B.weight"] = lora_B
        """
    )


def test_tg001_quiet_when_filling_a_preallocated_tensor_buffer():
    """Reduced from `axolotl/src/axolotl/integrations/diffusion/trainer.py`.

    A tensor buffer is not a container. It has a fixed size, and filling it chunk by chunk
    is the memory-efficient idiom — this one *becomes* the loss and is backwarded.
    """
    assert "TG001" not in codes(
        """
        import torch

        def loss_fn(weighted_loss, answer_lengths, masks, n):
            loss_per_sample = torch.zeros(n, device=weighted_loss.device)
            for i in range(n):
                sample_loss = weighted_loss[masks[i]].sum()
                loss_per_sample[i] = sample_loss / answer_lengths[i].clamp(min=1.0)
            return loss_per_sample.mean()
        """
    )


def test_tg001_quiet_inside_a_triton_kernel():
    """Reduced from `axolotl/src/axolotl/monkeypatch/attention/flash_attn_d512.py`.

    `@triton.jit` functions compile to GPU code and contain no autograd at all — `dv +=
    tl.dot(...)` in a hand-written flash-attention backward is arithmetic. Three errors in a
    file that never touches `torch.autograd`.
    """
    assert "TG001" not in codes(
        """
        import triton
        import triton.language as tl

        @triton.jit
        def _bwd_dkdv(Q, K, V, DO, sm_scale, BLOCK_M: tl.constexpr):
            dv = tl.zeros([16, 16], dtype=tl.float32)
            dk = tl.zeros([16, 16], dtype=tl.float32)
            for start_n in range(0, 128, BLOCK_M):
                p = tl.load(Q + start_n)
                do = tl.load(DO + start_n)
                dv += tl.dot(tl.trans(p).to(do.dtype), do)
                dk += tl.dot(tl.trans(p), do)
            tl.store(K + start_n, dk)
        """
    )


def test_tg001_flags_dict_and_self_containers():
    assert "TG001" in codes(
        """
        def train(model, loader, criterion, optimizer, cache):
            for i, (batch, y) in enumerate(loader):
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                cache[i] = loss
        """
    )


def test_tg001_flags_self_attribute_container_without_a_loop():
    assert "TG001" in codes(
        """
        class Trainer:
            def training_step(self, batch, y):
                loss = self.criterion(self.model(batch), y)
                self.outputs.append(loss)
        """
    )


def test_tg001_quiet_when_detached_or_scalarised():
    for stored in ("loss.item()", "loss.detach()", "float(loss)", "loss.detach().cpu()"):
        assert codes(
            f"""
            def train(model, loader, criterion, optimizer):
                losses = []
                for batch, y in loader:
                    loss = criterion(model(batch), y)
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append({stored})
            """
        ) == [], stored


def test_tg001_quiet_for_container_rebuilt_each_iteration():
    assert codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                losses = []
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
        """
    ) == []


def test_tg001_quiet_inside_no_grad():
    assert codes(
        """
        import torch

        def collect(model, loader, criterion):
            outputs = []
            with torch.no_grad():
                for batch, y in loader:
                    outputs.append(model(batch))
        """
    ) == []


def test_tg001_quiet_when_the_value_came_from_a_no_grad_forward():
    """`with torch.no_grad():` around only the forward, which is the standard eval loop.

    The append is outside the block, so a positional "am I inside no_grad" check misses it —
    but the tensor has no `grad_fn` at all, because autograd was off when it was produced.
    Reduced from `transformers/examples/pytorch/language-modeling/run_clm_no_trainer.py`.
    """
    assert codes(
        """
        import torch

        def evaluate(model, eval_dataloader, accelerator, args):
            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.n)))
        """
    ) == []


def test_tg001_reads_what_a_local_helper_actually_returns():
    """Reduced from `pytorch/tutorials/.../char_rnn_generation_tutorial.py`.

    `train()` calls `.item()` internally and hands back a float, so the caller's
    `total_loss += loss` accumulates a number. We flagged it because the caller's variable is
    named `loss` and the name hint is the strongest heuristic in the analysis — it wins
    unless something reads the callee, whose `return` is a few lines away in the same file.

    Note the tuple: `output, loss = train(...)` has to be matched element-wise against
    `return output, loss.item() / n`, which the flattened target list cannot express.
    """
    assert codes(
        """
        def train(rnn, criterion, category_tensor, input_line_tensor, target_line_tensor):
            loss = 0
            hidden = rnn.initHidden()
            for i in range(input_line_tensor.size(0)):
                output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
                loss += criterion(output, target_line_tensor[i])
            loss.backward()
            return output, loss.item() / input_line_tensor.size(0)

        total_loss = 0
        all_losses = []
        for iter in range(1, 100):
            output, loss = train(rnn, criterion, *randomTrainingExample())
            total_loss += loss
            if iter % 10 == 0:
                all_losses.append(total_loss / 10)
                total_loss = 0
        """
    ) == []


def test_tg001_still_fires_when_the_helper_returns_a_live_tensor():
    """The guard for the above: resolving the callee must not become a blanket excuse.

    The helper hands back `criterion(model(batch), y)` with nothing detaching it, so the
    caller really is retaining a graph. Only a *visible* detach in the return counts —
    per #46, "we could not prove it carries a graph" is not evidence that it does not.
    """
    assert "TG001" in codes(
        """
        def compute(model, batch, y, criterion):
            return criterion(model(batch), y)

        def run(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = compute(model, batch, y, criterion)
                optimizer.zero_grad()
                losses.append(loss)
            return losses
        """
    )


def test_tg001_absence_of_proof_is_not_evidence_of_detachment():
    """Guards a regression the rest of the suite could not see.

    Loop-scoped detachment first marked a name detached whenever its binding was not
    *provably* grad-bearing in that loop. That silenced both cases below — the second even
    though `loss.backward()` is definitive proof the tensor carries a graph.

    The suite missed it because every other TG001 fixture assigns from something we can
    resolve (`criterion(model(batch), y)`), and a seven-repo scan could not see it either:
    a true positive that stops firing is indistinguishable from a false positive that got
    fixed. Detachment now needs positive evidence, and these two exist so it stays that way.
    """
    unresolved = """
        def train(model, loader, optimizer):
            losses = []
            for batch, y in loader:
                loss = compute_loss(model, batch, y)
                optimizer.zero_grad()
                losses.append(loss)
    """
    assert "TG001" in codes(unresolved)

    backwarded = """
        def train(model, loader, optimizer):
            losses = []
            for batch, y in loader:
                loss = compute_loss(model, batch, y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
    """
    assert "TG001" in codes(backwarded)


def test_tg001_quiet_on_the_standard_accelerate_evaluation_loop():
    """The whole shape, reduced from `transformers/examples/pytorch/.../run_clm_no_trainer.py`.

    One function trains and then evaluates, binding `outputs` and `loss` in both loops. Two
    separate things had to be right for this to go quiet:

    * `no_grad` detachment is scoped to the loop it happened in, so the evaluation loop's
      `outputs` is not rescued into grad-ness by the training loop that binds the same name
      a few lines above. Python has function scope, not block scope, so this cannot simply
      shadow.
    * `accelerator.backward(loss)` seeds its *argument*, not its receiver. Seeding the
      receiver made `accelerator` itself read as a live tensor, so every later
      `accelerator.gather_for_metrics(...)` looked grad-bearing regardless.
    """
    assert codes(
        """
        import torch

        def main(model, train_dataloader, eval_dataloader, optimizer, accelerator, args):
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.n)))
        """
    ) == []


def test_tg001_accelerator_backward_still_marks_its_argument():
    """The mirror: the fix must not stop `accelerator.backward(loss)` marking `loss`."""
    diagnostics = analyze(
        """
        def train(model, loader, criterion, optimizer, accelerator):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss)
        """
    )
    assert [d.code for d in diagnostics] == ["TG001"]


def test_tg001_quiet_when_a_getter_hands_the_element_to_a_deferred_backward():
    """Reduced from ``torch/distributed/pipelining/schedules.py``, where we fired wrongly.

    Pipeline parallelism computes the loss for a microbatch, holds it, and backwards it when
    the schedule reaches that microbatch. The graph *must* survive the append, so the
    finding was not merely noisy — following the hint would have broken the schedule.
    """
    assert codes(
        """
        class PipelineSchedule:
            def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
                if stage.is_last and self._loss_fn is not None:
                    loss = self._compute_loss(output, target_mbs[mb_index])
                    self._internal_losses.append(loss)

            def _maybe_get_loss(self, stage, mb_index):
                return self._internal_losses[mb_index]
        """
    ) == []


def test_tg001_quiet_when_a_held_chain_is_passed_to_a_backward_call():
    """Reduced from ``torch/testing/_internal/distributed/rpc/dist_autograd_test.py``.

    The dict is a chain of intermediate tensors built precisely so the distributed
    backward can traverse it. The read reaches the backward through a list, a call and a
    subscript, which is why the exemption cannot just look at the call's direct arguments.
    """
    # Asserted against TG001 alone: the snippet also draws unseeded, which TG008 flags.
    assert "TG001" not in codes(
        """
        import torch
        from torch.distributed import autograd as dist_autograd

        def test_debug_info(self, context_id):
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            i = 0
            res = {}
            res[i] = t1
            for rank in range(self.world_size):
                res[i + 1] = torch.add(res[i], t2)
                i += 1
            dist_autograd.backward(context_id, [res[i].sum()])
        """
    )


def test_tg001_quiet_when_the_accumulated_sum_is_backwarded():
    """Summing losses and backwarding once is correct code, not a leak."""
    assert codes(
        """
        def train(model, loader, criterion, optimizer):
            total = 0.0
            for batch, y in loader:
                total += criterion(model(batch), y)
            optimizer.zero_grad()
            total.backward()
        """
    ) == []


def test_tg001_quiet_when_held_losses_are_backwarded_in_a_loop():
    assert codes(
        """
        class Trainer:
            def step(self, batch, y):
                self.losses.append(self.criterion(self.model(batch), y))

            def flush(self):
                for loss in self.losses:
                    loss.backward()
        """
    ) == []


def test_tg001_still_fires_when_the_container_is_only_reduced_for_logging():
    """The exemption needs a *backward*, not any read.

    ``torch.stack(self.outputs).mean()`` is the classic Lightning epoch-end reduction, and
    it needs no graph — so the retention is still a leak and the hint still applies.
    """
    diagnostics = analyze(
        """
        import torch

        class LitModel:
            def training_step(self, batch, y):
                loss = self.criterion(self.model(batch), y)
                self.outputs.append(loss)
                return loss

            def on_train_epoch_end(self):
                average = torch.stack(self.outputs).mean()
                self.log("loss", average)
                self.outputs.clear()
        """
    )
    assert [d.code for d in diagnostics] == ["TG001"]


def test_tg001_exemption_does_not_leak_between_functions():
    """A local ``losses`` backwarded in one function must not excuse another's.

    Instance state legitimately crosses methods, so ``self.*`` matches file-wide — but a
    bare local name matches only inside its own function. This is the file-wide fact
    leakage that has already been fixed in `models`, `criteria` and TG008.
    """
    diagnostics = analyze(
        """
        def deferred(model, loader, criterion):
            losses = []
            for batch, y in loader:
                losses.append(criterion(model(batch), y))
            losses[0].backward()

        def leaking(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
        """
    )
    assert [d.code for d in diagnostics] == ["TG001"]


# --------------------------------------------------------------------- TG002


def test_tg002_quiet_when_the_eval_looking_loop_is_the_one_that_backwards():
    """Reduced from `pytorch/tutorials/beginner_source/fgsm_tutorial.py`.

    An adversarial attack iterates `test_loader` and backwards through it deliberately, to
    get gradients with respect to the input. We reported that `test()` "never calls
    `.backward()`" while the call sat nine lines below.

    The carve-out being corrected exists for a function that both trains and validates, where
    a backward elsewhere should not excuse the validation loop. It just has to check that the
    backward is not in *this* loop.
    """
    assert "TG002" not in codes(
        """
        def test(model, device, test_loader, epsilon):
            correct = 0
            for data, target in test_loader:
                data.requires_grad = True
                output = model(data)
                loss = F.nll_loss(output, target)
                model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
            return correct
        """
    )


def test_tg002_flags_eval_function_without_no_grad():
    diagnostics = analyze(
        """
        import torch

        def validate(model, loader):
            for x, y in loader:
                out = model(x)
        """
    )
    assert [d.code for d in diagnostics] == ["TG002"]
    assert diagnostics[0].fixable


def test_tg002_offers_no_fix_when_torch_is_not_imported():
    """We cannot write ``@torch.no_grad()`` into a file that never imported torch."""
    diagnostics = analyze(
        """
        def validate(model, loader):
            for x, y in loader:
                out = model(x)
        """
    )
    assert [d.code for d in diagnostics] == ["TG002"]
    assert not diagnostics[0].fixable


def test_tg002_flags_inline_validation_loop_inside_training():
    diagnostics = analyze(
        """
        def train(model, loader, val_loader, criterion, optimizer):
            for x, y in loader:
                optimizer.zero_grad()
                criterion(model(x), y).backward()
            for x, y in val_loader:
                out = model(x)
        """
    )
    assert [d.code for d in diagnostics] == ["TG002"]
    # Never offer to decorate a training routine with @torch.no_grad().
    assert not diagnostics[0].fixable


def test_tg002_quiet_for_a_preprocessor_loaded_with_from_pretrained():
    """Reduced from `transformers/examples/pytorch/continuous_batching_simple.py`.

    `from_pretrained` loads preprocessors as well as models, and it sits in MODEL_WRAPPERS,
    so it matched before anything could object. That made `tokenizer(x["question"])` look
    like a forward pass, and TG002 reported a missing `no_grad` around *tokenisation*.
    """
    assert "TG002" not in codes(
        """
        from transformers import AutoTokenizer

        def main(dataset):
            tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
            return dataset.map(lambda x: tokenizer(x["question"]), batched=True)
        """
    )


def test_tg002_still_flags_a_real_model_loaded_with_from_pretrained():
    """The guard: excluding preprocessors must not exclude models."""
    assert "TG002" in codes(
        """
        from transformers import AutoModelForCausalLM

        def evaluate(dataset):
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model.eval()
            for batch in dataset:
                outputs = model(**batch)
            return outputs
        """
    )


def test_tg002_quiet_when_guarded():
    for guard in ("@torch.no_grad()", "@torch.inference_mode()", "@torch.no_grad"):
        assert codes(
            f"""
            import torch

            {guard}
            def validate(model, loader):
                for x, y in loader:
                    out = model(x)
            """
        ) == [], guard


def test_tg002_quiet_with_context_manager():
    assert codes(
        """
        import torch

        def validate(model, loader):
            with torch.no_grad():
                for x, y in loader:
                    out = model(x)
        """
    ) == []


def test_tg002_quiet_for_pytest_functions():
    assert codes(
        """
        def test_forward_shape(model, x):
            out = model(x)
            assert out.shape == (2, 10)
        """
    ) == []


def test_tg002_quiet_for_lightning_hooks():
    assert codes(
        """
        import pytorch_lightning as pl

        class Lit(pl.LightningModule):
            def validation_step(self, batch, idx):
                x, y = batch
                return self.model(x)
        """
    ) == []


# --------------------------------------------------------------------- TG003


def test_tg003_flags_backward_without_zero_grad():
    assert "TG003" in codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                loss.backward()
                optimizer.step()
        """
    )


def test_tg003_quiet_when_zero_grad_present():
    assert codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), y)
                loss.backward()
                optimizer.step()
        """
    ) == []


def test_tg003_quiet_for_gradient_accumulation():
    """``zero_grad`` guarded by an ``if`` is deliberate accumulation, not a bug.

    The loss is divided by ``accum`` here so the snippet is *correct* accumulation and the
    assertion stays a whole-file one. Without the division it is a real TG014 finding --
    summed gradients scaled as if they had been averaged -- which is what this fixture used
    to contain.
    """
    assert codes(
        """
        def train(model, loader, criterion, optimizer, accum=4):
            for i, (batch, y) in enumerate(loader):
                loss = criterion(model(batch), y)
                (loss / accum).backward()
                if (i + 1) % accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        """
    ) == []


def test_tg003_quiet_outside_a_loop():
    assert codes(
        """
        def one_step(model, batch, y, criterion, optimizer):
            loss = criterion(model(batch), y)
            loss.backward()
            optimizer.step()
        """
    ) == []


def test_tg003_quiet_in_lightning_module():
    assert codes(
        """
        import lightning as L

        class Lit(L.LightningModule):
            def training_step(self, batch, idx):
                x, y = batch
                for _ in range(2):
                    loss = self.criterion(self.model(x), y)
                    loss.backward()
                return loss
        """
    ) == []


# --------------------------------------------------------------------- TG004

_CUDA_PREAMBLE = """
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda")
"""


def test_tg004_flags_missing_workers_and_pin_memory():
    diagnostics = analyze(_CUDA_PREAMBLE + "loader = DataLoader(ds, batch_size=32)\n")
    assert [d.code for d in diagnostics] == ["TG004", "TG004"]
    assert any("num_workers" in d.message for d in diagnostics)
    assert any("pin_memory" in d.message for d in diagnostics)


def test_tg004_flags_explicit_zero_workers():
    diagnostics = analyze(
        _CUDA_PREAMBLE + "loader = DataLoader(ds, num_workers=0, pin_memory=True)\n"
    )
    assert [d.code for d in diagnostics] == ["TG004"]
    assert "num_workers=0" in diagnostics[0].message


def test_tg004_quiet_when_configured():
    assert codes(
        _CUDA_PREAMBLE + "loader = DataLoader(ds, num_workers=8, pin_memory=True)\n"
    ) == []


def test_tg004_quiet_without_a_gpu_target():
    assert codes(
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=32)
        """
    ) == []


# --------------------------------------------------------------------- TG005


def test_tg005_flags_softmax_wrapped_in_cross_entropy():
    diagnostics = analyze(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            logits = model(x)
            return F.cross_entropy(F.softmax(logits, dim=1), y)
        """
    )
    assert [d.code for d in diagnostics] == ["TG005"]
    assert diagnostics[0].fixable


def test_tg005_flags_criterion_variable():
    assert "TG005" in codes(
        """
        import torch
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()

        def step(model, x, y):
            return criterion(torch.softmax(model(x), dim=1), y)
        """
    )


def test_tg005_flags_indirect_softmax_variable():
    diagnostics = analyze(
        """
        import torch.nn as nn
        import torch.nn.functional as F

        criterion = nn.CrossEntropyLoss()

        def step(model, x, y):
            probs = F.softmax(model(x), dim=1)
            return criterion(probs, y)
        """
    )
    assert [d.code for d in diagnostics] == ["TG005"]
    assert not diagnostics[0].fixable


def test_tg005_flags_softmax_layer_in_model():
    assert "TG005" in codes(
        """
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()
        head = nn.Sequential(nn.Linear(8, 4), nn.Softmax(dim=1))
        """
    )


def test_tg005_quiet_for_an_attention_softmax_submodule():
    """Reduced from `pytorch/examples/gat/main.py`, where we fired wrongly.

    `self.softmax = nn.Softmax(dim=1)` normalises attention coefficients over neighbours.
    The model's output activation is `F.log_softmax`, which is exactly right for `NLLLoss`.
    Attention softmax appears in every transformer and GNN, so merely constructing the layer
    cannot be the evidence — final position in a `Sequential` can be.
    """
    assert codes(
        """
        import torch.nn as nn
        import torch.nn.functional as F

        class GraphAttentionLayer(nn.Module):
            def __init__(self, n_heads):
                super().__init__()
                self.leakyrelu = nn.LeakyReLU(0.2)
                self.softmax = nn.Softmax(dim=1)

        class GAT(nn.Module):
            def forward(self, x, adj):
                return F.log_softmax(self.out_layer(x, adj), dim=1)

        criterion = nn.NLLLoss()
        """
    ) == []


def test_tg005_reads_the_layer_class_not_the_attribute_name():
    """Reduced from `pytorch/tutorials/.../char_rnn_generation_tutorial.py`.

    `self.softmax = nn.LogSoftmax(dim=1)` feeding `NLLLoss` is correct code — the attribute
    is *named* softmax but bound to LogSoftmax. Trusting the name over the constructor two
    lines away reported PyTorch's own tutorial as a convergence bug.
    """
    assert codes(
        """
        import torch.nn as nn

        class RNN(nn.Module):
            def __init__(self, hidden_size, output_size):
                super().__init__()
                self.o2o = nn.Linear(hidden_size, output_size)
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x, hidden):
                output = self.o2o(x)
                output = self.softmax(output)
                return output, hidden

        criterion = nn.NLLLoss()

        def train(rnn, category_tensor, input_line_tensor, target_line_tensor):
            output, hidden = rnn(category_tensor, input_line_tensor)
            return criterion(output, target_line_tensor)
        """
    ) == []


def test_tg005_still_flags_a_real_softmax_layer_reaching_nll_loss():
    """The mirror of the two above: a real `nn.Softmax` does break `NLLLoss`, and the
    resolution has to work in both directions — the attribute here is *named* softmax and
    is bound to `nn.Softmax`, so it stays a finding.

    Note the shape: the activation's result is bound to a name that then reaches the loss.
    A model that returns `self.softmax(...)` straight out of `forward` is *not* caught, and
    cannot be without resolving `forward` through the call site — see the gap recorded in
    docs/rules.md.
    """
    assert "TG005" in codes(
        """
        import torch.nn as nn

        class RNN(nn.Module):
            def __init__(self, hidden_size, output_size):
                super().__init__()
                self.softmax = nn.Softmax(dim=1)

        criterion = nn.NLLLoss()

        def train(rnn, logits, target):
            probs = rnn.softmax(logits)
            return criterion(probs, target)
        """
    )


def test_tg005_quiet_on_raw_logits():
    assert codes(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            return F.cross_entropy(model(x), y)
        """
    ) == []


def test_tg005_quiet_for_nll_with_log_softmax():
    """``NLLLoss`` genuinely wants ``log_softmax`` — this pairing is correct."""
    assert codes(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            return F.nll_loss(F.log_softmax(model(x), dim=1), y)
        """
    ) == []


def test_tg005_flags_softmax_passed_to_nll():
    assert "TG005" in codes(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            return F.nll_loss(F.softmax(model(x), dim=1), y)
        """
    )


# ------------------------------------------------------------------ examples


def test_good_example_is_clean():
    from pathlib import Path

    from torch_preflight.engine import check_source

    path = Path(__file__).parent.parent / "examples" / "good_train.py"
    diagnostics, _ = check_source(str(path), path.read_text())
    assert diagnostics == [], [d.message for d in diagnostics]


def test_bad_example_triggers_every_rule():
    from pathlib import Path

    from torch_preflight.engine import check_source

    path = Path(__file__).parent.parent / "examples" / "bad_train.py"
    diagnostics, _ = check_source(str(path), path.read_text())
    assert {d.code for d in diagnostics} == {
        "TG001", "TG002", "TG003", "TG004", "TG005", "TG006", "TG007", "TG008",
        "TG011", "TG012", "TG013", "TG014",
    }


def test_tg001_quiet_for_non_differentiable_outputs():
    """``argmax`` returns indices with no graph — storing them is safe."""
    for stored in (
        "logits.argmax(-1)",
        "logits.argmax(dim=1)",
        "(logits > 0).long()",
        "logits.topk(5).indices",
        "torch.argmax(logits, dim=1)",
    ):
        assert codes(
            f"""
            import torch

            def evaluate(model, loader):
                preds = []
                with torch.no_grad():
                    for x, y in loader:
                        logits = model(x)
                        preds.append({stored})
                return preds
            """
        ) == [], stored


def test_tg001_still_flags_differentiable_reductions():
    """``.sum()``/``.mean()`` keep the graph, so they must still be caught."""
    for stored in ("loss.sum()", "loss.mean()", "loss.float()", "loss * 2"):
        assert "TG001" in codes(
            f"""
            def train(model, loader, criterion, optimizer):
                losses = []
                for batch, y in loader:
                    loss = criterion(model(batch), y)
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append({stored})
            """
        ), stored


def test_tg003_quiet_without_an_optimizer_step():
    """Raw autograd on a tensor recreated each iteration accumulates nothing.

    Found on torch's own test suite: a fresh leaf per iteration means `.grad` starts at
    None every time, so there is no stale gradient to clear.
    """
    assert codes(
        """
        import torch

        def bench(model):
            torch.manual_seed(0)
            for _ in range(10):
                x = torch.rand([1000, 1000], requires_grad=True)
                loss = (x * 2.0).sum()
                loss.backward()
        """
    ) == []


def test_tg003_ignores_scheduler_step():
    """`scheduler.step()` advances the LR; it applies no gradients."""
    assert "TG003" not in codes(
        """
        def train(model, loader, criterion, scheduler):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                loss.backward()
                scheduler.step()
        """
    )


def test_tg003_still_fires_with_an_optimizer_step():
    assert "TG003" in codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                loss.backward()
                optimizer.step()
        """
    )


# ------------------------------- regressions found on torch's own source (triage pass)


def test_tg002_eval_loader_needs_to_look_like_a_loader():
    """`models_to_test` is not a validation dataloader.

    In a test suite half the identifiers contain "test"; matching on that alone made
    TG002 fire on a training loop inside torch's distributed tests.
    """
    assert codes(
        """
        def _test_ddp_parity(models_to_test, inp, optimizer):
            for model in models_to_test:
                for _ in range(6):
                    optimizer.zero_grad()
                    out = model(inp)
                    out.sum().backward()
                    optimizer.step()
        """
    ) == []


def test_tg002_still_fires_on_a_real_validation_loader():
    assert "TG002" in codes(
        """
        def train(model, train_loader, val_loader, criterion, optimizer):
            for x, y in train_loader:
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            for x, y in val_loader:
                out = model(x)
        """
    )


def test_a_bare_prepare_call_is_not_a_model_wrapper():
    """torch's quantization utilities call their own function `prepare`."""
    assert codes(
        """
        def check(model, inputs, qconfig):
            model.eval()
            prepared = prepare(model, qconfig, example_inputs=inputs)
            prepared(*inputs)
        """
    ) == []


def test_accelerator_prepare_is_still_a_model_wrapper():
    assert "TG002" in codes(
        """
        def evaluate(accelerator, raw_model, loader):
            model = accelerator.prepare(raw_model)
            for batch in loader:
                out = model(batch)
        """
    )


def test_inner_scope_binding_shadows_an_outer_grad_name():
    """A nested helper with its own `loss` must not inherit grad-ness from outside."""
    assert codes(
        """
        def test_something(model, loader, criterion, optimizer):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            def get_loss(model_output):
                loss = 0.0
                for value in model_output.values():
                    loss += get_loss(value)
                return loss
        """
    ) == []


def test_autograd_grad_results_are_detached():
    """`torch.autograd.grad` returns detached tensors unless create_graph=True."""
    assert codes(
        """
        import torch

        def warmup_backward(f, *args):
            results = []
            for _ in range(3):
                r = torch.autograd.grad(f, *args)
                results.append(r)
            return results
        """
    ) == []


def test_autograd_grad_with_create_graph_still_retains():
    assert "TG001" in codes(
        """
        import torch

        def higher_order(f, *args):
            results = []
            for _ in range(3):
                r = torch.autograd.grad(f, *args, create_graph=True)
                results.append(r)
            return results
        """
    )


def test_model_names_do_not_leak_across_functions():
    """A model-ish binding in one function must not rename an unrelated local elsewhere.

    From torch's quantization utilities: `prepared = DistributedDataParallel(...)` in one
    helper made `prepared = prepare(...)` a thousand lines away look like a model.
    """
    assert codes(
        """
        from torch.nn.parallel import DistributedDataParallel

        def setup(rank, prepared):
            prepared = DistributedDataParallel(prepared, device_ids=[rank])
            return prepared

        def check_graph_op(model, inputs, qconfig):
            model.eval()
            prepared = prepare(model, qconfig, example_inputs=inputs)
            prepared(*inputs)
        """
    ) == []


def test_a_model_in_an_enclosing_scope_is_still_visible():
    """Scope-awareness must not break the ordinary nested-use case."""
    assert "TG002" in codes(
        """
        import torch
        from torch.nn.parallel import DistributedDataParallel

        def run(base, val_loader):
            wrapped = DistributedDataParallel(base)

            def evaluate():
                for batch in val_loader:
                    out = wrapped(batch)

            evaluate()
        """
    )


# --------------------------------------------------------------------- TG006


def test_tg006_flags_sigmoid_into_the_fused_loss():
    """Double sigmoid: the fused loss applies one itself."""
    assert "TG006" in codes(
        """
        import torch, torch.nn as nn
        def train(model, loader):
            criterion = nn.BCEWithLogitsLoss()
            for x, y in loader:
                loss = criterion(torch.sigmoid(model(x)), y)
                loss.backward()
        """
    )


def test_tg006_flags_sigmoid_through_a_variable():
    assert "TG006" in codes(
        """
        import torch, torch.nn as nn
        def train(model, loader):
            criterion = nn.BCEWithLogitsLoss()
            for x, y in loader:
                probs = torch.sigmoid(model(x))
                loss = criterion(probs, y)
                loss.backward()
        """
    )


def test_tg006_flags_raw_logits_into_plain_bce():
    """`log` of a negative number is `nan`, on the first negative logit."""
    assert "TG006" in codes(
        """
        import torch.nn as nn
        def train(model, loader):
            criterion = nn.BCELoss()
            for x, y in loader:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
        """
    )


def test_tg006_warns_on_the_numerically_fragile_but_correct_pairing():
    diagnostics = analyze(
        """
        import torch, torch.nn as nn
        def train(model, loader):
            criterion = nn.BCELoss()
            for x, y in loader:
                probs = torch.sigmoid(model(x))
                loss = criterion(probs, y)
                loss.backward()
        """
    )
    found = [d for d in diagnostics if d.code == "TG006"]
    assert found and all(d.severity.name == "WARNING" for d in found), (
        "sigmoid + BCELoss is correct, just fragile; it must not be an error"
    )


def test_tg006_silent_on_correct_usage():
    """Raw logits into the fused loss is the recommended pairing."""
    assert "TG006" not in codes(
        """
        import torch.nn as nn
        def train(model, loader):
            criterion = nn.BCEWithLogitsLoss()
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
        """
    )


def test_tg006_does_not_confuse_two_criterions_in_different_scopes():
    """Regression: `criteria` was a flat name->class map, so two functions each binding
    `crit` collided and whichever was parsed last decided the class for both. A correct
    `BCELoss` call was reported as a double-sigmoid error against `BCEWithLogitsLoss`."""
    diagnostics = analyze(
        """
        import torch, torch.nn as nn
        def fine(model, loader):
            crit = nn.BCELoss()
            for x, y in loader:
                probs = torch.sigmoid(model(x))
                crit(probs, y).backward()

        def also_fine(model, loader):
            crit = nn.BCEWithLogitsLoss()
            for x, y in loader:
                crit(model(x), y).backward()
        """
    )
    found = [d for d in diagnostics if d.code == "TG006"]
    assert len(found) == 1 and found[0].severity.name == "WARNING", (
        f"expected only the fragile-pairing warning, got {[(d.line, d.message) for d in found]}"
    )


def test_tg006_does_not_flag_a_bare_sigmoid_layer():
    """Regression: three false positives in torch/testing/_internal/common_nn.py.

    A local `sigmoid = nn.Sigmoid()` used to build a reference implementation is not a
    model ending in a sigmoid. Only final position in an `nn.Sequential` is evidence.
    """
    assert "TG006" not in codes(
        """
        import torch.nn as nn
        def reference_test():
            sigmoid = nn.Sigmoid()
            criterion = nn.BCEWithLogitsLoss()
            return sigmoid, criterion
        """
    )


def test_tg006_flags_sequential_ending_in_sigmoid():
    assert "TG006" in codes(
        """
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
                self.criterion = nn.BCEWithLogitsLoss()
        """
    )


def test_tg006_silent_without_any_bce_loss():
    assert "TG006" not in codes(
        """
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        """
    )


# --------------------------------------------------------------------- TG014


ACCUMULATION_LOOP = """
    import torch
    def train(model, loader, optimizer):
        accumulation_steps = 4
        for i, (x, y) in enumerate(loader):
            loss = torch.nn.functional.cross_entropy(model(x), y)
            {backward}
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    """


def test_tg014_flags_accumulation_without_scaling():
    assert "TG014" in codes(ACCUMULATION_LOOP.format(backward="loss.backward()"))


def test_tg014_silent_when_scaled_inline():
    assert "TG014" not in codes(
        ACCUMULATION_LOOP.format(backward="(loss / accumulation_steps).backward()")
    )


def test_tg014_silent_when_the_gradients_are_rescaled_instead():
    """Reduced from `torchtune/recipes/full_finetune_single_device.py`.

    Dividing the loss is the common compensation, not the only one. torchtune weights each
    micro-batch loss by its token count and rescales the gradients by `1/num_tokens` before
    stepping — a token-mean across micro-batches of unequal length, which is a *better*
    normalisation than dividing by the step count. Telling them to divide as well would
    shrink the gradient by the accumulation factor, so the false positive introduced a bug.
    """
    assert "TG014" not in codes(
        """
        def train(model, loader, optimizer, accumulation_steps, loss_fn, training):
            num_tokens = 0
            for i, (batch, y) in enumerate(loader):
                current_num_tokens = (batch["labels"] != -100).sum()
                num_tokens += current_num_tokens
                loss = loss_fn(model(batch), y) * current_num_tokens
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    training.scale_grads_(model.parameters(), 1.0 / num_tokens)
                    optimizer.step()
                    optimizer.zero_grad()
                    num_tokens = 0
        """
    )


def test_tg014_silent_when_the_rescaler_is_called_through_an_alias():
    """Reduced from `torchtune/recipes/full_finetune_distributed.py`.

    The recipe binds `self._grad_scaler = training.scale_grads_` (so it can wrap it in
    `torch.compile`) and calls it through that name, which on its own says nothing. Reading
    the binding rather than the call site is the same fix TG005 needed.
    """
    assert "TG014" not in codes(
        """
        class Recipe:
            def setup(self, training, torch):
                self._grad_scaler = training.scale_grads_
                self._grad_scaler = torch.compile(self._grad_scaler)

            def train(self, loader, optimizer, accumulation_steps, loss_fn):
                num_tokens = 0
                for i, (batch, y) in enumerate(loader):
                    loss = loss_fn(self._model(batch), y)
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0:
                        self._grad_scaler(
                            list(self._model.parameters()), self.world_size / num_tokens
                        )
                        optimizer.step()
                        optimizer.zero_grad()
        """
    )


def test_tg014_silent_when_scaled_by_reassignment():
    assert "TG014" not in codes(
        ACCUMULATION_LOOP.format(
            backward="loss = loss / accumulation_steps\n            loss.backward()"
        )
    )


def test_tg014_silent_when_scaled_in_place():
    assert "TG014" not in codes(
        ACCUMULATION_LOOP.format(
            backward="loss /= accumulation_steps\n            loss.backward()"
        )
    )


def test_tg014_silent_without_an_accumulation_guard():
    """One optimizer step per backward: nothing accumulates, so nothing needs scaling."""
    assert "TG014" not in codes(
        """
        import torch
        def train(model, loader, optimizer):
            for x, y in loader:
                loss = torch.nn.functional.cross_entropy(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        """
    )


def test_tg014_handles_an_integer_divisor():
    assert "TG014" in codes(
        """
        def train(model, loader, optimizer):
            for i, (x, y) in enumerate(loader):
                loss = model(x).sum()
                loss.backward()
                if i % 8 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        """
    )


def test_tg014_ignores_modulo_one():
    """`% 1` is every iteration, which is not accumulation, and dividing by 1 is a no-op."""
    assert "TG014" not in codes(
        """
        def train(model, loader, optimizer):
            for i, (x, y) in enumerate(loader):
                loss = model(x).sum()
                loss.backward()
                if i % 1 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        """
    )


def test_tg014_silent_when_a_framework_owns_the_scaling():
    """Accelerate divides internally; telling someone to divide again introduces a bug."""
    assert "TG014" not in codes(
        """
        from accelerate import Accelerator
        def train(model, loader, optimizer, accelerator):
            for i, (x, y) in enumerate(loader):
                with accelerator.accumulate(model):
                    loss = model(x).sum()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
        """
    )


# --------------------------------------------------------------------- TG012


def ddp_source(*body: str) -> str:
    """A DDP training function whose body is the given lines, indented consistently."""
    lines = "\n".join(f"        {line}" for line in body)
    return (
        "import torch.distributed as dist\n"
        "from torch.nn.parallel import DistributedDataParallel\n"
        "from torch.utils.data import DataLoader, DistributedSampler\n"
        "\n"
        "def train(dataset, model):\n"
        '    dist.init_process_group("nccl")\n'
        "    model = DistributedDataParallel(model)\n"
    ).replace("        ", "    ") + lines.replace("        ", "    ")


def test_tg012_flags_a_training_loader_without_a_sampler():
    assert "TG012" in codes(ddp_source("loader = DataLoader(dataset, batch_size=32, shuffle=True)"))


def test_tg012_silent_with_a_sampler_by_variable():
    assert "TG012" not in codes(ddp_source(
        "sampler = DistributedSampler(dataset)",
        "loader = DataLoader(dataset, sampler=sampler)",
    ))


def test_tg012_silent_with_an_inline_sampler():
    assert "TG012" not in codes(ddp_source(
        "loader = DataLoader(dataset, sampler=DistributedSampler(dataset))"
    ))


def test_tg012_silent_with_a_batch_sampler():
    """A custom batch sampler is presumed to handle sharding; second-guessing it is noise."""
    assert "TG012" not in codes(ddp_source(
        "loader = DataLoader(dataset, batch_sampler=batches)"
    ))


def test_tg012_silent_without_distributed_training():
    """Single-process training needs no sampler at all."""
    assert "TG012" not in codes(
        """
        from torch.utils.data import DataLoader
        def train(dataset):
            return DataLoader(dataset, batch_size=32, shuffle=True)
        """
    )


def test_tg012_warns_rather_than_errors_for_an_eval_loader():
    """Duplicated validation wastes work but computes the right number."""
    diagnostics = analyze(ddp_source("val_loader = DataLoader(dataset, batch_size=32)"))
    found = [d for d in diagnostics if d.code == "TG012"]
    assert found and all(d.severity.name == "WARNING" for d in found)


def test_tg012_separates_training_from_eval_loaders_in_one_file():
    diagnostics = analyze(ddp_source(
        "loader = DataLoader(dataset, shuffle=True)",
        "val_loader = DataLoader(dataset)",
    ))
    found = {d.severity.name for d in diagnostics if d.code == "TG012"}
    assert found == {"ERROR", "WARNING"}, found


def test_tg012_silent_when_accelerate_injects_the_sampler():
    """Accelerate re-creates loaders with a shard-aware sampler in `prepare`.

    Flagging this would be wrong twice: the code is already correct, and adding a sampler
    on top would shard data that is already sharded.
    """
    assert "TG012" not in codes(
        """
        from accelerate import Accelerator
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader
        def train(dataset, model):
            accelerator = Accelerator()
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model, loader = accelerator.prepare(model, loader)
        """
    )


def test_tg014_silent_for_a_scheduler_step():
    """`scheduler.step()` applies no gradients, so a modulo around it is not accumulation."""
    assert "TG014" not in codes(
        """
        def train(model, loader, optimizer, scheduler):
            for i, (x, y) in enumerate(loader):
                loss = model(x).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    scheduler.step()
        """
    )


def test_tg014_autofix_divides_only_what_autograd_sees():
    """`(loss / N).backward()` keeps any later logging of `loss` reporting the same value."""
    import textwrap

    from torch_preflight.engine import check_source
    from torch_preflight.fixer import apply_fixes

    source = textwrap.dedent(ACCUMULATION_LOOP.format(backward="loss.backward()")).lstrip("\n")
    diagnostics, ctx = check_source("t.py", source)
    fixed, applied = apply_fixes(ctx.module, [d for d in diagnostics if d.code == "TG014"])
    assert applied, "the TG014 fix should have applied"
    assert "(loss / accumulation_steps).backward()" in fixed
    assert "loss = torch.nn.functional.cross_entropy" in fixed, "must not touch the loss line"
def test_tg012_does_not_leak_across_functions():
    """A DDP setup in one function must not flag loaders in another.

    Regression: `uses_distributed` is a file-level fact, and firing on it alone flagged
    every DataLoader in the file — including single-process helpers that never touch DDP.
    Same file-wide leakage that once made `prov.models` and `criteria` report against
    unrelated names. The marker must be in the loader's own function or at module level.
    """
    diagnostics = analyze(
        """
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, DistributedSampler

        def single_process(dataset):
            return DataLoader(dataset, batch_size=32, shuffle=True)

        def distributed(dataset, model):
            dist.init_process_group("nccl")
            model = DistributedDataParallel(model)
            return DataLoader(dataset, batch_size=32, shuffle=True)
        """
    )
    found = [d for d in diagnostics if d.code == "TG012"]
    assert len(found) == 1, [(d.line, d.message) for d in found]
    # The DDP function's loader is the *second* DataLoader in the snippet.
    assert found[0].line > 8, f"flagged the single-process loader, at line {found[0].line}"


def test_tg012_fires_when_the_process_group_is_at_module_level():
    """Module-level setup governs every loader in the file."""
    assert "TG012" in codes(
        """
        import torch.distributed as dist
        from torch.utils.data import DataLoader

        dist.init_process_group("nccl")

        def train(dataset):
            return DataLoader(dataset, batch_size=32, shuffle=True)
        """
    )


# --------------------------------------------------------------------- TG011


def epoch_loop(*, train_call: str = "", eval_receiver: str = "model") -> str:
    """A train-then-validate epoch loop, with the `train()` call placed by the caller."""
    return (
        "import torch\n"
        "def fit(model, train_loader, val_loader, optimizer, criterion):\n"
        "    for epoch in range(10):\n"
        f"{train_call}"
        "        for x, y in train_loader:\n"
        "            loss = criterion(model(x), y)\n"
        "            optimizer.zero_grad()\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        f"        {eval_receiver}.eval()\n"
        "        for x, y in val_loader:\n"
        "            model(x)\n"
    )


def test_tg011_flags_eval_with_no_train_in_the_epoch_loop():
    assert "TG011" in codes(epoch_loop())


def test_tg011_silent_when_train_is_called_each_epoch():
    assert "TG011" not in codes(epoch_loop(train_call="        model.train()\n"))


def test_tg011_flags_train_called_only_before_the_loop():
    """The classic shape: `model.train()` once outside, so only epoch one trains properly."""
    source = (
        "import torch\n"
        "def fit(model, train_loader, val_loader, optimizer, criterion):\n"
        "    model.train()\n"
        "    for epoch in range(10):\n"
        "        for x, y in train_loader:\n"
        "            loss = criterion(model(x), y)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "        model.eval()\n"
        "        for x, y in val_loader:\n"
        "            model(x)\n"
    )
    assert "TG011" in codes(source)


def test_tg011_silent_for_a_deliberately_frozen_submodule():
    """`model.backbone.eval()` freezes batch-norm for fine-tuning and is not undone by
    `model.train()` — pairing them would both hide a real bug and invent a fake one."""
    assert "TG011" not in codes(
        epoch_loop(train_call="        model.train()\n", eval_receiver="model.backbone")
    )


def test_tg011_silent_without_a_backward_pass():
    """An evaluation-only script never trains, so eval mode cannot be stuck."""
    assert "TG011" not in codes(
        """
        def evaluate(model, test_loader):
            for epoch in range(3):
                model.eval()
                for x, y in test_loader:
                    model(x)
        """
    )


def test_tg011_silent_without_a_validation_pass():
    """`eval()` with no validation iteration is some other pattern; do not guess at it."""
    assert "TG011" not in codes(
        """
        def fit(model, train_loader, optimizer, criterion):
            for epoch in range(10):
                model.eval()
                for x, y in train_loader:
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
        """
    )


def test_tg011_silent_outside_any_loop():
    assert "TG011" not in codes(
        """
        def once(model, train_loader, val_loader, optimizer, criterion):
            for x, y in train_loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            model.eval()
            for x, y in val_loader:
                model(x)
        """
    )


def test_tg011_accepts_train_mode_restored_on_the_parent_module():
    """`model.train()` recurses, so it does restore `model.backbone.eval()`."""
    assert "TG011" not in codes(
        epoch_loop(train_call="        model.train()\n", eval_receiver="model.backbone.bn")
    )


# --------------------------------------------------------------------- TG013


def transfer_loop(*body: str) -> str:
    lines = "\n".join(f"        {line}" for line in body)
    return (
        "import torch\n"
        "def train(model, loader, optimizer, criterion, class_weights, device):\n"
        "    for x, y in loader:\n"
    ) + lines


def test_tg013_quiet_for_a_download():
    """Reduced from `pytorch/tutorials/intermediate_source/pinmem_nonblock.py`.

    The rule is about re-*uploading* the same data every iteration. `.to("cpu")` is a
    download, and the file it fired on is a tutorial whose whole subject is measuring
    transfer behaviour — the tensor is created with `device="cuda"` and copied to the host
    100 times deliberately. Wrong on both counts.
    """
    assert "TG013" not in codes(
        """
        import torch

        tensor = torch.arange(1, 1000, device="cuda")
        for i in range(100):
            cpu_tensor = tensor.to("cpu", non_blocking=True)
            torch.testing.assert_close(cpu_tensor.mean(), torch.tensor(500.0))
        """
    )


def test_tg013_quiet_when_restoring_the_device_after_a_deliberate_cpu_move():
    """Reduced from `pytorch/examples/fast_neural_style`.

    The model is moved to the host to write a checkpoint and moved back afterwards. That
    `.to(device)` is required, not redundant — hoisting it out would leave the model on the
    host for the rest of training.
    """
    assert "TG013" not in codes(
        """
        import torch

        def train(transformer, loader, device, optimizer):
            for batch_id, (x, _) in enumerate(loader):
                loss = transformer(x).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if batch_id % 100 == 0:
                    transformer.eval().cpu()
                    torch.save(transformer.state_dict(), "ckpt.pth")
                    transformer.to(device).train()
        """
    )


def test_tg013_flags_a_loop_invariant_transfer():
    assert "TG013" in codes(transfer_loop("w = class_weights.to(device)", "model(x)"))


def test_tg013_flags_a_host_factory_then_transfer():
    assert "TG013" in codes(transfer_loop("idx = torch.tensor([0, 1]).to(device)", "model(x)"))


def test_tg013_flags_moving_the_model_every_iteration():
    assert "TG013" in codes(transfer_loop("model.to(device)", "model(x)"))


def test_tg013_silent_for_the_batch_transfer():
    """`x.to(device)` on the loop variable is the whole point of the loop."""
    assert "TG013" not in codes(transfer_loop("xb = x.to(device)", "model(xb)"))


def test_tg013_silent_for_self_assignment():
    """`Tensor.to` returns self when already resident, so only iteration one copies."""
    assert "TG013" not in codes(
        transfer_loop("class_weights = class_weights.to(device)", "model(x)")
    )


def test_tg013_silent_when_the_factory_already_targets_the_device():
    assert "TG013" not in codes(
        transfer_loop("buf = torch.zeros(4, device=device)", "model(x)")
    )


def test_tg013_silent_for_a_dtype_cast():
    """`.to(dtype)` is not a transfer. Reading it as one produced false positives in
    torch's own FSDP code, so a device move now needs positive evidence."""
    assert "TG013" not in codes(transfer_loop("half = x.to(torch.float16)", "model(half)"))


def test_tg013_silent_for_a_bare_dtype_variable():
    assert "TG013" not in codes(transfer_loop("cast = x.to(dtype)", "model(cast)"))


def test_tg013_silent_when_the_destination_varies_per_iteration():
    """`clip_coef.to(device)` inside `for device, grads in ...` cannot be hoisted.

    Regression: found in torch's own `clip_grad`, where the loop iterates *over devices*.
    """
    assert "TG013" not in codes(
        """
        def clip(grouped_grads, clip_coef_clamped):
            for device, grads in grouped_grads.items():
                torch._foreach_mul_(grads, clip_coef_clamped.to(device))
        """
    )


def test_tg013_silent_for_a_comprehension_target():
    """`[t.cuda(rank) for t in tensors]` iterates, but binds in a comprehension scope.

    Regression: `bound_in_any_loop` does not see comprehension targets, which flagged every
    one of these in torch's distributed tests.
    """
    assert "TG013" not in codes(
        """
        def move(tensors, rank):
            for _ in range(3):
                moved = [t.cuda(rank) for t in tensors]
            return moved
        """
    )


def test_tg013_silent_for_an_attribute_of_a_loop_variable():
    """`for shard in shards: shard.tensor.to(device)` binds `shard`, not `shard.tensor`."""
    assert "TG013" not in codes(
        """
        def gather(shards, device):
            for shard in shards:
                out = shard.tensor.to(device)
            return out
        """
    )


def test_tg013_does_not_tell_you_to_hoist_a_random_draw():
    """`torch.randn(...)` must stay in the loop; only the double allocation is the problem."""
    diagnostics = analyze(transfer_loop("noise = torch.randn(4).to(device)", "model(x)"))
    found = [d for d in diagnostics if d.code == "TG013"]
    assert found, "the double allocation is still worth reporting"
    assert not any("once outside the loop" in (d.hint or "") for d in found), (
        "hoisting a random draw would change the semantics"
    )


def test_tg013_silent_outside_any_loop():
    assert "TG013" not in codes(
        """
        def setup(model, class_weights, device):
            model.to(device)
            return class_weights.to(device)
        """
    )


# --------------------------------------------------------------------- TG007


def test_tg007_flags_a_sync_in_a_nested_loop():
    assert "TG007" in codes(
        """
        def train(model, loader, optimizer, criterion):
            correct = 0
            for x, y in loader:
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                for i in range(len(preds)):
                    correct += preds[i].item()
        """
    )


def test_tg007_does_not_flag_its_own_recommended_fix():
    """`(preds == targets).sum().item()` is what this rule's hint tells you to write.

    Found by re-triaging seven real repos: all six TG007 findings were this shape, in a
    validation loop nested inside a training loop. The rule had a batch-loop exemption, but
    it matched iterable *names* — `loader`, `dataloader`, `batches` — and these loops iterate
    `dev_iter` and `valloader`.

    A longer name list would have patched the instance. The rule now requires evidence of
    per-element iteration instead: a loop over `range(...)` indexes elements, a loop over
    anything else yields batches whatever it is called.
    """
    assert "TG007" not in codes(
        """
        def run(model, loader, dev_iter, criterion, optimizer):
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                correct = 0
                for dev_batch in dev_iter:
                    preds = model(dev_batch.text)
                    correct += (preds.argmax(1) == dev_batch.label).sum().item()
        """
    )


def test_tg007_does_not_contradict_tg001():
    """`loss.item()` once per step is exactly what TG001 tells you to write.

    If this ever fires, the two rules are giving opposite advice about the same line and
    one of them has to change.
    """
    assert "TG007" not in codes(
        """
        def train(model, loader, optimizer, criterion):
            total = 0.0
            for x, y in loader:
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total += loss.item()
                acc = (preds.argmax(1) == y).sum().item()
            return total, acc
        """
    )


def test_tg007_flags_explicit_synchronize_in_the_training_loop():
    assert "TG007" in codes(
        """
        import torch
        def train(model, loader, optimizer, criterion):
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
        """
    )


def test_tg007_flags_a_sync_inside_a_comprehension():
    """`[p.item() for p in preds]` drains the queue once per element."""
    assert "TG007" in codes(
        """
        def train(model, loader, optimizer, criterion):
            for x, y in loader:
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                values = [p.item() for p in preds]
            return values
        """
    )


def test_tg007_silent_without_a_backward_pass():
    """A nested loop with syncs is only a training-loop problem."""
    assert "TG007" not in codes(
        """
        def collect(loader):
            out = []
            for batch in loader:
                for i in range(len(batch)):
                    out.append(batch[i].item())
            return out
        """
    )


def test_tg007_silent_for_a_sync_outside_the_loop():
    assert "TG007" not in codes(
        """
        def train(model, loader, optimizer, criterion):
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            return loss.item()
        """
    )


def test_tg007_silent_when_the_receiver_is_not_a_tensor():
    """`.tolist()` on a plain list and `.numpy()` on an ndarray involve no device."""
    assert "TG007" not in codes(
        """
        def train(model, loader, optimizer, criterion, history):
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
                for name in history:
                    rows = name.tolist()
            return rows
        """
    )


# --------------------------------------------------------------------- TG008


def test_tg008_no_seeding_at_all_is_a_note():
    """RFC 0003: often a choice, and often one this tool cannot see.

    Seeding frequently lives in the launcher or the job scheduler rather than the training
    script, and some runs deliberately want variance across invocations. We are reading the
    script, so we report it without claiming the run is defective.
    """
    diagnostics = analyze(
        """
        import torch
        def train(model, loader, optimizer, criterion):
            noise = torch.randn(4)
            for x, y in loader:
                loss = criterion(model(x + noise), y)
                loss.backward()
                optimizer.step()
        """
    )
    found = [d for d in diagnostics if d.code == "TG008"]
    assert found
    assert found[0].severity is Severity.NOTE


def test_tg008_flags_partial_seeding():
    """The common shape: torch seeded, the augmentation pipeline draws from NumPy."""
    diagnostics = analyze(
        """
        import numpy as np
        import torch
        def train(model, loader, optimizer, criterion):
            torch.manual_seed(42)
            for x, y in loader:
                jitter = np.random.rand(4)
                loss = criterion(model(x), y)
                loss.backward()
        """
    )
    found = [d for d in diagnostics if d.code == "TG008"]
    assert found, "seeding torch does nothing for NumPy"
    assert "NumPy" in found[0].message
    # A defect rather than a choice: the intent is visible in the code -- someone asked for
    # reproducibility -- and a generator is escaping anyway. So this one gates a build.
    assert found[0].severity is Severity.WARNING


def test_tg008_silent_when_every_generator_is_seeded():
    assert "TG008" not in codes(
        """
        import random
        import numpy as np
        import torch
        def train(model, loader, optimizer, criterion):
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            for x, y in loader:
                jitter = np.random.rand(4)
                pick = random.choice([0, 1])
                loss = criterion(model(x), y)
                loss.backward()
        """
    )


def test_tg008_silent_for_seed_everything_helpers():
    assert "TG008" not in codes(
        """
        import torch
        from transformers import set_seed
        def train(model, loader, optimizer, criterion):
            set_seed(42)
            for x, y in loader:
                loss = criterion(model(x + torch.randn(4)), y)
                loss.backward()
        """
    )


def test_tg008_silent_for_a_library_helper():
    """A file that draws randomly but does not train leaves seeding to its caller."""
    assert "TG008" not in codes(
        """
        import torch
        def augment(x):
            return x + torch.randn_like(x)
        """
    )


def test_tg008_silent_for_a_random_helper_beside_training_code():
    """Regression: `_trains` is a file-level fact, so a random helper in a file that
    trains *elsewhere* was flagged. Found in torch's own common_nn.py."""
    assert "TG008" not in codes(
        """
        import torch
        def make_input(*size):
            return torch.randperm(16).view(*size).double()

        def train(model, loader, optimizer, criterion):
            torch.manual_seed(0)
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
        """
    )


def test_tg008_silent_for_an_explicit_generator():
    """Regression: `torch.rand(..., generator=g)` is deliberately controlled randomness.

    torch's dist_optimizer_test builds a dedicated `Generator` precisely to avoid
    non-determinism, and counting that as unseeded was wrong.
    """
    assert "TG008" not in codes(
        """
        import torch
        def train(model, loader, optimizer, criterion):
            g = torch.Generator()
            g.manual_seed(0)
            w = torch.rand((3, 3), generator=g)
            for x, y in loader:
                loss = criterion(model(x @ w), y)
                loss.backward()
        """
    )


def test_tg007_silent_for_one_sync_per_validation_batch():
    """A loop over a dataloader yields batches, and one sync per batch is normal logging.

    Regression: found in this project's own `examples/bad_train.py`. The validation loop is
    nested inside the epoch loop, which contains the backward pass, so nesting alone flagged
    `print(val_loss.item())` — which is not the per-element thrashing this rule is about.
    """
    assert "TG007" not in codes(
        """
        def fit(model, loader, val_loader, optimizer, criterion):
            for epoch in range(10):
                for x, y in loader:
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                for x, y in val_loader:
                    val_loss = criterion(model(x), y)
                    print(val_loss.item())
        """
    )
