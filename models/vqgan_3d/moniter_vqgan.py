"""Opt-in CUDA memory diagnostics for VQGAN, without changing state_dict keys.

The first 3 training batches and first 3 discriminator-active batches per run
are monitored. Set VQGAN_MEMORY_BATCHES=0 to disable, or another nonnegative
integer to change the count. Each process prints its own GPU measurements.
Numbers include tensors retained by earlier stages; stage peaks are NOT additive.
Peak counters are reset per stage, so do not combine with a whole-step peak meter.
Decoder is monitored at final_block, each up/res1/res2, and conv_last rather
than at its outer boundary. This avoids nested resets and identifies OOM stages.
Set VQGAN_MODEL_PARALLEL=1 when using DecoderParallelStrategy in train_vqgan.py.
CUDA_VISIBLE_DEVICES selects physical GPUs; logical cuda:0 and cuda:1 are used.
The original module forward/loss code is reused via differentiable routing hooks.
No activation checkpointing is enabled.
With VQGAN_MODEL_PARALLEL=1 VQGAN_DECODER_GPUS=3, split each residual block:
res1 first half cuda:1, second half cuda:2; res2 first half cuda:0, second
half cuda:2. Final output convolution runs on cuda:0. Defaults to 2 GPUs.
Residual-half measurements cover all participating GPUs, without nested hooks.
VQGAN_UPDATE_DIAGNOSTICS=10 (default) reports the first 10 attempted updates
per optimizer per run: unscaled gradient finiteness, AMP scales, actual Adam
step execution and state counters. Set to 0 to disable the synchronizing checks.
These diagnostics do not change scaler decisions, gradients or optimizer state.
"""

import os
from contextlib import contextmanager
from types import MethodType
import weakref

import torch
from pytorch_lightning.strategies import SingleDeviceStrategy

from .vqgan import VQGAN, silu


def model_parallel_enabled():
    value = os.environ.get("VQGAN_MODEL_PARALLEL", "0")
    if value not in ("0", "1"):
        raise ValueError("VQGAN_MODEL_PARALLEL must be 0 or 1")
    return value == "1"


def decoder_gpu_count():
    value = os.environ.get("VQGAN_DECODER_GPUS", "2")
    if value not in ("2", "3"):
        raise ValueError("VQGAN_DECODER_GPUS must be 2 or 3")
    return int(value)


def _route_to_module_device(module, inputs):
    return (inputs[0].to(next(module.parameters()).device), *inputs[1:])


def _split_resblock_forward(block, x):
    """Original ResBlock operations, with device copies at two boundaries.

    Do not add dropout here: the original ResBlock.forward does not call it.
    The weak owner reference avoids registering a model inside its own child.
    """
    owner = block._memory_owner()
    first, second = block._split_devices
    with owner._memory_stage(block._split_name + ".norm1_silu_conv1"):
        x = x.to(first)
        h = block.norm1(x)
        h = silu(h)
        h = block.conv1(h)
    with owner._memory_stage(block._split_name + ".norm2_silu_conv2_add"):
        h = h.to(second)
        h = block.norm2(h)
        h = silu(h)
        h = block.conv2(h)
        x = x.to(second)
        if block.in_channels != block.out_channels:
            x = block.conv_shortcut(x)
        return x + h


class DecoderParallelStrategy(SingleDeviceStrategy):
    """One process, two visible CUDA devices; not data parallelism."""

    def __init__(self):
        if torch.cuda.device_count() < decoder_gpu_count():
            raise RuntimeError(f"Model parallel mode requires {decoder_gpu_count()} visible CUDA GPUs")
        super().__init__(device=torch.device("cuda:0"))

    def model_to_device(self):
        self.model.place_decoder_on_second_gpu()

    def load_optimizer_state_dict(self, checkpoint):
        states = checkpoint["optimizer_states"]
        if len(states) != len(self.optimizers):
            raise ValueError("Checkpoint optimizer count does not match VQGAN")
        for optimizer, state in zip(self.optimizers, states):
            # PyTorch restores moment tensors onto each parameter's device.
            # Do NOT invoke Lightning's subsequent move-all-to-root operation.
            optimizer.load_state_dict(state)


class MoniterVQGAN(VQGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._memory_batches = int(os.environ.get("VQGAN_MEMORY_BATCHES", "3"))
        if self._memory_batches < 0:
            raise ValueError("VQGAN_MEMORY_BATCHES must be nonnegative")
        self._memory_seen = 0
        self._memory_disc_seen = 0
        self._memory_enabled = False
        self._memory_phase = "idle"
        self._memory_device = None
        self._memory_batch_idx = None
        self._parallel_routes = []
        self._parallel_devices = None
        self._split_decoder = False
        self._parameter_devices = {}
        self._update_diagnostic_limit = int(os.environ.get("VQGAN_UPDATE_DIAGNOSTICS", "10"))
        if self._update_diagnostic_limit < 0:
            raise ValueError("VQGAN_UPDATE_DIAGNOSTICS must be nonnegative")
        self._update_diagnostic_counts = {}
        self._diagnosing_update = False
        self._loaded_checkpoint_summary = None

    def place_decoder_on_second_gpu(self):
        # Placement happens before optimizer creation. No registered wrapper
        # modules are introduced, so existing state_dict keys are unchanged.
        root, decoder_device = torch.device("cuda:0"), torch.device("cuda:1")
        self.to(root)
        self.post_vq_conv.to(decoder_device)
        self.decoder.to(decoder_device)
        self._parallel_devices = (root, decoder_device)
        for handle in self._parallel_routes:
            handle.remove()
        self._parallel_routes = [
            self.post_vq_conv.register_forward_pre_hook(self._route_to_decoder),
            self.decoder.register_forward_hook(self._route_to_root),
        ]
        self._split_decoder = decoder_gpu_count() == 3
        if self._split_decoder:
            self._parallel_devices = (root, decoder_device, torch.device("cuda:2"))
            for index, block in enumerate(self.decoder.conv_blocks):
                # Spread four large convolution/normalization halves over 3 GPUs.
                # GPU 0 already holds Encoder; GPU 1 holds the upsampling stage.
                for name, device_ids in (("res1", (1, 2)), ("res2", (0, 2))):
                    residual = getattr(block, name)
                    first, second = (torch.device(f"cuda:{i}") for i in device_ids)
                    residual.norm1.to(first)
                    residual.conv1.to(first)
                    residual.norm2.to(second)
                    residual.conv2.to(second)
                    if residual.in_channels != residual.out_channels:
                        residual.conv_shortcut.to(second)
                    residual._split_devices = (first, second)
                    residual._split_name = f"decoder.conv_blocks.{index}.{name}"
                    residual._memory_owner = weakref.ref(self)
                    residual.forward = MethodType(_split_resblock_forward, residual)
                    print(f"[VQGAN placement] {residual._split_name}: "
                          f"norm1/conv1={first}, norm2/conv2/add={second}", flush=True)
                self._parallel_routes.append(
                    block.up.register_forward_pre_hook(_route_to_module_device))
            self.decoder.conv_last.to(root)
            self._parallel_routes.append(
                self.decoder.conv_last.register_forward_pre_hook(_route_to_module_device))
        self._parameter_devices = {name: p.device for name, p in self.named_parameters()}
        print(f"[VQGAN placement] single process; GPUs={len(self._parallel_devices)}; "
              "Encoder/Codebook/losses=cuda:0; post_vq_conv/upsampling=cuda:1", flush=True)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self._loaded_checkpoint_summary = {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "legacy_steps": self._legacy_optimizer_steps,
        }

    @staticmethod
    def _adam_step_summary(optimizer):
        steps = [float(state["step"].item()) for state in optimizer.state.values()
                 if "step" in state]
        return f"states={len(steps)},min={min(steps) if steps else 0:g},max={max(steps) if steps else 0:g}"

    def on_train_start(self):
        if self._update_diagnostic_limit:
            scaler = getattr(self.trainer.precision_plugin, "scaler", None)
            print(f"[VQGAN restore] source={self._loaded_checkpoint_summary} "
                  f"epoch={self.current_epoch} global_step={self.global_step} "
                  f"legacy_steps={self._legacy_optimizer_steps} "
                  f"codebook_need_init={self.codebook._need_init} "
                  f"scale={scaler.get_scale() if scaler else None}", flush=True)
            for label, optimizer in zip(("G", "D"), self.optimizers(use_pl_optimizer=False)):
                print(f"[VQGAN restore] {label} Adam({self._adam_step_summary(optimizer)})", flush=True)
        if self._parallel_devices is None:
            return
        for name, parameter in self.named_parameters():
            expected = self._parameter_devices[name]
            if parameter.device != expected:
                raise RuntimeError(f"Wrong placement: {name} on {parameter.device}, expected {expected}")
        for optimizer in self.optimizers(use_pl_optimizer=False):
            for parameter, state in optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and state[key].device != parameter.device:
                        raise RuntimeError(f"Restored Adam {key} is on the wrong GPU")
        print("[VQGAN placement] parameter and restored optimizer devices verified", flush=True)

    def _route_to_decoder(self, module, inputs):
        return (inputs[0].to(self._parallel_devices[1]), *inputs[1:])

    def _route_to_root(self, module, inputs, output):
        # Tensor.to keeps autograd's cross-device CopyBackward chain intact.
        return output.to(self._parallel_devices[0])

    def _begin_memory(self, devices=None):
        devices = devices or self._parallel_devices or (self._memory_device,)
        before = {}
        for device in devices:
            torch.cuda.synchronize(device)
            before[device] = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        return before

    def _end_memory(self, name, before, status="ok"):
        for device, initial in before.items():
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_allocated(device)
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            gib = 1024 ** 3
            print(
                f"[VQGAN memory rank={self.global_rank} device={device} "
                f"batch={self._memory_batch_idx} phase={self._memory_phase}] {name} "
                f"status={status} before={initial / gib:.3f}GiB "
                f"after={after / gib:.3f}GiB peak={peak / gib:.3f}GiB "
                f"extra_peak={(peak - initial) / gib:.3f}GiB "
                f"reserved_peak={reserved / gib:.3f}GiB", flush=True,
            )

    @contextmanager
    def _memory_stage(self, name):
        if not self._memory_enabled:
            yield
            return
        before = self._begin_memory()
        try:
            yield
        except BaseException:
            # Preserve the original training error if CUDA reporting also fails.
            try:
                self._end_memory(name, before, "failed")
            except Exception:
                pass
            raise
        else:
            self._end_memory(name, before)

    def _memory_modules(self):
        # Never monitor both a module and its children: their peak resets clash.
        for name in ("encoder", "pre_vq_conv", "codebook", "post_vq_conv",
                     "image_discriminator", "video_discriminator",
                     "perceptual_model"):
            yield name, getattr(self, name)
        yield "decoder.final_block", self.decoder.final_block
        for index, block in enumerate(self.decoder.conv_blocks):
            for name in (("up",) if self._split_decoder else ("up", "res1", "res2")):
                yield f"decoder.conv_blocks.{index}.{name}", getattr(block, name)
        yield "decoder.conv_last", self.decoder.conv_last

    @staticmethod
    def _tensor_description(value):
        # Store metadata only, never retain tensor references for diagnostics.
        if isinstance(value, torch.Tensor):
            return f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        return type(value).__name__

    def _install_memory_hooks(self):
        handles = []
        for name, module in self._memory_modules():
            state = {}

            def before_hook(module, inputs, state=state, name=name):
                description = self._tensor_description(inputs[0]) if inputs else "no input"
                print(
                    f"[VQGAN memory rank={self.global_rank} device={inputs[0].device} "
                    f"batch={self._memory_batch_idx} phase={self._memory_phase}] "
                    f"{name} status=enter input({description})", flush=True,
                )
                state["before"] = self._begin_memory((inputs[0].device,))

            def after_hook(module, inputs, output, state=state, name=name):
                before = state.pop("before", None)
                if before is not None:
                    if output is None:
                        try:
                            self._end_memory(name, before, "failed")
                        except Exception:
                            pass
                    else:
                        self._end_memory(name, before)
                        if isinstance(output, torch.Tensor):
                            print(f"[VQGAN memory] {name} "
                                  f"output({self._tensor_description(output)})", flush=True)

            handles.append(module.register_forward_pre_hook(before_hook))
            handles.append(module.register_forward_hook(after_hook, always_call=True))
        return handles

    def forward(self, x, optimizer_idx=None, log_image=False):
        self._memory_phase = {0: "G", 1: "D"}.get(optimizer_idx, "evaluation")
        return super().forward(x, optimizer_idx=optimizer_idx, log_image=log_image)

    def training_step(self, batch, batch_idx):
        # The base loop increments the legacy counter once before the D branch.
        disc_active = (
            self._legacy_optimizer_steps + 1 >= self.cfg.model.discriminator_iter_start
        )
        selected = self._memory_seen < self._memory_batches or (
            disc_active and self._memory_disc_seen < self._memory_batches
        )
        self._memory_device = batch["data"].device
        self._memory_batch_idx = batch_idx
        self._memory_enabled = selected and self._memory_device.type == "cuda"
        self._memory_seen += 1
        if disc_active:
            self._memory_disc_seen += 1
        handles = []
        try:
            if self._memory_enabled:
                print(f"[VQGAN memory] input={tuple(batch['data'].shape)} "
                      f"dtype={batch['data'].dtype} disc_active={disc_active}", flush=True)
                handles = self._install_memory_hooks()
            return super().training_step(batch, batch_idx)
        finally:
            for handle in handles:
                handle.remove()
            self._memory_enabled = False

    def _manual_update(self, optimizer, loss):
        phase = self._memory_phase
        count = self._update_diagnostic_counts.get(phase, 0)
        diagnose = count < self._update_diagnostic_limit
        self._update_diagnostic_counts[phase] = count + 1
        self._diagnosing_update = diagnose
        raw_optimizer = optimizer.optimizer
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        scale_before = scaler.get_scale() if diagnose and scaler is not None else None
        before_steps = self._adam_step_summary(raw_optimizer) if diagnose else None
        executed = []
        handle = None
        if diagnose:
            # This hook runs only if GradScaler actually calls the underlying Adam.
            handle = raw_optimizer.register_step_post_hook(
                lambda opt, args, kwargs: executed.append(True))
            print(f"[VQGAN update] phase={phase} batch={self._memory_batch_idx} "
                  f"loss_finite={bool(torch.isfinite(loss.detach()).all().item())}", flush=True)
        try:
            with self._memory_stage("backward"):
                self.manual_backward(loss)
            with self._memory_stage("optimizer.step (including unscale/clip)"):
                optimizer.step()
            if diagnose:
                print(f"[VQGAN update] phase={phase} batch={self._memory_batch_idx} "
                      f"adam_step_executed={bool(executed)} "
                      f"scale_before={scale_before} "
                      f"scale_after={scaler.get_scale() if scaler is not None else None} "
                      f"adam_before=({before_steps}) "
                      f"adam_after=({self._adam_step_summary(raw_optimizer)})", flush=True)
        finally:
            if handle is not None:
                handle.remove()
            self._diagnosing_update = False

    def on_before_optimizer_step(self, optimizer):
        # Lightning invokes this AFTER unscale and BEFORE our clipping / Adam step.
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if self._diagnosing_update:
            names = {id(p): name for name, p in self.named_parameters()}
            stats = {}
            bad_names = []
            for p in parameters:
                key = str(p.device)
                entry = stats.setdefault(key, {"with_grad": 0, "missing_grad": 0,
                                               "nonfinite_grad_tensors": 0})
                if p.grad is None:
                    entry["missing_grad"] += 1
                    continue
                entry["with_grad"] += 1
                if not bool(torch.isfinite(p.grad).all().item()):
                    entry["nonfinite_grad_tensors"] += 1
                    if len(bad_names) < 5:
                        bad_names.append(names.get(id(p), "unknown"))
            print(f"[VQGAN gradients] phase={self._memory_phase} "
                  f"unscaled_before_clip={stats} first_nonfinite={bad_names}", flush=True)
        clip_val = self.cfg.model.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            # One global norm across all devices, after AMP unscaling.
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_val)
            if self._diagnosing_update:
                print(f"[VQGAN gradients] phase={self._memory_phase} "
                      f"unscaled_global_norm_before_clip={float(norm.item())}", flush=True)
