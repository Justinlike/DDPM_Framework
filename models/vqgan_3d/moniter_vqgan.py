"""Opt-in CUDA memory diagnostics for VQGAN, without changing state_dict keys.

The first 3 training batches and first 3 discriminator-active batches per run
are monitored. Set VQGAN_MEMORY_BATCHES=0 to disable, or another nonnegative
integer to change the count. Each process prints its own GPU measurements.
Numbers include tensors retained by earlier stages; stage peaks are NOT additive.
Peak counters are reset per stage, so do not combine with a whole-step peak meter.
Decoder is monitored at final_block, each up/res1/res2, and conv_last rather
than at its outer boundary. This avoids nested resets and identifies OOM stages.
"""

import os
from contextlib import contextmanager

import torch

from .vqgan import VQGAN


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

    def _begin_memory(self):
        device = self._memory_device
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        return before

    def _end_memory(self, name, before, status="ok"):
        device = self._memory_device
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_allocated(device)
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        gib = 1024 ** 3
        print(
            f"[VQGAN memory rank={self.global_rank} device={device} "
            f"batch={self._memory_batch_idx} phase={self._memory_phase}] {name} "
            f"status={status} before={before / gib:.3f}GiB "
            f"after={after / gib:.3f}GiB peak={peak / gib:.3f}GiB "
            f"extra_peak={(peak - before) / gib:.3f}GiB "
            f"reserved_peak={reserved / gib:.3f}GiB",
            flush=True,
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
            for name in ("up", "res1", "res2"):
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
                    f"[VQGAN memory rank={self.global_rank} device={self._memory_device} "
                    f"batch={self._memory_batch_idx} phase={self._memory_phase}] "
                    f"{name} status=enter input({description})", flush=True,
                )
                state["before"] = self._begin_memory()

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
        # Same operations/order as VQGAN._manual_update; only diagnostic scopes added.
        with self._memory_stage("backward"):
            self.manual_backward(loss)
        clip_val = self.cfg.model.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            with self._memory_stage("gradient_clip"):
                self.clip_gradients(optimizer, gradient_clip_val=clip_val,
                                    gradient_clip_algorithm="norm")
        with self._memory_stage("optimizer.step"):
            optimizer.step()
