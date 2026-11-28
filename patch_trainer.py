"""Epoch-based patch training for single-process CT3D runs.

Integration: replace ``from trainer import Trainer`` in main.py with
``from patch_trainer import Trainer``. The constructor is unchanged.

Optional config:
    num_train_epochs: 100
    train_patches_per_subject: 16

Each epoch consumes every subject's sampled training patches (resampled next
 epoch). Validation visits every subject with a deterministic spatial grid.
Reported metrics aggregate PATCH predictions, not reconstructed whole volumes;
edge patches may overlap when the volume size is not divisible by patch size.
validation_steps/validation_start_steps/save_step_freq are replaced by epoch-end
validation and saving. If num_train_epochs is omitted, max_train_steps is rounded
UP to complete epochs, so the final update count can exceed max_train_steps.
Latent datasets keep their existing whole-latent-per-item behavior.
"""
import math
import json
import random
from pathlib import Path

import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.build import collate_fn
from data.ct import data_transform_3d
from tools.metric import build_metric
from tools.save_model import save_model
from trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_latent = self.cfg.data_range == "latent"
        if self.accelerator.num_processes != 1:
            raise ValueError("patch_trainer currently requires --num_processes 1")
        if self.cfg.dataset_name not in {"ct3d", "ct3d_latent"}:
            raise ValueError("patch_trainer supports ct3d and ct3d_latent")
        if self.cfg.use_ema:
            raise NotImplementedError("EMA is not implemented; use use_ema: false")

        train_dataset = self.train_loader.dataset
        self.valid_subjects = self.valid_loader.dataset.dataset
        self.overfit_one_patch = bool(self.cfg.get("overfit_one_patch", False))
        if self.overfit_one_patch:
            if self.is_latent:
                raise ValueError("overfit_one_patch currently supports image-space ct3d only")
            # Select by stable subject ID, not the filesystem's listing order.
            subject_id = str(self.cfg.fixed_subject_id)
            subjects = [s for s in train_dataset.subjects if str(s["ID"]) == subject_id]
            if len(subjects) != 1:
                raise ValueError(f"Expected one training subject with ID {subject_id}, found {len(subjects)}")
            # Use deterministic preprocessing: no training RandomFlip.
            subject = tio.SubjectsDataset(
                subjects, transform=data_transform_3d()["test"]
            )[0]
            grid = tio.GridSampler(subject, patch_size=tuple(self.cfg.resolution), patch_overlap=0)
            self.fixed_patch_indices = [int(i) for i in self.cfg.get(
                "fixed_patch_indices", [self.cfg.get("fixed_patch_index", 0)]
            )]
            if not self.fixed_patch_indices or len(set(self.fixed_patch_indices)) != len(self.fixed_patch_indices):
                raise ValueError("fixed_patch_indices must be nonempty and contain no duplicates")
            if any(i < 0 or i >= len(grid) for i in self.fixed_patch_indices):
                raise ValueError(f"fixed_patch_indices must be in [0, {len(grid) - 1}]")
            repeats = int(self.cfg.get("fixed_patch_repeats", 100))
            if repeats < 1:
                raise ValueError("fixed_patch_repeats must be positive")
            self.valid_subjects = []
            pairs = []
            debug_dir = Path(self.cfg.output_dir) / self.cfg.job_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            for patch_index in self.fixed_patch_indices:
                fixed_patch = grid[patch_index]
                pair = {key: fixed_patch[key].data.clone() for key in ("imgs", "labels")}
                pairs.append(pair)
                self.valid_subjects.append(fixed_patch)
                filename = ("fixed_patch.pt" if len(self.fixed_patch_indices) == 1
                            else f"fixed_patch-{patch_index:03d}.pt")
                torch.save({**pair, "subject_id": subject_id,
                            "location": fixed_patch[tio.LOCATION],
                            "patch_index": patch_index}, debug_dir / filename)
                self.logger.info(
                    f"Fixed training patch: subject={subject_id}, grid index={patch_index}, "
                    f"location={fixed_patch[tio.LOCATION].tolist()}"
                )
            # Each cached pair is used repeats times per epoch; no augmentation.
            patch_dataset = pairs * repeats
            self.logger.info(
                f"Fixed-patch fitting: {len(pairs)} patches, {repeats} repeats EACH. "
                "Evaluation uses these SAME training patches, not held-out data."
            )
        elif self.is_latent:
            patch_dataset = train_dataset
        else:
            samples = int(self.cfg.get("train_patches_per_subject", 16))
            if samples < 1:
                raise ValueError("train_patches_per_subject must be positive")
            patch_dataset = tio.Queue(
                train_dataset.dataset,
                max_length=16,
                samples_per_volume=samples,
                sampler=tio.UniformSampler(tuple(self.cfg.resolution)),
                num_workers=0,
                shuffle_subjects=True,
                shuffle_patches=True,
            )
        # Queue length is subjects * samples_per_volume, unlike CT3D.__len__.
        # A TorchIO Queue must be consumed by a DataLoader with zero workers.
        loader = DataLoader(
            patch_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=self.is_latent or self.overfit_one_patch,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.train_loader = self.accelerator.prepare_data_loader(loader)
        if len(self.train_loader) == 0:
            raise ValueError("Training dataset is empty")

    def train(self):
        updates_per_epoch = math.ceil(
            len(self.train_loader) / self.accelerator.gradient_accumulation_steps
        )
        epochs = self.cfg.get("num_train_epochs")
        if epochs is None:
            epochs = math.ceil(self.cfg.max_train_steps / updates_per_epoch)
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("num_train_epochs must be positive")
        self.logger.info(
            f"Patch training: {len(self.train_loader.dataset)} samples/epoch, "
            f"{updates_per_epoch} updates/epoch, {epochs} epochs; "
            "validation and checkpoint saving after every full epoch"
        )
        seed_rng = random.Random(self.cfg.seed)
        best_score = -math.inf if self.cfg.compare == "max" else math.inf
        global_step = 0
        self.optimizer.zero_grad()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            sample_count = 0
            progress = tqdm(self.train_loader, desc=f"Train epoch {epoch + 1}/{epochs}")
            for batch_index, batch in enumerate(progress):
                with self.accelerator.accumulate(self.model):
                    generator = None
                    if self.cfg.seed is not None:
                        generator = torch.Generator(device=self.accelerator.device)
                        generator.manual_seed(seed_rng.randrange(2**63))
                    output = self.forward(batch, generator=generator)
                    loss = self.compute_loss(output, batch["labels"])
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
                    count = batch["imgs"].shape[0]
                    epoch_loss += loss.detach().item() * count
                    sample_count += count
                    # Accelerate divides by accumulation_steps. Correct the last
                    # incomplete accumulation group so its gradients are averaged.
                    accumulation = self.accelerator.gradient_accumulation_steps
                    remainder = len(self.train_loader) % accumulation
                    scale = (
                        accumulation / remainder
                        if remainder and batch_index >= len(self.train_loader) - remainder
                        else 1.0
                    )
                    self.accelerator.backward(loss * scale)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    global_step += 1
                    logs = {
                        "step_loss": loss.item(),
                        "train_loss": epoch_loss / sample_count,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                    }
                    self.accelerator.log(logs, step=global_step)
                    progress.set_postfix(loss=logs["train_loss"], step=global_step)

            metrics = self.evaluate(global_step)
            score = metrics[self.cfg.monitor]
            if self.compare(best_score, score):
                best_score = score
                self._save(epoch, "best")
            self._save(epoch, "last")
            self.logger.info(
                f"Epoch {epoch + 1}: samples={sample_count}, "
                f"global_step={global_step}, validation={metrics}"
            )
        self.logger.info(f"Best {self.cfg.monitor}: {best_score}")

    def _save(self, epoch, name):
        save_model(
            self.cfg, epoch, self.model, self.accelerator, self.optimizer,
            loss_scaler=self.lr_scheduler, epoch_name=name,
        )

    @torch.no_grad()
    def evaluate(self, step):
        self.model.eval()
        # postprocess returns CPU tensors, so metric states must also be on CPU.
        metrics = build_metric(self.cfg, torch.device("cpu"))
        patches = 0
        patch_logs = {}
        for subject_index, subject in enumerate(tqdm(
            self.valid_subjects,
            desc="Evaluate fixed training patches" if self.overfit_one_patch else "Validate subjects",
        )):
            if self.is_latent or self.overfit_one_patch:
                subject_patches = [subject]
            else:
                subject_patches = tio.GridSampler(
                    subject,
                    patch_size=tuple(self.cfg.resolution),
                    patch_overlap=0,
                )
            loader = DataLoader(
                subject_patches,
                batch_size=self.cfg.test_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
            for batch in loader:
                output = self.forward(batch, require_grad=False, generator=None)
                prediction, target = self.postprocess(batch, output)
                metrics.update(prediction, target)
                if self.overfit_one_patch:
                    patch_index = self.fixed_patch_indices[subject_index]
                    patch_metrics = build_metric(self.cfg, torch.device("cpu"))
                    patch_metrics.update(prediction, target)
                    scores = patch_metrics.compute()
                    patch_logs.update({f"overfit_patch/patch_{patch_index:03d}/{key}": value
                                       for key, value in scores.items()})
                    self.logger.info(f"Fixed patch {patch_index}: {scores}")
                    self._save_fixed_patch_images(batch, prediction, target, step, patch_index)
                    destination = self._fixed_patch_image_dir(step, patch_index)
                    (destination / "metrics.json").write_text(json.dumps(scores, indent=2))
                patches += batch["imgs"].shape[0]
        results = metrics.compute()
        logged = (
            {f"overfit_patch/{key}": value for key, value in results.items()}
            if self.overfit_one_patch else results
        )
        logged.update(patch_logs)
        self.accelerator.log(logged, step=step)
        self.logger.info(
            (f"Training-patch evaluation: 1 subject, {patches} fixed patches"
             if self.overfit_one_patch else
             f"Validation: {len(self.valid_subjects)} subjects, {patches} patches")
        )
        self.model.train()
        return results

    def _fixed_patch_image_dir(self, step, patch_index):
        destination = (Path(self.cfg.output_dir) / self.cfg.job_name
                       / "fixed_patch_visuals" / f"step-{step:06d}")
        if len(self.fixed_patch_indices) > 1:
            destination = destination / f"patch-{patch_index:03d}"
        return destination

    def _save_fixed_patch_images(self, batch, prediction, target, step, patch_index):
        """Save every depth slice with fixed intensity/error scales across epochs."""
        from PIL import Image, ImageDraw

        destination = self._fixed_patch_image_dir(step, patch_index)
        destination.mkdir(parents=True, exist_ok=True)
        # Dataset inputs are [0,1]; postprocess outputs follow model_range.
        ncct = batch["imgs"][0].detach().cpu().float()
        predicted = prediction[0].detach().cpu().float()
        truth = target[0].detach().cpu().float()
        if self.model_range == "m11":
            predicted = (predicted + 1) / 2
            truth = (truth + 1) / 2
        # Match image metric clipping; retain unclipped prediction for diagnosis.
        unclipped_prediction = predicted.clone()
        predicted = predicted.clamp(0, 1)
        truth = truth.clamp(0, 1)
        error = (predicted - truth).abs()
        torch.save({
            "ncct": ncct, "ccta": truth, "predicted_ccta": predicted,
            "unclipped_prediction": unclipped_prediction, "absolute_error": error,
            "layout": "CHWD", "intensity_range": "normalized [0,1], not HU",
        }, destination / "volumes.pt")
        # Fixed 0..0.2 error scale: brighter means larger normalized error.
        # Never independently min/max-normalize the four images.
        for z in range(ncct.shape[-1]):
            planes = [ncct[0, :, :, z], truth[0, :, :, z],
                      predicted[0, :, :, z], error[0, :, :, z] / 0.2]
            height, width = planes[0].shape
            canvas = Image.new("RGB", (4 * width, height + 60), "white")
            draw = ImageDraw.Draw(canvas)
            labels = ["NCCT [0,1]", "True CCTA [0,1]",
                      "Pred CCTA [0,1]", "Abs error [0,0.2]"]
            for column, (plane, label) in enumerate(zip(planes, labels)):
                pixels = (plane.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                canvas.paste(Image.fromarray(pixels).convert("RGB"), (column * width, 40))
                draw.text((column * width + 4, 22), label, fill="black")
            draw.text((4, 4), f"Step {step}, patch {patch_index}, slice {z}; error white >= 0.2", fill="black")
            canvas.save(destination / f"slice-{z:03d}.png")
        self.logger.info(f"Fixed patch comparison images saved to {destination}")
