import datetime
from tqdm import tqdm
from tools import *
import sys
from timm.utils import AverageMeter
from baser import Baser
import math
import torch
import time
import os
import numpy as np
from PIL import Image
import torchio as tio

class Trainer(Baser):
    def __init__(self, args, accelerator, model, ema_model, train_loader, valid_loader, 
                 loss_fn, optimizer, lr_scheduler, logger, **kwargs):
        super().__init__(args, model, loss_fn, **kwargs)
        self.model = model
        self.ema_model = ema_model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.cfg = args
        self.accelerator = accelerator

    def train(self, ):
        start_time = time.time()
        self.logger.info("Start Training")

        global_seed_sequence = generate_seed_sequence(
            initial_seed=self.cfg.seed,
            length=self.cfg.max_train_steps,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.accelerator.gradient_accumulation_steps
        )
        max_epoch = math.ceil(self.cfg.max_train_steps / num_update_steps_per_epoch)
        
        progress_bar = tqdm(
            range(0, self.cfg.max_train_steps),
            initial=0,
            desc="Traing Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process, 
        )

        metric = {}
        # PSNR: MAX, MSE: MIN
        if self.cfg.compare == 'max':
            best_score = 0
        else:
            best_score = 1e8
        global_step = 0

        for epoch in range(max_epoch):
            self.model.train()
            trian_loss = 0.0

            if global_step >= self.cfg.max_train_steps:
                break

            for step, batch in enumerate(self.train_loader):
                # globally consistent random generators 
                # 随机数生成器，用来控制后续的随机过程
                with self.accelerator.accumulate(self.model):
                    if self.cfg.seed is not None:
                        try:
                            local_seed = global_seed_sequence.pop()
                        except:
                            global_step += 1
                            break
                        rand_num_generator = torch.Generator(device=self.accelerator.device)
                        rand_num_generator.manual_seed(local_seed)
                    else:
                        rand_num_generator = None

                    output = self.forward(batch, **{'generator': rand_num_generator})
                    if self.args.ctiterion == "in":
                        loss = output["loss"]
                    else:
                        loss = self.compute_loss(output, batch['labels'])

                    if not math.isfinite(loss.item()):
                        print(f"Loss is {loss}, stopping training")
                        sys.exit(1)

                    # Gather the losses across all processes for logging (if we use distributed training).
                    # 多卡求平均
                    gathered_losses = self.accelerator.gather(loss.unsqueeze(0))
                    avg_loss = gathered_losses.mean().item()
                    train_loss += avg_loss

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(List(self.model.parameterss()), 1.0)

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.cfg.use_ema:
                        #ema_unet.step(unet.parameters())
                        pass   #ema 以后加
                    progress_bar.update(1)
                    global_step += 1
                    train_loss = 0.0

                    #if self.accelerator.is_main_process:
                    if global_step > self.cfg.validation_start_steps and global_step % self.cfg.validation_steps == 0:
                        if self.cfg.use_ema:
                            self.accelerator.wait_for_everyone()

                        metrics = self.evaluate(global_step)

                        if self.cfg.use_ema:
                            self.accelerator.wait_for_everyone()

                        if self.accelerator.is_main_process and self.compare(best_score, metrics[self.cfg.monitor]):
                            save_model(self.cfg, epoch, self.model, self.accelerator, self.optimizer, loss_scaler=self.lr_scheduler, epoch_name="bset")
                            best_score = metrics[self.cfg.monitor]
                            self.logger.info(f"Bset Dice Score at epoch / step: {epoch} / {global_step}: {best_score:.4f}")

                logs = {"step_loss": loss.detach().items(), 
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "train_loss": train_loss / self.cfg.gradient_accumulation_steps}
                progress_bar.set_postfix({**logs, **metrics})
                self.accelerator.log(logs, step)
            if global_step % self.cfg.save_step_freq == 0 and self.accelerator.is_main_process:
                save_model(self.cfg, epoch, self.model, self.accelerator, self.optimizer, loss_scaler=self.lr_scheduler, epoch_name="last")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"Training time {total_time_str}")
        self.logger.info(f"Best Dice Score: {best_score:.4f}")

    def compare(self, best_score, cur_score):
        if self.cfg.compare == "min":
            return cur_score < best_score
        elif self.cfg.compare == "max":
            return cur_score > best_score
        else:
            raise("error compare !!! ")

    def evaluate(self, step):
        val_seed_ls = generate_seed_sequence(self.cfg.seed, len(self.valid_loader))
        self.model.eval()
        if not self.accelerator.is_main_process:
            return {}

        metrics = build_metric(self.cfg, self.accelerator.device)
        visual = Visualization()
        with torch.autocast(self.accelerator.device.type):
            # 
            for batch in self.valid_loader:
                seed = val_seed_ls. pop()
                if seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=self.accelerator.device)
                    generator.manual_seed(seed)
                output = self.forward(batch, require_grad=False, **{"generator": generator})
                pred, gt = self.postprocess(batch, output)

                metrics.update()



                    

