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
            # 新加
            leave=False,
            disable=not self.accelerator.is_local_main_process, 
        )

        metrics = {}
        # PSNR: MAX, MSE: MIN
        if self.cfg.compare == 'max':
            best_score = 0
        else:
            best_score = 1e8
        global_step = 0

        for epoch in range(max_epoch):
            self.model.train()
            train_loss = 0.0

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
                    if self.args.criterion == "in":
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
                        self.accelerator.clip_grad_norm_(list(self.model.parameters()), 1.0)

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                # 累积到一定步数 (gradient_accumulation_steps) 后才会执行优化器的 step()，所以这里要判断是否执行了优化器的 step()
                if self.accelerator.sync_gradients:
                    if self.cfg.use_ema:
                        #ema_unet.step(unet.parameters())
                        pass   #ema 以后加
                    progress_bar.update(1)
                    global_step += 1


                    # EVALUATE the model on the validation set for save the best model checkpoints
                    #if self.accelerator.is_main_process:
                    if global_step > self.cfg.validation_start_steps and global_step % self.cfg.validation_steps == 0:
                        if self.cfg.use_ema:
                            self.accelerator.wait_for_everyone()

                        metrics = self.evaluate(global_step)

                        if self.cfg.use_ema:
                            self.accelerator.wait_for_everyone()

                        if self.accelerator.is_main_process and self.compare(best_score, metrics[self.cfg.monitor]):
                            # 保存最佳模型
                            save_model(self.cfg, epoch, self.model, self.accelerator, self.optimizer, loss_scaler=self.lr_scheduler, epoch_name="bset")
                            best_score = metrics[self.cfg.monitor]
                            self.logger.info(f"Bset {self.cfg.monitor} Score at epoch / step: {epoch} / {global_step}: {best_score:.4f}")

                # logs = {"step_loss": loss.detach().items(), 
                #         "lr": self.lr_scheduler.get_last_lr()[0],
                #         "train_loss": train_loss / self.cfg.gradient_accumulation_steps}
                logs = {"step_loss": loss.item(), # detach冗余，因为item取出的标量本身就不在计算图中了
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        # train 和 step loss 不能用来选择模型，单步step是随机的，大小只取决于随机出来的t
                        "train_loss": train_loss / self.cfg.gradient_accumulation_steps} 
                progress_bar.set_postfix({**logs, **metrics})

                # self.accelerator.log(logs, step) 会导致覆盖
                self.accelerator.log(logs, step=global_step)

            if global_step % self.cfg.save_step_freq == 0 and self.accelerator.is_main_process:
                save_model(self.cfg, epoch, self.model, self.accelerator, self.optimizer, loss_scaler=self.lr_scheduler, epoch_name="last")

        # 难不成都用的是Dice？
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"Training time {total_time_str}")
        self.logger.info(f"Best Moniter(PSNR) Score: {best_score:.4f}")

    def compare(self, best_score, cur_score):
        if self.cfg.compare == "min":
            return cur_score < best_score
        elif self.cfg.compare == "max":
            return cur_score > best_score
        # 相等的情况怎么办？
        else:
            raise("error compare !!! ")

    def evaluate(self, step):
        val_seed_ls = generate_seed_sequence(self.cfg.seed, len(self.valid_loader))
        self.model.eval()
        if not self.accelerator.is_main_process:
            return {}

        metrics = build_metric(self.cfg, self.accelerator.device)
        visual = Visualization() if self.cfg.visual else None

        with torch.autocast(self.accelerator.device.type):
            val_bar = tqdm(self.valid_loader, 
                           desc="Validation", 
                           leave=False,
                           disable=not self.accelerator.is_main_process)
            # for batch in tqdm(self.valid_loader, desc="Validation", disable=not self.accelerator.is_main_process):
            # for batch in self.valid_loader:
            for batch in val_bar:
                seed = val_seed_ls. pop()
                if seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=self.accelerator.device)
                    generator.manual_seed(seed)
                output = self.forward(batch, require_grad=False, **{"generator": generator})
                pred, gt = self.postprocess(batch, output)

                metrics.update(pred, gt)
                if visual is not None:
                    visual.update(batch["imgs"].cpu(), pred.cpu(), gt.cpu())

        if self.cfg.visual:
            grid = visual.sample(nrow=self.cfg.nrow)
            grid = np.clip(grid, 0.0, 1.0)
            images = Image.fromarray((grid * 255).astype(np.uint8))
            images.save(os.path.join(self.cfg.output_dir, self.cfg.job_name, "visualization", f"step-{step:06d}.jpg"))
        results = metrics.compute()
        self.accelerator.log(results, step=step)
        return results
    




                    

