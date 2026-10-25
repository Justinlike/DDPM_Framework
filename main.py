import argparse
import logging
import math
import os
import shutil
from copy import deepcopy
import copy
from typing import Dict

from models import build_model
from tools import *
# import utils
# print("utils", utils.__file__)
from data import *
from losses import *
from trainer import Trainer
from tester import Tester

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm

import tensorboard
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

import diffusers
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

logger = get_logger(__name__, log_level = "INFO")

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Train Latent Diffusion model with VAE and UNet"
    )
    parser.add_argument(
        "--config", type = str, required = True, help = "Path to YAML config file"
    )
    return parser.parse_args()

def setup_directories(base_output: str, job_name: str) -> Dict[str, str]:
    out_run = os.path.join(base_output, job_name)
    dirs = {
        "run": out_run,
        "ckpt": os.path.join(out_run, "checkpoint"),
        "tb": os.path.join(out_run, "tensorboard"),
        "vis": os.path.join(out_run, "visualization"),
    }
    return dirs

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # create output directories
    dirs = setup_directories(cfg.output_dir, cfg.job_name)

    # define what tracker to use for logging, can be "wandb", "tensorboard", or "all"
    report_to = cfg.report_to
    if report_to == "all":
        trackers = ["wandb", "tensorboard"]
    elif report_to == "wandb":
        trackers = ["wandb"]
    elif report_to == "tensorboard":
        trackers = ["tensorboard"]
    else:
        raise ValueError(f"Invalid tracker specified: {report_to}. Choose from 'wandb', 'tensorboard', or 'all'.")

    # initialize accelerator
    accelerator = Accelerator(
        # gradient_accumulation_steps: accumulate steps for gradient updates
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        mixed_precision = cfg.mixed_precision,
        log_with = trackers,
        # save and log paths
        project_config = ProjectConfiguration(project_dir = dirs["run"], logging_dir = dirs["tb"]),
    )

    # config the logging 
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO, 
    )

    # make output dirs and initialize trackers
    if accelerator.is_main_process:
        for path in dirs.values():
            os.makedirs(path, exist_ok = True)

        init_kwargs = {}
        if "wandb" in trackers:
            init_kwargs["wandb"] = {
                "name": cfg.get("wandb_run_name") or cfg.job_name,
                "entity": cfg.get("wandb_entity", None),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "dir": dirs["run"],
            }

        accelerator.init_trackers(
            project_name=cfg.get("wandb_project", "ddpm_ct"),
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=init_kwargs,
        )

    # Logging configurations
    log_file = os.path.join(dirs["run"], "logging.log")
    log_format = logging.Formatter('%(asctime)s - %(levelname)s -%(filename)s - %(funcName)s >> %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    logger.logger.addHandler(file_handler)

    logger.info(cfg, main_process_only = True)
    logger.info(accelerator.state, main_process_only = False)

    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # -------------------- Device --------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------- Model --------------------
    # Create EMA for the unet.
    # 有待实现
    ema_unet = None

    # For mixed precision training, cast non-trainable weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = build_model(cfg).to(accelerator.device).to(weight_dtype)
    freeze_by_names(model, cfg.freeze_modules)

    # -------------------- Optimizer & loss & LR scheduler --------------------
    optimizer = build_optimizer(model, cfg)

    loss_fn = build_criterion(cfg)

    lr_scheduler = get_scheduler(
        name = cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes
    )

    # -------------------- Resume from checkpoint --------------------
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        

        print(f"Resume checkpoint {cfg.resume} loaded. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        # # load EMA model parameters if available in the checkpoint
        # if "model_ema" in checkpoint:
        #     ema_state_dict = checkpoint["model_ema"]
        #     ema_params = [ema_state_dict[name].cuda() for name, _ in model.named_parameters()]
        # else:
        #     model_params = list(model.parameters())
        #     ema_params = copy.deepcopy(model_params)

        # resume the optimizer and lr_scheduler states if available
        # start_epoch 没有被加载，且没有函数入口，待修复
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["scaler"])
            print("With optim & scheduler!")
        del checkpoint
        
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Numeber of trainable parameters: {}M".format(n_params / 1e6))

    # -------------------- DataLoader --------------------
    logger.info(f"Loading dataset {cfg.dataset_name}...")

    train_dataset, train_dataloader, valid_dataset, valid_dataloader, test_dataset, test_dataloader = build_loader(cfg)

    if cfg.use_ema:
        # 以后实现
        raise NotImplementedError("EMA 尚未实现，请保持 use_ema: false")

    # evaluate only mode
    if cfg.evaluate:
        if accelerator.is_main_process:
            torch.cuda.empty_cache()
            print("evaluating ... ")
            tester = Tester(cfg, model, test_dataset, test_dataloader, loss_fn)
            tester.test()
        accelerator.end_training()
        return

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # -------------------- Training --------------------
    trainer = Trainer(cfg, accelerator, model, None, train_dataloader, valid_dataloader, loss_fn, optimizer, lr_scheduler, logger)
    trainer.train()

    # -------------------- testing --------------------
    tester = Tester(cfg, model, test_dataset, test_dataloader, loss_fn)
    tester.test()

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
