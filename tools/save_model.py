import os
import torch
from pathlib import Path
import copy

def save_model(args, epoch, model, accelerator, optimizer, loss_scaler, ema_params=None, epoch_name=None):
    out_run = os.path.join(args.output_dir, args.job_name, 'checkpoint')
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(out_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / ('checkpoint-%s.ckpt' % epoch_name)

    # 关键1：用 Accelerator 的 is_main_process 判断主进程（比自定义 is_main_process 更可靠）
    if accelerator.is_main_process:
        # 关键2：自动检测并解除分布式模型包装（DDP/FSDP）
        # 处理 args 中的模型权重（若传入的是模型实例，先提取 state_dict 并 unwrap）
        if isinstance(model, torch.nn.Module):
                # 解除分布式包装（如 DDP -> 原始模型）
            unwrapped_model = accelerator.unwrap_model(model)
    else:
        return

    # ema
    if ema_params is not None:
            # 创建 EMA 模型的状态字典
            ema_state_dict = copy.deepcopy(unwrapped_model .state_dict())
            for i, (name, _value) in enumerate(unwrapped_model.named_parameters()):
                assert name in ema_state_dict
                ema_state_dict[name] = ema_params[i]
    else:
        ema_state_dict = None

    if args.excluding_modules is None:
        model_dict = unwrapped_model.state_dict()
    else:
        model_dict = save_excluding_modules(model, exclude_modules=args.excluding_modules)

    if epoch_name == 'best':
            # 只存模型
        to_save = {
            'model': model_dict,
            'args': args,
            }
    else:
        to_save = {
            'model': model_dict,
           # 'model_ema': ema_state_dict,    #以后再处理ema
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
    save_on_master(to_save, checkpoint_path)

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def save_excluding_modules(model, exclude_modules):
    """
    保存模型时排除特定模块的权重
    
    参数:
    model: 完整模型
    exclude_modules: 需要排除的模块名称列表（如['dropout', 'fc2']）
    save_path: 保存路径
    """
    # 获取完整状态字典
    full_state_dict = model.state_dict()
    
    # 筛选：排除以exclude_modules中模块名开头的参数
    filtered_state_dict = {}
    for name, param in full_state_dict.items():
        # 检查当前参数是否属于需要排除的模块
        exclude = False
        for module in exclude_modules:
            if name.startswith(f"{module}."):
                exclude = True
                break
        # 不排除的参数才保留
        if not exclude:
            filtered_state_dict[name] = param
    
    return filtered_state_dict