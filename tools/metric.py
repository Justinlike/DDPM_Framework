import torch
from typing import List, Dict
from torchvision.utils import make_grid
from torchmetrics.metric import Metric
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from torchmetrics.classification import Accuracy, Specificity, MulticlassSpecificityAtSensitivity
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import time
from tools.gmsd import MeanGMSD

class SegmentationMetric:

    def __init__(self, metrics: List[str], device: torch.device):
        self.metrics = metrics
        self.accumulator: Dict[str, Metric] = {
            "dice": GeneralizedDiceScore(num_classes=2, weight_type="linear").to(device),
            "miou": MeanIoU(num_classes=2).to(device)
        }

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        for metric_name in self.metrics:
            self.accumulator[metric_name].update(output, target)
            
    def compute(self) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self.accumulator[metric_name].compute().item()
        return results

class ReconstructionMetric:

    def __init__(self, metrics: List[str], device: torch.device):
        self.metrics = metrics
        # 整幅 vs patch 的差值 — 若整幅 PSNR 明显低于 patch PSNR，说明 patch 边缘拼接有问题，GridAggregator 的 overlap_mode="average" 值得尝试 "hann"。
        self.accumulator: Dict[str, Metric] = {
            "mse": MeanSquaredError().to(device),
            'psnr': PeakSignalNoiseRatio(data_range=(0.0,1.0)).to(device),
            'ssim': StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).to(device),
        }

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        for metric_name in self.metrics:
            self.accumulator[metric_name].update(output, target)
            
    def compute(self) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self.accumulator[metric_name].compute().item()
        return results

class ClassificationMetric:
    #以后改
    pass
    def __init__(self, metrics: List[str], device: torch.device):
        self.metrics = metrics
        self.accumulator: Dict[str, Metric] = {
            "dice": GeneralizedDiceScore(num_classes=2, weight_type="linear").to(device),
            "miou": MeanIoU(num_classes=2).to(device)
        }

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        for metric_name in self.metrics:
            self.accumulator[metric_name].update(output, target)
            
    def compute(self) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self.accumulator[metric_name].compute().item()
        return results

class AllMetrics:
    def __init__(self, metrics: List[str], datarange: str, device: torch.device):
        self.datarange = datarange
        self.metrics = metrics
        # 在这里统一datarange，避免不同metric之间的差异导致结果不一致
        self.accumulator: Dict[str, Metric] = {
            # recon
            "mse": MeanSquaredError().to(device),
            'psnr': PeakSignalNoiseRatio(data_range=(0.0,1.0)).to(device),
            'ssim': StructuralSimilarityIndexMeasure(data_range=(0.0,1.0)).to(device),
            # seg
            "dice": GeneralizedDiceScore(num_classes=2, weight_type="linear").to(device),
            "miou": MeanIoU(num_classes=2).to(device),
            # quality
            # 下面这两个有待完善
            'vif': None, 
            'gmsd': MeanGMSD().to(device), 
            "fid": None,
        }

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        # 先对输出范围约束, constraint data range from [-1, 1] to [0, 1] 以统一计算指标
        output, target = dorange_inputs(output, target, self.datarange)

        # 时间调试信息
        # 总计 8s 左右
        # SSIM 比较慢 7s
        for metric_name in self.metrics:
            # time_begin = time.time()
            self.accumulator[metric_name].update(output, target)
            # time_end = time.time()
            # print(f"Updating metric: {metric_name}, takes {time_end - time_begin:.4f} seconds")
        
    def compute(self) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self.accumulator[metric_name].compute().item()
        return results

def dorange_inputs(output: torch.Tensor, target: torch.Tensor, datarange: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将输出和目标的范围约束到指定的 datarange 范围内，主要用于计算指标时的一致性。
    """
    # 非原地修改，防止影响梯度
    output = output.detach().float()
    target = target.detach().float()

    # 如果数据范围是 [-1, 1]，则将其转换为 [0, 1]
    if datarange == "m11":
        # print("Converting output and target from [-1, 1] to [0, 1]")
        output = (output + 1.0) / 2.0
        target = (target + 1.0) / 2.0
    elif datarange != "01":
        raise ValueError(
            f"Unsupported datarange: {datarange}. "
            'Expected "01" or "m11".'
        )

    # target 越界通常说明数据预处理存在问题，不应静默掩盖。
    eps = 1e-6
    target_min = target.min().item()
    target_max = target.max().item()

    if target_min < -eps or target_max > 1.0 + eps:
        raise ValueError(
            "Target is outside [0, 1] after conversion: "
            f"min={target_min:.6f}, max={target_max:.6f}"
        )

    target = target.clamp(0.0, 1.0)
    output = output.clamp(0.0, 1.0)
    
    return output, target

def build_metric(cfg, device):
    if cfg.task == 'seg':
        return SegmentationMetric(cfg.metrics, device)
    elif cfg.task == 'recon':
        return ReconstructionMetric(cfg.metrics, device)
    elif cfg.task == 'all':
        return AllMetrics(cfg.metrics, cfg.data_range, device)


class Visualization:

    def __init__(self):
        self.images = torch.tensor([])
        self.outputs = torch.tensor([])
        self.targets = torch.tensor([])

    def update(self, images: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> None:
       # output = output.repeat(1, 3, 1, 1)
       # target = target.repeat(1, 3, 1, 1)
        self.images = torch.cat([self.images, images.float()], dim=0)
        self.outputs = torch.cat([self.outputs, output.float()], dim=0)
        self.targets = torch.cat([self.targets, target.float()], dim=0)
    
    def sample(self, nrow: int, padding: int = 3):
        def add_padding(image: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(image, (padding, padding, padding, padding), mode="constant", value=1)
        
        target_tensors = [add_padding(tensor) for tensor in [self.images, self.targets, self.outputs]]
        units = [item for i in range(nrow) for item in (target_tensors[0][i], target_tensors[1][i], target_tensors[2][i])]
        grid = make_grid(units, nrow=nrow, normalize=False, padding=3)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        return grid_np