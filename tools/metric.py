import torch
from typing import List, Dict
from torchvision.utils import make_grid
from torchmetrics.metric import Metric
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from torchmetrics.classification import Accuracy, Specificity, MulticlassSpecificityAtSensitivity
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import piq

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
            'psnr': PeakSignalNoiseRatio(data_range=(-1.0,1.0)).to(device),
            'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
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
    def __init__(self, metrics: List[str], device: torch.device):
        self.metrics = metrics

        self.accumulator: Dict[str, Metric] = {
            "mse": MeanSquaredError().to(device),
            'psnr': PeakSignalNoiseRatio(data_range=(-1.0,1.0)).to(device),
            'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            "dice": GeneralizedDiceScore(num_classes=2, weight_type="linear").to(device),
            "miou": MeanIoU(num_classes=2).to(device),
            # 下面这两个有待完善
            'vif': None, 
            'gmsd': piq.GMSDLoss().to(device), 
        }

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        for metric_name in self.metrics:
            print(f"Updating metric: {metric_name}")
            self.accumulator[metric_name].update(output, target)

    def compute(self) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            results[metric_name] = self.accumulator[metric_name].compute().item()
        return results

def build_metric(cfg, device):
    if cfg.task == 'seg':
        return SegmentationMetric(cfg.metrics, device)
    elif cfg.task == 'recon':
        return ReconstructionMetric(cfg.metrics, device)
    elif cfg.task == 'all':
        return AllMetrics(cfg.metrics, device)


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