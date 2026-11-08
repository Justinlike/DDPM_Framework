from torchmetrics import Metric
import piq
import torch

class MeanGMSD(Metric):
    full_state_update = False

    def __init__(
        self,
        depth_dim: int = -1,
    ):
        super().__init__()

        # 当前数据预计为 [B, C, H, W, D]，
        # 因此深度维默认是最后一维。
        self.depth_dim = depth_dim

        self.add_state(
            "score_sum",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sample_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        # GMSD preds shape: (1, 1, 512, 512, 8) GMSD target shape: (1, 1, 512, 512, 8)
        # print(
        #     "GMSD preds shape:",
        #     tuple(preds.shape),
        #     "GMSD target shape:",
        #     tuple(target.shape),
        # )
        if preds.shape != target.shape:
            raise ValueError(
                "preds and target must have the same shape, "
                f"got preds={tuple(preds.shape)}, "
                f"target={tuple(target.shape)}"
            )

        if preds.ndim == 5:
            # 把深度维移动到最后：
            # [B, C, H, W, D]
            preds = preds.movedim(
                self.depth_dim,
                -1,
            )
            target = target.movedim(
                self.depth_dim,
                -1,
            )

            batch_size, channels, height, width, depth = (
                preds.shape
            )

            # [B, C, H, W, D]
            # → [B, D, C, H, W]
            # → [B*D, C, H, W]
            preds_2d = (
                preds.permute(0, 4, 1, 2, 3)
                .reshape(
                    batch_size * depth,
                    channels,
                    height,
                    width,
                )
            )

            target_2d = (
                target.permute(0, 4, 1, 2, 3)
                .reshape(
                    batch_size * depth,
                    channels,
                    height,
                    width,
                )
            )

            # 得到每张二维切片的 GMSD：
            # shape = [B*D]
            slice_scores = piq.gmsd(
                preds_2d,
                target_2d,
                data_range=1.0,
                reduction="none",
            )

            # [B*D] → [B, D]
            slice_scores = slice_scores.reshape(
                batch_size,
                depth,
            )

            # 每个 3D volume 内部对所有切片取平均：
            # shape = [B]
            scores = slice_scores.mean(dim=1)

        elif preds.ndim == 4:
            # 普通二维图像：[B, C, H, W]
            scores = piq.gmsd(
                preds,
                target,
                data_range=1.0,
                reduction="none",
            )

        else:
            raise ValueError(
                "MeanGMSD expects a 4D or 5D tensor, "
                f"got shape={tuple(preds.shape)}"
            )

        self.score_sum += scores.sum().to(
            dtype=self.score_sum.dtype,
            device=self.score_sum.device,
        )

        self.sample_count += torch.tensor(
            scores.numel(),
            dtype=self.sample_count.dtype,
            device=self.sample_count.device,
        )

    def compute(self) -> torch.Tensor:
        return (
            self.score_sum
            / self.sample_count.clamp_min(1)
        )