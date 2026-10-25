from tqdm import tqdm
from baser import Baser
from timm.utils import AverageMeter
import torch
from tools import *
import json
import cv2
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image
import torchio as tio
from torch.utils.data import DataLoader

class Tester(Baser):
    def __init__(self, args, model, test_dataset, test_loader, loss_fn, save_results = None):
        super().__init__(args, model, loss_fn, )
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.save_results = save_results
        self.cfg = args

    def test(self, mode="test"):
        metrics = build_metric(self.cfg, self.model.device.type)
        self.model.eval()
        with torch.autocast(self.model.device.type):
            # 遍历测试集所有subject 
            for subject in self.test_dataset.dataset:
                # patch sampler
                # 边缘用padding补 
                grid_sampler = tio.inference.GridSampler(
                    subject=subject,
                    patch_size = (512, 512, 8),
                )

                #loader = DataLoader(grid_sampler, batch_size=self.cfg.test_batch_size, shuffle=False, pin_memory=False, num_workers=16)
                # GridAggregator: 用来最后拼回整幅
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode = "average")

                with torch.no_grad():
                    for data in tqdm(grid_sampler):
                        # ### 调试信息
                        # print("tester data type =", type(data))
                        # if isinstance(data, dict):
                        #     for k, v in data.items():
                        #         if hasattr(v, "shape"):
                        #             print(k, v.shape)

                        output = self.forward(data, require_grad=False, **{"generator": None})
                        location_test = data[tio.LOCATION]
                        pred, gt = self.postprocess(data, output)
                        metrics.update(pred, gt)
                        location_test = location_test.reshape(1, -1)

                        aggregator.add_batch(pred, location_test)
                # 拿到整幅图像
                outputs_tensor = aggregator.get_output_tensor()
                self.save_test_3d(outputs_tensor, subject["ID"], subject["imgs"]["affine"])

        results = metrics.compute()
        print("Test results:", results)

    def save_test_3d(self, pred_results, name_test, affine):
        pred_results = 255 * ((pred_results + 1.0) / 2.0)
        print("pred_results shape:", pred_results.shape)
        print("pred_results min:", pred_results.min(), "max:", pred_results.max())
        pred_results = pred_results.cpu()
        pred_results = pred_results.type(torch.int32)
        print(pred_results.max(), pred_results.min())

        output_image = tio.ScalarImage(tensor = pred_results, affine=affine)
        output_image.save(os.path.join(self.cfg.output_dir, self.cfg.job_name, "visualization", f"{name_test}.nii.gz"))

    def visual(self, samples, outputs):
        os.makedirs(os.path.join(self.cfg.output_dir, self.cfg.job_name, "result"), exist_ok=True)
        output_path = os.path.join(self.cfg.output_dir, self.cfg.job_name, "result")
        
        for i, (gt, predicted) in enumerate(zip(samples["labels"], outputs["predicted"])):
            mask = predicted.permute(1, 2, 0)
            mask_arr = (mask.cpu().numpy() * 255).astype(np.uint8)

            img = Image.fromarray(mask_arr)
            img.save(os.path.join(output_path, f"{i}.png"))