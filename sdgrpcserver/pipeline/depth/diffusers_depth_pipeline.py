import contextlib

import torch
from diffusers.pipeline_utils import DiffusionPipeline
from transformers import DPTForDepthEstimation
from transformers.models.clip import CLIPFeatureExtractor


class DiffusersDepthPipeline(DiffusionPipeline):
    feature_extractor: CLIPFeatureExtractor
    depth_estimator: DPTForDepthEstimation

    def __init__(
        self,
        feature_extractor: CLIPFeatureExtractor,
        depth_estimator: DPTForDepthEstimation,
    ):
        super().__init__()

        self.register_modules(
            feature_extractor=feature_extractor, depth_estimator=depth_estimator
        )

    @torch.no_grad()
    def __call__(self, tensor):
        sample = tensor
        device, dtype = self.depth_estimator.device, self.depth_estimator.dtype

        # CHW in RGB only (strip batch, strip A)
        sample = sample[0, 0:3]

        # Diffusers depth estimator uses the feature extractor as an image transformer
        # to prep pixels in right format
        sample = self.feature_extractor(images=sample, return_tensors="pt").pixel_values
        sample = sample.to(device, dtype)

        # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
        # So we use `torch.autocast` here for half precision inference.
        context_manger = (
            torch.autocast("cuda", dtype=dtype)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )

        # Do the actual estimation
        with context_manger:
            depth_map = self.depth_estimator(sample).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=tensor.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map.to(tensor.device, tensor.dtype)
