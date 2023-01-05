import torch

from sdgrpcserver import images
from sdgrpcserver.pipeline.depth.midas_model_wrapper import MidasModelWrapper


class MidasDepthPipeline:
    midas_depth_estimator: MidasModelWrapper

    def __init__(self, midas_depth_estimator):
        self.midas_depth_estimator = midas_depth_estimator

    def to(self, device):
        self.midas_depth_estimator.to(device)

    def pipeline_modules(self):
        return [("midas_depth_estimator", self.midas_depth_estimator)]

    @torch.no_grad()
    def __call__(self, tensor):
        sample = tensor

        # Get device and dtype of model
        device = self.midas_depth_estimator.device
        dtype = self.midas_depth_estimator.dtype

        # CHW in RGB only (strip batch, strip A)
        sample = sample[0, 0:3]

        # Convert sample to appropriate input
        sample = self.midas_depth_estimator.preprocess(sample)

        # And predict
        sample = sample.to(device, dtype)
        depth_map = self.midas_depth_estimator(sample)

        # _model output is a single monochrome 1HW. Convert to B1HW format and
        # resize back to original size

        depth_map = images.resize_right(
            depth_map.unsqueeze(1),
            out_shape=tensor.shape[-2:],
            interp_method=images.interp_methods.lanczos2,
            pad_mode="replicate",
            antialiasing=False,
        )

        # Normalise
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map.to(tensor.device, tensor.dtype)
