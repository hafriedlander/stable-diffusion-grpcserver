import glob
import os
import sys
import threading
from copy import deepcopy

import torch
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype

sdgrpcserver_path = __file__
while sdgrpcserver_path and os.path.split(sdgrpcserver_path)[1] != "sdgrpcserver":
    sdgrpcserver_path = os.path.dirname(sdgrpcserver_path)

sys.path.append(os.path.join(sdgrpcserver_path, "src", "midas"))

from midas.backbones.utils import activations
from midas.model_loader import default_models, load_model

# MiDaS library has global shared state. Need to ensure only one thread at a time inferences
global_midas_lock = threading.Lock()


class MidasModelWrapper(torch.nn.Module):
    def __init__(self, model, transform):
        super().__init__()
        self.model = model
        self.transform = transform

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def __deepcopy__(self, memo):
        self.model.pretrained.activations = None
        try:
            copy = deepcopy(self.model, memo)
            copy.pretrained.activations = activations
        finally:
            self.model.pretrained.activations = activations

        return MidasModelWrapper(copy, deepcopy(self.transform, memo))

    def preprocess(self, tensor):
        # Transform input is NumPy array in HWC format, but 0..1 (not 0..255)
        sample = tensor.permute(1, 2, 0).to(torch.float32).cpu().numpy()

        # Run transform. Output is NumPy array in CHW format
        sample = self.transform({"image": sample})["image"]

        # Convert back to BCHW tensor
        return torch.from_numpy(sample).to(tensor.device, tensor.dtype).unsqueeze(0)

    def forward(self, tensor):
        # Just run the model
        with global_midas_lock:
            return self.model(tensor)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        dtype = kwargs.get("torch_dtype", torch.float32)

        candidates = glob.glob(os.path.join(path, "*.pt"))

        if not candidates:
            raise EnvironmentError(f"No models in {path}")
        elif len(candidates) > 1:
            raise EnvironmentError(f"Too many models in {path}")

        model_path = candidates[0]
        model_type = os.path.splitext(os.path.split(model_path)[1])[0]

        if model_type not in default_models.keys():
            model_type = kwargs.get("midas_type")

        if model_type not in default_models.keys():
            raise EnvironmentError(f"Unknown MiDaS model type for path {model_path}")

        model, transform, _, _ = load_model("cpu", model_path, model_type, False)

        result = cls(model, transform)
        result.to(dtype)
        return result
