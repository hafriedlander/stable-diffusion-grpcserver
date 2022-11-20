# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

SCHEDULER_CONFIG_NAME = "scheduler_config.json"

class KSchedulerMixin:
    """
    Mixin containing common functions for the schedulers.
    """

    config_name = SCHEDULER_CONFIG_NAME
    ignore_for_config = ["tensor_format"]

    def match_shape(self, values: Union[np.ndarray, torch.Tensor], broadcast_array: Union[np.ndarray, torch.Tensor]):
        """
        Turns a 1-D array into an array or tensor with len(broadcast_array.shape) dims.

        Args:
            values: an array or tensor of values to extract.
            broadcast_array: an array with a larger shape of K dimensions with the batch
                dimension equal to the length of timesteps.
        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]
            
        values = values.to(broadcast_array.device)

        return values

    """
    All the K-Schedulers handle these methods in the same way
    """

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        sigma = self.t_to_sigma(timestep)
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def add_noise(
        self,
        original_samples: Union[torch.FloatTensor, np.ndarray],
        noise: Union[torch.FloatTensor, np.ndarray],
        timesteps: Union[float, torch.FloatTensor],
    ) -> Union[torch.FloatTensor, np.ndarray]:
        index = self.t_to_index(timesteps)
        sigmas = self.match_shape(self.sigmas[index], noise)
        noisy_samples = original_samples + noise * sigmas

        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

    """
    Taken from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

    These assume that:
    len(self.timesteps) is num_inference_steps (not num_train_timesteps)
    len(self.sigmas) is num_inference_steps (not num_train_timesteps)
    
    BUT
    
    len(self.log_sigmas) is num_train_timesteps (not num_inference_steps)
    """

    def t_to_index(self, timestep):
        self.timesteps = self.timesteps.to(timestep.device)

        dists = timestep - self.timesteps
        return dists.abs().argmin().item()

    def sigma_to_t(self, sigma, quantize=True):
        self.log_sigmas = self.log_sigmas.to(sigma.device)

        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        # Stable Diffusion should be quantized
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        # For continuous distributions
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()
