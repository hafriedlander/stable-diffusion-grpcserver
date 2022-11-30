import torch


class VaeApproximator:
    """Decodes latent data to an approximate representation in RGB.
    Values determined experimentally for Stable Diffusion 1.4.
    See https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2
    """

    # grayscale_factors = torch.tensor([
    #    #    R       G       B
    #    [ 0.342,  0.341,  0.343 ], # L1
    #    [ 0.342,  0.342,  0.340 ], # L2
    #    [-0.110, -0.110, -0.113 ], # L3
    #    [-0.208, -0.209, -0.208 ]  # L4
    # ])

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.latent_rgb_factors = torch.tensor(
            [
                #   R        G        B
                [0.298, 0.207, 0.208],  # L1
                [0.187, 0.286, 0.173],  # L2
                [-0.158, 0.189, 0.264],  # L3
                [-0.184, -0.271, -0.473],  # L4
            ],
            dtype=dtype,
            device=device,
        )

    @classmethod
    def for_pipeline(cls, pipeline):
        return cls(device=pipeline.execution_device, dtype=pipeline.unet.dtype)

    def __call__(self, latents):
        """Get an RGB JPEG representation of the latent data."""
        self.to(latents.device, latents.dtype)
        return torch.einsum("...lhw,lr -> ...rhw", latents, self.latent_rgb_factors)

    def to(self, device, dtype):
        self.latent_rgb_factors = self.latent_rgb_factors.to(device, dtype)
