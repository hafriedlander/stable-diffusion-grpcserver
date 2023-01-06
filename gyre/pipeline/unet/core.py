import torch

from gyre.pipeline.unet.types import (
    DiffusersUNet,
    EpsTensor,
    ScheduleTimestep,
    XtTensor,
)


class UNetWithEmbeddings:
    def __init__(self, unet: DiffusersUNet, text_embeddings: torch.Tensor):
        self.unet = unet
        self.text_embeddings = text_embeddings

    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        return self.unet(latents, t, encoder_hidden_states=self.text_embeddings).sample
