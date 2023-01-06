
import io
from array import ArrayType

import cv2 as cv
import generation_pb2
import numpy as np
import PIL
import torch
from PIL import PngImagePlugin

from gyre import images
from gyre.pipeline.vae_approximator import VaeApproximator


def artifact_to_image(artifact):
    if artifact.type == generation_pb2.ARTIFACT_IMAGE or artifact.type == generation_pb2.ARTIFACT_MASK:
        img = PIL.Image.open(io.BytesIO(artifact.binary))
        return img
    else:
        raise NotImplementedError("Can't convert that artifact to an image")

def image_to_artifact(im, artifact_type=generation_pb2.ARTIFACT_IMAGE, meta=None):
    binary=None

    if isinstance(im, torch.Tensor):
        im = images.toPIL(im)[0]
 
    if isinstance(im, PIL.Image.Image):
        buf = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        if meta:
            for k, v in meta.items(): info.add_text(k, v)
        im.save(buf, format='PNG', pnginfo=info)
        buf.seek(0)
        binary=buf.getvalue()
    else:
        binary=cv.imencode(".png", im)[1]

    return generation_pb2.Artifact(
        type=artifact_type,
        binary=binary,
        mime="image/png"
    )

class CallbackImageWrapper:
    def __init__(self, callback, device, dtype):
        self.callback = callback
        self.vae_approximator = VaeApproximator()
    
    def __call__(self, i, t, latents):
        pixels = self.vae_approximator(latents)
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        self.callback(i, t, pixels)
