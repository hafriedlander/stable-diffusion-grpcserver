
import io
import PIL
import numpy as np
import cv2 as cv

from generated import generation_pb2

def artifact_to_image(artifact):
    if artifact.type == generation_pb2.ARTIFACT_IMAGE:
        img = PIL.Image.open(io.BytesIO(artifact.binary))
        return img

def image_to_artifact(im):
    #print(type(im), isinstance(im, PIL.Image.Image), isinstance(im, np.ndarray))
    binary=None

    if isinstance(im, PIL.Image.Image):
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        buf.seek(0)
        binary=buf.getvalue()
    else:
        binary=cv.imencode(".png", im)[1]

    return generation_pb2.Artifact(
        type=generation_pb2.ARTIFACT_IMAGE,
        binary=binary,
        mime="image/png"
    )

