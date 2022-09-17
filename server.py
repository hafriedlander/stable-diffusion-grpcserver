import argparse, io, os, sys
from types import SimpleNamespace as SN
from concurrent import futures

import numpy as np
import cv2 as cv
import PIL

import grpc
from flask import Flask, send_from_directory
from sonora.wsgi import grpcWSGI
from wsgicors import CORS

# Google protoc compiler is dumb about imports (https://github.com/protocolbuffers/protobuf/issues/1491)
# TODO: Move to https://github.com/danielgtaylor/python-betterproto
generatedPath = os.path.join(os.path.dirname(__file__), "generated")
sys.path.append(generatedPath)

from generated import generation_pb2, generation_pb2_grpc, dashboard_pb2, dashboard_pb2_grpc, engines_pb2, engines_pb2_grpc

import torch
from diffusers import StableDiffusionPipeline

def image_to_artifact(im):
    print(type(im), isinstance(im, PIL.Image.Image), isinstance(im, np.ndarray))

    buf = io.BytesIO()
    im.save(buf, format='PNG')
    buf.seek(0)

    return generation_pb2.Artifact(
        type=generation_pb2.ARTIFACT_IMAGE,
        binary=buf.getvalue(), #cv.imencode(".png", im)[1]
        mime="image/png"
    )

class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, pipe):
        self.pipe = pipe

    def unimp(self):
        raise NotImplementedError('Generation of anything except images not implemented')
        
    def Generate(self, request, context):
        try:
            # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
            if request.requested_type != generation_pb2.ARTIFACT_NONE and request.requested_type != generation_pb2.ARTIFACT_IMAGE:
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details('Generation of anything except images not implemented')
                raise NotImplementedError('Generation of anything except images not implemented')

            image=None
            mask=None
            text=""
            for prompt in request.prompt:
                which = prompt.WhichOneof("prompt")
                if which == "text": text += prompt.text
                elif which == "sequence": self.unimp()
                else: self.unimp()

            image=SN(
                height=512,
                width=512,
                samples=1,
                steps=50
            )

            for field in vars(image):
                if request.image.HasField(field):
                    setattr(image, field, getattr(request.image, field))
            
            image.seed = -1
            for seed in request.image.seed: image.seed = seed

            print(repr(image))

            generator = torch.Generator("cuda").manual_seed(image.seed) if image.seed != -1 else None

            with torch.autocast("cuda"):
                images = pipe(
                    prompt=text,
                    width=image.width,
                    height=image.height,
                    num_inference_steps=image.steps,
                    guidance_scale=7.5, # TODO: read from sampler parameters
                    generator=generator,
                    return_dict=False
                )

            print(repr(images))

            answer = generation_pb2.Answer()
            answer.request_id="x" # TODO - not this, copy from request
            answer.answer_id="y"
            for image in images[0]: answer.artifacts.append(image_to_artifact(image))

            yield answer
        except Exception as e:
            print(e)

class DashboardServiceServicer(dashboard_pb2_grpc.DashboardServiceServicer):
    def __init__(self):
        pass

    def GetMe(self, request, context):
        user = dashboard_pb2.User()
        user.id="0000-0000-0000-0001"
        return user

class EnginesServiceServicer(engines_pb2_grpc.EnginesServiceServicer):
    def __init__(self):
        pass

    def ListEngines(self, request, context):
        engine = engines_pb2.EngineInfo()
        engine.id="stable-diffusion-v1"
        engine.name="Stable Diffusion v1.4"
        engine.description="Stable Diffusion v1.4"
        engine.owner="stable-diffusion-grpcserver"
        engine.ready=True
        engine.type=engines_pb2.EngineType.PICTURE

        engines = engines_pb2.Engines()
        engines.engine.append(engine)

        return engines


class DartGRPCCompatibility(object):
    """Fixes a couple of compatibility issues between Dart GRPC-WEB and Sonora

    - Dart GRPC-WEB doesn't set HTTP_ACCEPT header, but Sonora needs it to build Content-Type header on response
    - Sonora sets Access-Control-Allow-Origin to HTTP_HOST, and we need to strip it out so CORSWSGI can set the correct value
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def wrapped_start_response(status, headers):
            headers = [header for header in headers if header[0] != 'Access-Control-Allow-Origin']
            return start_response(status, headers)
        
        if environ.get("HTTP_ACCEPT") == "*/*":
            environ["HTTP_ACCEPT"] = "application/grpc-web+proto"

        return self.app(environ, wrapped_start_response)

def serve(pipe):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(pipe), server)
    dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(DashboardServiceServicer(), server)
    engines_pb2_grpc.add_EnginesServiceServicer_to_server(EnginesServiceServicer(), server)

    server.add_insecure_port('[::]:50051')
    server.start()

    app = Flask(__name__)
    grpcapp = app.wsgi_app = grpcWSGI(app.wsgi_app)
    app.wsgi_app = DartGRPCCompatibility(app.wsgi_app)
    app.wsgi_app = CORS(app.wsgi_app, headers="*", methods="*", origin="*")

    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(pipe), grpcapp)
    dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(DashboardServiceServicer(), grpcapp)
    engines_pb2_grpc.add_EnginesServiceServicer_to_server(EnginesServiceServicer(), grpcapp)

    print("Ready, listening on port 50051")
    app.run(host='0.0.0.0')
    #server.wait_for_termination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", "-C", type=str, required=True, help="Path to the model"
    )
    args = parser.parse_args()

    pipe = None
    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, revision="fp16", torch_dtype=torch.float16) 
    pipe = pipe.to("cuda")
    print("Loaded pipe onto ", str(pipe.device))

    serve(pipe)

