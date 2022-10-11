
import os, sys, re

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

basePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(basePath)

from sdgrpcserver.server import main
from sdgrpcserver.services.generate import GenerationServiceServicer
from sdgrpcserver.manager import EngineMode, EngineManager

import generation_pb2, generation_pb2_grpc

algorithms = {
    "ddim": generation_pb2.SAMPLER_DDIM,
    "plms": generation_pb2.SAMPLER_DDPM,
    "k_euler": generation_pb2.SAMPLER_K_EULER,
    "k_euler_ancestral": generation_pb2.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation_pb2.SAMPLER_K_HEUN,
    "k_dpm_2": generation_pb2.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation_pb2.SAMPLER_K_LMS,
}

args = {
    "sampler": ["ddim", "plms", "k_euler", "k_euler_ancestral", "k_lms"],
    "image": [
        {},
        {"image": True, "strength": 0.25},
        {"image": True, "strength": 0.5},
        {"image": True, "strength": 0.75},
        {"image": True, "mask": True, "strength": 0.5},
        {"image": True, "mask": True, "strength": 1},
        {"image": True, "mask": True, "strength": 1.5,}
    ]
}

with open("image.png", "rb") as file:
    test_image = file.read()

with open("mask.png", "rb") as file:
    test_mask = file.read()

class TestRunner(object):

    def sampler(self, item, request, prompt):
        request.image.transform.diffusion=algorithms[item]
    
    def image(self,  item, request, prompt):
        if item.get("image", False):
            prompt.append(generation_pb2.Prompt(
                parameters = generation_pb2.PromptParameters(
                    init=True
                ),
                artifact = generation_pb2.Artifact(
                    type=generation_pb2.ARTIFACT_IMAGE,
                    binary=test_image
                )
            ))


            request.image.parameters.append(generation_pb2.StepParameter(
                schedule=generation_pb2.ScheduleParameters(
                    start=item["strength"],
                    end=0.01,
                )
            ))

        if item.get("mask", False):
            prompt.append(generation_pb2.Prompt(
                artifact = generation_pb2.Artifact(
                    type=generation_pb2.ARTIFACT_MASK,
                    binary=test_mask
                )
            ))


    def build_combinations(self, args, idx):
        if idx == len(args.keys()) - 1:
            key = list(args.keys())[idx]
            return [{key: item} for item in args[key]]

        key = list(args.keys())[idx]
        result = []

        for item in args[key]:
            result += [{key: item} | combo for combo in self.build_combinations(args, idx+1)]

        return result


    def run(self, args, manager, context, prefix=""):
        combinations = self.build_combinations(args, 0)
        generator = GenerationServiceServicer(manager)

        for combo in combinations:
            request_id=re.sub('[^\w]+', '_', repr(combo))
            request_id=request_id.strip("_")

            prompt=[generation_pb2.Prompt(text="A frog wearing a tophat")]

            request = generation_pb2.Request(
                engine_id="stable-diffusion-v1-4",
                request_id=request_id,
                prompt=[],
                image=generation_pb2.ImageParameters(
                    height=512,
                    width=512,
                    seed=[420420420], # It's the funny number
                    steps=50,
                    samples=1,
                ),
            )

            for key, item in combo.items():
                getattr(self, key)(item, request, prompt)
            
            for part in prompt:
                request.prompt.append(part)
            
            for result in generator.Generate(request, context):
                for artifact in result.artifacts:
                    if artifact.type == generation_pb2.ARTIFACT_IMAGE:
                        with open(f"out/{prefix}{request_id}.png", "wb") as f:
                            f.write(artifact.binary)

class FakeContext():
    def add_callback(self, callback):
        pass

instance = TestRunner()

with open(os.path.normpath("testengines.yaml"), 'r') as cfg:
    engines = yaml.load(cfg, Loader=Loader)

for vramO in range(4):
    print("opt", vramO)

    manager = EngineManager(
        engines, 
        weight_root="../weights/",
        mode=EngineMode(vram_optimisation_level=vramO, enable_cuda=True, enable_mps=False), 
        nsfw_behaviour="flag"
    )

    manager.loadPipelines()

    instance.run(args, manager, FakeContext(), prefix=f"vram_{vramO}_")

