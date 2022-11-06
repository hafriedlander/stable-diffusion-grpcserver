from test_harness import TestHarness, VRAMUsageMonitor, ALGORITHMS
import os, sys, re, time

import torch

import generation_pb2, generation_pb2_grpc

args = {
    "sampler": [
        {"sampler": "ddim", "eta": 0},
        {"sampler": "ddim", "eta": 0.8},
        {"sampler": "plms"}, 
        {"sampler": "k_lms"},
        {"sampler": "k_euler"}, 
        {"sampler": "k_euler_ancestral"}, 
        {"sampler": "k_heun"}, 
        {"sampler": "k_dpm_2"}, 
        {"sampler": "k_dpm_2_ancestral"}, 
    ],
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

class TestRunner(TestHarness):

    def __init__(self, combos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combos = combos

    def sampler(self, item, request, prompt, parameters):
        request.image.transform.diffusion=ALGORITHMS[item["sampler"]]

        eta = item.get("eta", None)
        if eta != None:
            parameters.sampler.eta=eta

    def image(self,  item, request, prompt, parameters):
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


            parameters.schedule.start=item["strength"]
            parameters.schedule.end=0.01

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
            result += [{**combo, key: item} for combo in self.build_combinations(args, idx+1)]

        return result


    def test(self):
        combinations = self.build_combinations(self.combos, 0)

        for combo in combinations:
            request_id=re.sub('[^\w]+', '_', repr(combo))
            request_id=request_id.strip("_")

            prompt=[generation_pb2.Prompt(text="A digital painting of a shark in the deep ocean, highly detailed, trending on artstation")]

            parameters = generation_pb2.StepParameter()

            request = generation_pb2.Request(
                engine_id="testengine",
                request_id=request_id,
                prompt=[],
                image=generation_pb2.ImageParameters(
                    height=512,
                    width=512,
                    seed=[420420420], # It's the funny number
                    steps=50,
                    samples=1,
                    parameters = []
                ),
            )

            for key, item in combo.items():
                getattr(self, key)(item, request, prompt, parameters)
            
            for part in prompt:
                request.prompt.append(part)
            
            request.image.parameters.append(parameters)

            self.save_output(request_id, self.call_generator(request))

monitor = VRAMUsageMonitor()
monitor.start()

stats = {}

for vramO in [2]: #range(4):
    instance = TestRunner(engine_path="engines.inpaint.yaml", combos=args, prefix=f"hp_{vramO}", vramO=vramO, monitor=monitor)
    stats[f"run vram-optimisation-level={vramO}"] = instance.run()

monitor.stop()

print("Stats")
print(repr(stats))

