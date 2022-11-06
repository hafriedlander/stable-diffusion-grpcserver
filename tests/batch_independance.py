from test_harness import TestHarness, VRAMUsageMonitor, ALGORITHMS
import os, sys, re, time
from types import SimpleNamespace as SN

import torch

from sdgrpcserver import images
import generation_pb2, generation_pb2_grpc

class TestRunner(TestHarness):

    def params(self, **extra):
        return {
            "height": 512,
            "width": 512,
            "guidance_scale": 7.5,
            "sampler": ALGORITHMS["plms"],
            "eta": 0,
            "num_inference_steps": 50,
            "seed": -1,
            "strength": 0.8,
            **extra
        }

    def test(self):
        with open("image.png", "rb") as file:
            test_image = file.read()
            image = images.fromPngBytes(test_image).to(self.manager.mode.device)

        with open("mask.png", "rb") as file:
            test_mask = file.read()
            mask = images.fromPngBytes(test_mask).to(self.manager.mode.device)

        for mode in ["txt2img", "img2img", "inpaint"]:
            for clip_guidance in [0, 1.0]:
                for i, seed in enumerate([[420420420], [420420421], [420420420, 420420421]]):

                    kwargs = self.params(seed = seed, clip_guidance_scale = clip_guidance)
                    if mode == "img2img" or mode == "inpaint": kwargs["init_image"] = image
                    if mode == "inpaint": kwargs["mask_image"] = mask

                    suffix=f"bt_{mode}_{clip_guidance}_{i}_" 
                    print(suffix)

                    self.save_output(suffix, self.get_pipeline().generate(prompt="A Crocodile", num_images_per_prompt=len(seed), **kwargs)[0])

runner = TestRunner(engine_path="engines.clip.yaml", prefix=f"seed", vramO=2)
runner.run()
