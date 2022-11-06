from test_harness import TestHarness, VRAMUsageMonitor, ALGORITHMS
import os, sys, re, time
from types import SimpleNamespace as SN

import torch

from sdgrpcserver import images
import generation_pb2, generation_pb2_grpc

class TestRunner(TestHarness):
    """
    Tests to make sure that the unified pipeline it batch-independant.

    Batch-independant means that we should get the same results for a single batch of four images,
    two batches of two images each, or four batches of a single image each, so long as the seeds are the same.

    e.g [1,2,3,4,] == [1,2], [3,4] == [1], [2], [3], [4]

    This should be true both from batches that are created by a single prompt with num_images_per_prompt,
    and for multiple prompts
    """



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

        def gen(args, prompts, seeds, tag):
            args = {
                **args, 
                "prompt": prompts,
                "seed": seeds, 
                "num_images_per_prompt": len(seeds) // len(prompts)
            }

            suffix=f"{mode}{'_clip' if args['clip_guidance_scale'] > 0 else ''}_{tag}_" 

            self.save_output(suffix, self.get_pipeline().generate(**args)[0])

        for mode in ["txt2img", "img2img", "inpaint"]:
            for clip_guidance in [1.0, 0]:

                kwargs = self.params(clip_guidance_scale = clip_guidance)
                if mode == "img2img" or mode == "inpaint": kwargs["init_image"] = image
                if mode == "inpaint": kwargs["mask_image"] = mask

                # Most common is going to be num_images_per_prompt, so check that first

                for i, seed in enumerate([[420420420, 420420421], [420420420], [420420421]]):
                    gen(kwargs, ["A Crocodile"], seed, f"croc{i}")

                # Then check 2 prompts and 2 images per prompt - first all four              

                seed = [420420420, 420420421, 520520520, 520520521]
                gen(kwargs, ["A Crocodile", "A Shark"], seed, f"both")

                # Then create the two sharks independantly

                for i, seed in enumerate([[520520520], [520520521]]):
                    gen(kwargs, ["A Shark"], seed, f"shark{i}")
                    

runner = TestRunner(engine_path="engines.clip.yaml", prefix=f"bi", vramO=2)
runner.run()
