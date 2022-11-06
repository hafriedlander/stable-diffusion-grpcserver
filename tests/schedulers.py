from test_harness import TestHarness, VRAMUsageMonitor, ALGORITHMS
import os, sys, re, time
from types import SimpleNamespace as SN

import torch

import generation_pb2, generation_pb2_grpc

class TestRunner(TestHarness):

    def params(self, **extra):
        return {
            "height": 512,
            "width": 512,
            "guidance_scale": 7.5,
            "sampler": ALGORITHMS["k_euler_ancestral"],
            "eta": 0,
            "num_inference_steps": 50,
            "seed": -1,
            "strength": 0.8,
            **extra
        }

    def test(self):
        prompt = 'anime girl holding a giant NVIDIA Tesla A100 GPU graphics card, Anime Blu-Ray boxart, super high detail'
        seed = self.string_to_seed('hlky')

        for name, sampler in ALGORITHMS.items():
            kwargs = self.params(sampler=sampler, seed=seed)
            self.save_output(name, self.get_pipeline('testengine').generate(prompt=prompt, **kwargs)[0])

runner = TestRunner(engine_path="engines.sd14.yaml", prefix=f"seed", vramO=2)
runner.run()
