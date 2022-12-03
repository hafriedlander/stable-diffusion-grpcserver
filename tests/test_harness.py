"""
isort:skip_file
"""

import os, sys, re, time, inspect, random

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch

basePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basePath)

# This line adds the various other module paths into the import searchpath
from sdgrpcserver.server import main

from sdgrpcserver.services.generate import GenerationServiceServicer
from sdgrpcserver.manager import EngineMode, EngineManager
from sdgrpcserver import images

import generation_pb2

from VRAMUsageMonitor import VRAMUsageMonitor

ALGORITHMS = {
    "ddim": generation_pb2.SAMPLER_DDIM,
    "plms": generation_pb2.SAMPLER_DDPM,
    "k_euler": generation_pb2.SAMPLER_K_EULER,
    "k_euler_ancestral": generation_pb2.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation_pb2.SAMPLER_K_HEUN,
    "k_dpm_2": generation_pb2.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation_pb2.SAMPLER_K_LMS,
    "dpm_fast": generation_pb2.SAMPLER_DPM_FAST,
    "dpm_adaptive": generation_pb2.SAMPLER_DPM_ADAPTIVE,
    "dpmspp_1": generation_pb2.SAMPLER_DPMSOLVERPP_1ORDER,
    "dpmspp_2": generation_pb2.SAMPLER_DPMSOLVERPP_2ORDER,
    "dpmspp_3": generation_pb2.SAMPLER_DPMSOLVERPP_3ORDER,
    "dpmspp_2s_ancestral": generation_pb2.SAMPLER_DPMSOLVERPP_2S_ANCESTRAL,
    "dpmspp_sde": generation_pb2.SAMPLER_DPMSOLVERPP_SDE,
    "dpmspp_2m": generation_pb2.SAMPLER_DPMSOLVERPP_2M,
}


class FakeContext:
    def __init__(self, monitor):
        self.monitor = monitor

    def add_callback(self, callback):
        pass

    def set_code(self, code):
        print("Test failed")
        self.monitor.stop()
        sys.exit(-1)

    def set_details(self, code):
        pass


class TestHarness:
    def __init__(self, engine_path, vramO=2, monitor=None, prefix=None):
        self.monitor_is_ours = False

        if monitor is None:
            self.monitor_is_ours = True
            monitor = VRAMUsageMonitor()

        self.monitor = monitor

        self.prefix = self.__class__ if prefix is None else prefix

        with open(os.path.normpath(engine_path), "r") as cfg:
            engines = yaml.load(cfg, Loader=Loader)

            self.manager = EngineManager(
                engines,
                weight_root="../weights/",
                mode=EngineMode(
                    vram_optimisation_level=vramO, enable_cuda=True, enable_mps=False
                ),
                nsfw_behaviour="ignore",
                refresh_on_error=True,
            )

            self.manager.loadPipelines()

    def get_pipeline(self, id="testengine"):
        return self.manager.getPipe(id)

    def call_generator(self, request):
        generator = GenerationServiceServicer(self.manager)
        context = FakeContext(self.monitor)

        return generator.Generate(request, context)

    def string_to_seed(self, string):
        return random.Random(string).randint(0, 2**32 - 1)

    def _flatten_outputs(self, output):
        if isinstance(output, list) or inspect.isgenerator(output):
            for item in output:
                yield from self._flatten_outputs(item)

        elif isinstance(output, torch.Tensor):
            if len(output.shape) == 4 and output.shape[0] > 1:
                yield from output.chunk(output.shape[0], dim=0)
            else:
                yield output

        elif isinstance(output, generation_pb2.Answer):
            yield from self._flatten_outputs(
                [
                    artifact
                    for artifact in output.artifacts
                    if artifact.type == generation_pb2.ARTIFACT_IMAGE
                ]
            )

        else:
            yield output

    def save_output(self, suffix, output):

        for i, output in enumerate(self._flatten_outputs(output)):
            path = (
                f"out/{self.prefix}_{suffix}_{i}.png"
                if i is not None
                else f"out/{self.prefix}_{suffix}.png"
            )

            if isinstance(output, torch.Tensor):
                binary = images.toPngBytes(output)[0]
                with open(path, "wb") as f:
                    f.write(binary)

            elif isinstance(output, generation_pb2.Artifact):
                with open(path, "wb") as f:
                    f.write(output.binary)

            else:
                raise ValueError(
                    f"Don't know how to handle output of class {output.__class__}"
                )

    def run(self):
        if self.monitor_is_ours:
            self.monitor.start()

        self.monitor.read_and_reset()
        start_time = time.monotonic()
        print("Running....")
        self.test()
        end_time = time.monotonic()
        used, total = self.monitor.read_and_reset()

        if self.monitor_is_ours:
            self.monitor.stop()

        runstats = {"vramused": used, "time": end_time - start_time}
        print("Run complete", repr(runstats))

        return runstats
