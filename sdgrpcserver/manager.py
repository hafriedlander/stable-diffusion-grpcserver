
import os, gc
from diffusers import StableDiffusionPipeline
import torch

class PipelineWrapper(object):

    def __init__(self, id, pipeline, device):
        self._id = id
        self._pipeline = pipeline
        self._device = device

    @property
    def id(self): return self._id

    @property
    def device(self): return self._device

    def activate(self):
        # Pipeline.to is in-place, so we move to the device on activate, and out again on deactivate
        self._pipeline.to(self._device)
        
    def deactivate(self):
        self._pipeline.to("cpu")
        if self._device == "cuda": torch.cuda.empty_cache()

    def generate(self, prompt, image):
        generator = torch.Generator(self._device).manual_seed(image.seed) if image.seed != -1 else None

        with torch.autocast(self._device):
            images = self._pipeline(
                prompt=prompt,
                width=image.width,
                height=image.height,
                num_inference_steps=image.steps,
                guidance_scale=7.5, # TODO: read from sampler parameters
                generator=generator,
                return_dict=False
            )

        return images

class EngineManager(object):

    def __init__(self, engines):
        self.engines = engines
        self._default = None
        self._pipelines = {}
        self._activeId = None
        self._active = None

        self._hasCuda = getattr(torch, 'cuda', False) and torch.cuda.is_available()
        self._hasMps = getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available()
        self._device = "cuda" if self._hasCuda else "mps" if self._hasMps else "cpu"

        self._token=os.environ.get("HF_API_TOKEN", True)

        self.loadPipelines()
    
    def _getWeightPath(self, remote_path, local_path):
        if os.path.isdir(os.path.normpath(local_path)): return local_path
        return remote_path

    def buildPipeline(self, engine):
        fp16 = engine.get('fp16', False) if self._device == "cude" else False
        revision = "fp16" if fp16 else "main"
        dtype = torch.float16 if fp16 else None

        if engine["class"] == "StableDiffusionPipeline":
            return PipelineWrapper(engine["id"], StableDiffusionPipeline.from_pretrained(
                self._getWeightPath(engine["model"], engine["local_model"]), 
                revision=revision, 
                torch_dtype=dtype, 
                use_auth_token=self._token if engine.get("use_auth_token", False) else False
            ), self._device)
    
    def loadPipelines(self):
        for engine in self.engines:
            if not engine.get("enabled", False): continue

            pipe=self.buildPipeline(engine)

            if pipe:
                self._pipelines[pipe.id] = pipe
                if engine.get("default", False): self._default = pipe
            else:
                raise Exception(f'Unknown engine class "{engine["class"]}"')

    def getPipe(self, id):
        """
        Get and activate a pipeline
        TODO: Better activate / deactivate logic. Right now we just keep a max of one pipeline active.
        """

        # If we're already active, just return it
        if self._active and id == self._active.id: return self._active

        # Otherwise deactivate it
        if self._active: self._active.deactivate()

        self._active = self._pipelines[id]
        self._active.activate()

        return self._active
            


