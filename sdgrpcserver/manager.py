
import os, gc, warnings
import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler
from diffusers.configuration_utils import FrozenDict

from generated import generation_pb2
from sdgrpcserver.unified_pipeline import UnifiedPipeline

class WithNoop(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

class PipelineWrapper(object):

    def __init__(self, id, pipeline, device, vramO=0):
        self._id = id
        self._pipeline = pipeline
        self._device = device
        self._vramO = vramO

        self._plms = pipeline.scheduler
        self._klms = self._prepScheduler(LMSDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._ddim = self._prepScheduler(DDIMScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear", 
                clip_sample=False, 
                set_alpha_to_one=False
            ))

    def _prepScheduler(self, scheduler):
        scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            warnings.warn(
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file",
                DeprecationWarning,
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        return scheduler

    @property
    def id(self): return self._id

    @property
    def device(self): return self._device

    def activate(self):
        if self._vramO > 0: self._pipeline.enable_attention_slicing(1)

        # Pipeline.to is in-place, so we move to the device on activate, and out again on deactivate
        if self._vramO > 1 and self._device == "cuda": self._pipeline.unet.to(torch.device("cuda"))
        else: self._pipeline.to(self._device)
        
    def deactivate(self):
        self._pipeline.to("cpu")
        if self._device == "cuda": torch.cuda.empty_cache()

    def _autocast(self):
        if self._device == "cuda": return torch.autocast(self._device)
        return WithNoop()

    def generate(self, text, params, image=None):
        generator=None

        if params.seed > 0:
            latents_device = "cpu" if self._pipeline.device.type == "mps" else self._pipeline.device
            generator = torch.Generator(latents_device).manual_seed(params.seed)


        if not params.sampler or params.sampler == generation_pb2.SAMPLER_DDPM:
            scheduler=self._plms
        elif params.sampler == generation_pb2.SAMPLER_K_LMS:
            scheduler=self._klms
        elif params.sampler == generation_pb2.SAMPLER_DDIM:
            scheduler=self._ddim
        else:
            raise NotImplementedError("Scheduler not implemented")

        scheduler_device = self._device
        if self._vramO > 1 and scheduler_device == "cuda": scheduler_device = "cpu"

        self._pipeline.scheduler = scheduler
        #self._pipeline.scheduler.to(scheduler_device)

        with self._autocast():
            images = self._pipeline(
                prompt=text,
                init_image=image,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg_scale, # TODO: read from sampler parameters
                generator=generator,
                return_dict=False
            )

        return images

class EngineManager(object):

    def __init__(self, engines, enable_mps=False, vram_optimisation_level=0):
        self.engines = engines
        self._default = None
        self._pipelines = {}
        self._activeId = None
        self._active = None

        self._vramO = vram_optimisation_level

        self._hasCuda = getattr(torch, 'cuda', False) and torch.cuda.is_available()
        self._hasMps = enable_mps and getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available()
        self._device = "cuda" if self._hasCuda else "mps" if self._hasMps else "cpu"

        self._token=os.environ.get("HF_API_TOKEN", True)

        self.loadPipelines()
    
    def _getWeightPath(self, remote_path, local_path):
        if os.path.isdir(os.path.normpath(local_path)): return local_path
        return remote_path

    def buildPipeline(self, engine):
        fp16 = engine.get('fp16', False) if self._device == "cuda" else False
        revision = "fp16" if fp16 else "main"
        dtype = torch.float16 if fp16 else None

        print(f"Using device {self._device}, revision {revision}, dtype {dtype}")

        if engine["class"] == "StableDiffusionPipeline":
            return PipelineWrapper(id=engine["id"], device=self._device, vramO=self._vramO, pipeline=StableDiffusionPipeline.from_pretrained(
                self._getWeightPath(engine["model"], engine["local_model"]), 
                revision=revision, 
                torch_dtype=dtype, 
                use_auth_token=self._token if engine.get("use_auth_token", False) else False
            ))
        elif engine["class"] == "UnifiedPipeline":
            return PipelineWrapper(id=engine["id"], device=self._device, vramO=self._vramO, pipeline=UnifiedPipeline.from_pretrained(
                self._getWeightPath(engine["model"], engine["local_model"]), 
                revision=revision, 
                torch_dtype=dtype, 
                use_auth_token=self._token if engine.get("use_auth_token", False) else False
            ))
    
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
            


