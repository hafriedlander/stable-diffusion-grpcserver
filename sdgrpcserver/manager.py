
from functools import cache
import os, warnings, traceback, math, json, importlib, inspect, tempfile
from fnmatch import fnmatch
from copy import deepcopy
from types import SimpleNamespace as SN
from typing import Callable, List, Tuple, Optional, Union, Literal, Iterable

import torch
import accelerate
import huggingface_hub
from huggingface_hub.file_download import http_get

from tqdm.auto import tqdm

from transformers import PreTrainedModel, CLIPFeatureExtractor, CLIPModel

from diffusers import pipelines, ModelMixin, ConfigMixin, StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate, logging
from diffusers.pipeline_utils import DiffusionPipeline

import generation_pb2

from sdgrpcserver.pipeline.unified_pipeline import UnifiedPipeline, UnifiedPipelinePromptType, UnifiedPipelineImageType
from sdgrpcserver.pipeline.upscaler_pipeline import NoiseLevelAndTextConditionedUpscaler, UpscalerPipeline
from sdgrpcserver.pipeline.safety_checkers import FlagOnlySafetyChecker

from sdgrpcserver.pipeline.schedulers.scheduling_ddim import DDIMScheduler
from sdgrpcserver.pipeline.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from sdgrpcserver.pipeline.kschedulers import *


class ProgressBarWrapper(object):

    class InternalTqdm(tqdm):
        def __init__(self, progress_callback, stop_event, suppress_output, iterable):
            self._progress_callback = progress_callback
            self._stop_event = stop_event
            super().__init__(iterable, disable=suppress_output)

        def update(self, n=1):
            displayed = super().update(n)
            if displayed and self._progress_callback: self._progress_callback(**self.format_dict)
            return displayed

        def __iter__(self):
            for x in super().__iter__():
                if self._stop_event and self._stop_event.is_set(): 
                    self.set_description("ABORTED")
                    break
                yield x

    def __init__(self, progress_callback, stop_event, suppress_output=False):
        self._progress_callback = progress_callback
        self._stop_event = stop_event
        self._suppress_output = suppress_output

    def __call__(self, iterable):
        return ProgressBarWrapper.InternalTqdm(self._progress_callback, self._stop_event, self._suppress_output, iterable)
    

class EngineMode(object):
    def __init__(self, vram_optimisation_level=0, enable_cuda = True, enable_mps = False):
        self._vramO = vram_optimisation_level
        self._enable_cuda = enable_cuda
        self._enable_mps = enable_mps
    
    @property
    def device(self):
        self._hasCuda = self._enable_cuda and getattr(torch, 'cuda', False) and torch.cuda.is_available()
        self._hasMps = self._enable_mps and getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available()
        return "cuda" if self._hasCuda else "mps" if self._hasMps else "cpu"

    @property
    def attention_slice(self):
        return self.device == "cuda" and self._vramO > 0

    @property
    def fp16(self):
        return self.device == "cuda" and self._vramO > 1

    @property
    def cpu_offload(self):
        return True if self.device == "cuda" and self._vramO > 2 else False


class BatchMode:
    def __init__(self, autodetect=False, points=None, simplemax=1, safety_margin=0.2):
        self.autodetect = autodetect
        self.points = json.loads(points) if isinstance(points, str) else points
        self.simplemax = simplemax
        self.safety_margin = safety_margin
    
    def batchmax(self, pixels):
        if self.points:
            # If pixels less than first point, return that max
            if pixels <= self.points[0][0]: return self.points[0][1]

            # Linear interpolate between bracketing points
            pairs = zip(self.points[:-1], self.points[1:])
            for pair in pairs:
                if pixels >= pair[0][0] and pixels <= pair[1][0]:
                    i = (pixels - pair[0][0]) / (pair[1][0] - pair[0][0])
                    return math.floor(pair[0][1] + i * (pair[1][1] - pair[0][1]))

            # Off top of points - assume max of 1
            return 1
        
        if self.simplemax is not None:
            return self.simplemax

        return 1

    def run_autodetect(self, manager, resmax=2048, resstep=256):
        torch.cuda.set_per_process_memory_fraction(1-self.safety_margin)

        pipe = manager.getPipe()
        params = SN(height=512, width=512, cfg_scale=7.5, sampler=generation_pb2.SAMPLER_DDIM, eta=0, steps=8, strength=1, seed=-1)

        l = 32 # Starting value - 512x512 fails inside PyTorch at 32, no amount of VRAM can help

        pixels=[]
        batchmax=[]

        for x in range(512, resmax, resstep):
            params.width = x
            print(f"Determining max batch for {x}")
            # Quick binary search
            r = l # Start with the max from the previous run
            l = 1

            while l < r-1:
                b = (l+r)//2;
                print (f"Trying {b}")
                try:
                    pipe.generate(["A Crocodile"]*b, params, suppress_output=True)
                except Exception as e:
                    r = b
                else:
                    l = b
            
            print (f"Max for {x} is {l}")

            pixels.append(params.width * params.height)
            batchmax.append(l)

            if l == 1:
                print(f"Max res is {x}x512")
                break
            

        self.points=list(zip(pixels, batchmax))
        print("To save these for next time, use these for batch_points:", json.dumps(self.points))

        torch.cuda.set_per_process_memory_fraction(1.0)

class PipelineWrapper(object):

    def __init__(self, id, mode, pipeline):
        self._id = id
        self._mode = mode

        self._pipeline = pipeline
        self._previous = None

        self._pipeline.enable_attention_slicing(1 if self.mode.attention_slice else None)

        if self.mode.cpu_offload:
            for key, value in self._pipeline.__dict__.items():
                if isinstance(value, torch.nn.Module): accelerate.cpu_offload(value, self.mode.device)

        self._plms = self._prepScheduler(PNDMScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                skip_prk_steps=True
        ))
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
        self._euler = self._prepScheduler(EulerDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._eulera = self._prepScheduler(EulerAncestralDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._dpm2 = self._prepScheduler(DPM2DiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._dpm2a = self._prepScheduler(DPM2AncestralDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))
        self._heun = self._prepScheduler(HeunDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear",
                num_train_timesteps=1000
            ))

        self._dpmspp1 = self._prepScheduler(DPMSolverMultistepScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear", 
                num_train_timesteps=1000,
                solver_order=1
        ))
        self._dpmspp2 = self._prepScheduler(DPMSolverMultistepScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear", 
                num_train_timesteps=1000,
                solver_order=2
        ))
        self._dpmspp3 = self._prepScheduler(DPMSolverMultistepScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear", 
                num_train_timesteps=1000,
                solver_order=3
        ))

    def _prepScheduler(self, scheduler):
        if isinstance(scheduler, KSchedulerMixin):
            scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        return scheduler

    @property
    def id(self): return self._id

    @property
    def mode(self): return self._mode

    def activate(self):
        if self.mode.cpu_offload: return

        if self._previous is not None: 
            raise Exception("Activate called without previous deactivate")

        self._previous = {}

        module_names, _ = self._pipeline.extract_init_dict(dict(self._pipeline.config))
        for name in module_names.keys():
            module = getattr(self._pipeline, name)
            if isinstance(module, torch.nn.Module):
                self._previous[name] = module
                setattr(self._pipeline, name, deepcopy(module).to(self.mode.device))

    def deactivate(self):
        if self.mode.cpu_offload: return

        if self._previous is None: 
            raise Exception("Deactivate called without previous activate")

        module_names, _ = self._pipeline.extract_init_dict(dict(self._pipeline.config))
        for name in module_names.keys():
            module = self._previous.get(name, None)
            if module and isinstance(module, torch.nn.Module):
                setattr(self._pipeline, name, module)
                
        self._previous = None
        
        if self.mode.device == "cuda": torch.cuda.empty_cache()

    def generate(
        self,

        # The prompt, negative_prompt, and number of images per prompt
        prompt: UnifiedPipelinePromptType,
        negative_prompt: Optional[UnifiedPipelinePromptType] = None,
        num_images_per_prompt: Optional[int] = 1,

        # The seeds - len must match len(prompt) * num_images_per_prompt if provided
        seed: Optional[Union[int, Iterable[int]]] = None,

        # The size - ignored if an init_image is passed
        height: int = 512,
        width: int = 512,

        # Guidance control    
        guidance_scale: float = 7.5,
        clip_guidance_scale: Optional[float] = None,
 
        # Sampler control
        sampler: generation_pb2.DiffusionSampler = None,
        eta: Optional[float] = 0.0,
        num_inference_steps: int = 50,

        # Providing these changes from txt2img into either img2img (no mask) or inpaint (mask) mode
        init_image: Optional[UnifiedPipelineImageType] = None,
        mask_image: Optional[UnifiedPipelineImageType] = None,
        outmask_image: Optional[UnifiedPipelineImageType] = None,

        # The strength of the img2img or inpaint process, if init_image is provided
        strength: float = 0.0,

        # Process controll
        progress_callback=None, 
        stop_event=None, 
        suppress_output=False
    ):
        generator=None

        generator_device = "cpu" if self.mode.device == "mps" else self.mode.device

        if isinstance(seed, Iterable):
            generator = [torch.Generator(generator_device).manual_seed(s) for s in seed] 
        elif seed > 0:
            generator = torch.Generator(generator_device).manual_seed(seed)

        if sampler is None or sampler == generation_pb2.SAMPLER_DDPM:
            scheduler=self._plms
        elif sampler == generation_pb2.SAMPLER_K_LMS:
            scheduler=self._klms
        elif sampler == generation_pb2.SAMPLER_DDIM:
            scheduler=self._ddim
        elif sampler == generation_pb2.SAMPLER_K_EULER:
            scheduler=self._euler
        elif sampler == generation_pb2.SAMPLER_K_EULER_ANCESTRAL:
            scheduler=self._eulera
        elif sampler == generation_pb2.SAMPLER_K_DPM_2:
            scheduler=self._dpm2
        elif sampler == generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL:
            scheduler=self._dpm2a
        elif sampler == generation_pb2.SAMPLER_K_HEUN:
            scheduler=self._heun
        elif sampler == generation_pb2.SAMPLER_DPMSOLVERPP_1ORDER:
            scheduler=self._dpmspp1
        elif sampler == generation_pb2.SAMPLER_DPMSOLVERPP_2ORDER:
            scheduler=self._dpmspp2
        elif sampler == generation_pb2.SAMPLER_DPMSOLVERPP_3ORDER:
            scheduler=self._dpmspp3
        else:
            raise NotImplementedError("Scheduler not implemented")

        self._pipeline.scheduler = scheduler
        self._pipeline.progress_bar = ProgressBarWrapper(progress_callback, stop_event, suppress_output)

        images = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            clip_guidance_scale=clip_guidance_scale,
            eta=eta,
            num_inference_steps=num_inference_steps,
            init_image=init_image,
            mask_image=mask_image,
            outmask_image=outmask_image,
            strength=strength,
            output_type="tensor",
            return_dict=False
        )

        return images

def clone_model(model, share_parameters=True, share_buffers=True):
    """
    Copies a model so you get a different set of instances, but they share
    all their parameters and buffers
    """

    # Start by deep cloning the model
    clone = deepcopy(model)

    # If this isn't actually a model, return the deepcopy as is
    if not isinstance(model, torch.nn.Module): return clone

    for (_, source), (_, dest) in zip(model.named_modules(), clone.named_modules()):
        if share_parameters:
            for name, param in source.named_parameters(recurse=False):
                dest.register_parameter(name, param)
        if share_buffers:
            for name, buf in source.named_buffers(recurse=False):
                dest.register_buffer(name, buf)

    return clone

# TODO: Not here
default_home = os.path.join(os.path.expanduser("~"), ".cache")
sd_cache_home = os.path.expanduser(
    os.getenv(
        "SD_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "sdgrpcserver"),
    )
)


class EngineManager(object):

    def __init__(self, engines, weight_root="./weights", refresh_models=None, mode=EngineMode(), nsfw_behaviour="block", batchMode=BatchMode()):
        self.engines = engines
        self._default = None

        self._models = {}
        self._pipelines = {}

        self._activeId = None
        self._active = None

        self._weight_root = weight_root
        self._refresh_models = refresh_models

        self._mode = mode
        self._batchMode = batchMode
        self._nsfw = nsfw_behaviour
        self._token = os.environ.get("HF_API_TOKEN", True)

    @property
    def mode(self): return self._mode

    @property
    def batchMode(self): return self._batchMode

    def _getWeightPath(self, opts, force_recheck=False, force_redownload=False):
        # TODO: Break this up, it's too long

        usefp16 = self.mode.fp16 and opts.get("has_fp16", True)

        local_path = opts.get("local_model_fp16" if usefp16 else "local_model", None)
        model_path = opts.get("model", None)
        subfolder = opts.get("subfolder", None)
        use_auth_token=self._token if opts.get("use_auth_token", False) else False
        urls = opts.get("urls", None)

        # Keep a list of the things we tried that failed, so we can report them all in one go later
        # in the case that we weren't able to load it any way at all
        failures = ["Loading model failed, because:"]

        if local_path:
            test_path = local_path if os.path.isabs(local_path) else os.path.join(self._weight_root, local_path)
            test_path = os.path.normpath(test_path)
            if os.path.isdir(test_path): 
                return test_path
            else:
                failures.append(f"    - Local path '{test_path}' doesn't exist")
        else:
            failures.append("    - No local path for " + ("fp16" if usefp16 else "fp32") + " model was provided")
            

        if model_path:
            # We always download the file ourselves, rather than passing model path through to from_pretrained
            # This lets us control the behaviour if the model fails to download
            extra_kwargs={}
            if subfolder: extra_kwargs["allow_patterns"]=f"{subfolder}*"
            if use_auth_token: extra_kwargs["use_auth_token"]=use_auth_token

            if force_redownload:
                try:
                    repo_info = next((repo for repo in huggingface_hub.scan_cache_dir().repos if repo.repo_id==model_path))
                    hashes = [revision.commit_hash for revision in repo_info.revisions]
                    huggingface_hub.scan_cache_dir().delete_revisions(*hashes).execute()
                except:
                    pass

            cache_path = None
            attempt_download = force_recheck or force_redownload

            try:
                # Try getting the cached path without connecting to internet
                cache_path = huggingface_hub.snapshot_download(
                    model_path, 
                    repo_type="model", 
                    local_files_only=True,
                    **extra_kwargs
                )
            except FileNotFoundError:
                attempt_download = True
            except:
                failures.append("    - Couldn't query local HuggingFace cache. Error was:")
                failures.append(traceback.format_exc())

            if self._refresh_models:
                attempt_download = attempt_download or any((True for pattern in self._refresh_models if fnmatch(model_path, pattern)))

            if attempt_download:
                try:
                    cache_path = huggingface_hub.snapshot_download(
                        model_path, 
                        repo_type="model", 
                        **extra_kwargs
                    )
                except:
                    if cache_path: 
                        print(f"Couldn't refresh cache for {model_path}. Using existing cache.")
                    else:
                        failures.append("    - Downloading from HuggingFace failed. Error was:")
                        failures.append(traceback.format_exc())
            
            if cache_path: return cache_path

        else:
            failures.append("    - No remote model name was provided")

        if urls:
            id = urls["id"]
            cache_path = os.path.join(sd_cache_home, id)
            os.makedirs(cache_path, exist_ok=True)

            try:
                for name, url in urls.items():
                    if name == "id": continue
                    full_name = os.path.join(cache_path, name)
                    if os.path.exists(full_name): continue

                    with tempfile.NamedTemporaryFile(mode="wb", dir=cache_path, delete=False) as temp_file:
                        http_get(url, temp_file)
                        os.replace(temp_file.name, full_name)
            except:
                failures.append("    - Download failed. Error was:")
                failures.append(traceback.format_exc())
            else:
                return cache_path

        raise EnvironmentError("\n".join(failures))

    def _fromLoaded(self, klass, opts, extra_kwargs):
        parts = opts["model"][1:].split("/")
        local_model = self.loadModel(parts.pop(0))
        if parts: local_model = getattr(local_model, parts.pop(0))

        if isinstance(local_model, klass): 
            return clone_model(local_model)

        if isinstance(local_model, SN) and issubclass(klass, DiffusionPipeline):
            available = local_model.__dict__
            expected_modules = set(inspect.signature(klass.__init__).parameters.keys()) - set(["self"])
            modules = {k: clone_model(getattr(local_model, k)) for k in expected_modules if hasattr(local_model, k)}
            modules = {**modules, **extra_kwargs}
            return klass(**modules)
        
        raise ValueError(f"Error loading model - {local_model.__class__} is not an instance of {klass}")

    def _fromWeights(self, klass, opts, extra_kwargs, force_redownload=False):
        weight_path=self._getWeightPath(opts, force_redownload=force_redownload)
        if self.mode.fp16: extra_kwargs["torch_dtype"]=torch.float16
        if opts.get('subfolder', None): extra_kwargs['subfolder'] = opts.get('subfolder')

        constructor_keys = set(inspect.signature(klass.__init__).parameters.keys())
        accepts_safety_checker = "safety_checker" in constructor_keys
        accepts_inpaint_unet = "inpaint_unet" in constructor_keys
        accepts_clip_model = "clip_model" in constructor_keys

        if accepts_safety_checker:
            if self._nsfw == "flag":
                extra_kwargs["safety_checker"] = self.fromPretrained(FlagOnlySafetyChecker, {**opts, "subfolder": "safety_checker"})
            if self._nsfw == "ignore":
                extra_kwargs["safety_checker"] = None
        if accepts_inpaint_unet and "inpaint_unet" not in extra_kwargs: 
            extra_kwargs["inpaint_unet"] = None
        if accepts_clip_model and "clip_model" not in extra_kwargs: 
            extra_kwargs["clip_model"] = None

        # Supress warnings during pipeline load. Unfortunately there isn't a better 
        # way to override the behaviour (beyond duplicating a huge function)
        current_log_level = logging.get_verbosity()
        logging.set_verbosity(logging.ERROR)

        result = klass.from_pretrained(weight_path, **extra_kwargs)

        logging.set_verbosity(current_log_level)

        return result

    def _weightRetry(self, callback, *args, **kwargs):
        try: return callback(*args, **kwargs)
        except: pass

        return callback(*args, **kwargs, force_redownload=True)

    def fromPretrained(self, klass, opts, extra_kwargs = None):
        if extra_kwargs is None: extra_kwargs = {}

        model = opts.get("model", None)
        is_local = model and len(model) > 1 and model[0] == "@"

        # Handle copying a local model
        if is_local: return self._fromLoaded(klass, opts, extra_kwargs)
        else: return self._weightRetry(self._fromWeights, klass, opts, extra_kwargs)

    def _nakedFromLoaded(self, opts):
        local_model = self.loadModel(opts["model"][1:])
        if not isinstance(local_model, SN): raise ValueError(f"{model} is not a pipeline, it's a {local_model}")
        return local_model

    def _nakedFromWeights(self, opts, force_redownload=False):
        weight_path = self._getWeightPath(opts, force_redownload=force_redownload)
        config_dict = DiffusionPipeline.get_config_dict(weight_path)

        whitelist = opts.get("whitelist", None)
        if isinstance(whitelist, str): whitelist = [whitelist]
        if whitelist: whitelist = set(whitelist)
        blacklist = opts.get("blacklist", None)
        if isinstance(blacklist, str): blacklist = [blacklist]
        if blacklist: blacklist = set(blacklist)

        pipeline = {}
        for name, (library_name, class_name) in [item for item in config_dict.items() if item[0][0] != "_"]:
            if whitelist and name not in whitelist: continue
            if blacklist and name in blacklist: continue

            if name == "safety_checker":
                if self._nsfw == 'flag':
                    library_name = 'sdgrpcserver.pipeline.safety_checkers'
                    class_name = 'FlagOnlySafetyChecker'
                elif self._nsfw == "ignore":
                    pipeline[name] = None
                    continue

            # This is mostly from DiffusersPipeline.from_preloaded. Why is that method _so long_?
            is_pipeline_module = hasattr(pipelines, library_name)

            if is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)

            load_method_names = ['from_pretrained', 'from_config']
            load_candidates = [getattr(class_obj, name, None) for name in load_method_names]
            load_method = [m for m in load_candidates if m is not None][0]
            
            loading_kwargs = {}

            if self.mode.fp16 and issubclass(class_obj, torch.nn.Module):
                loading_kwargs["torch_dtype"]=torch.float16

            is_diffusers_model = issubclass(class_obj, ModelMixin)
            is_transformers_model = issubclass(class_obj, PreTrainedModel)

            if is_diffusers_model or is_transformers_model:
                loading_kwargs["low_cpu_mem_usage"] = True

            # check if the module is in a subdirectory
            if os.path.isdir(os.path.join(weight_path, name)):
                loaded_sub_model = load_method(os.path.join(weight_path, name), **loading_kwargs)
            else:
                # else load from the root directory
                loaded_sub_model = load_method(weight_path, **loading_kwargs)

            pipeline[name] = loaded_sub_model

        return SN(**pipeline)

    def buildModel(self, opts, name = None):
        if name is None: name = opts["type"]

        if name == "vae":
            return self.fromPretrained(AutoencoderKL, opts)
        elif name == "unet" or name == "inpaint_unet":
            return self.fromPretrained(UNet2DConditionModel, opts)
        elif name == "clip_model":
            return self.fromPretrained(CLIPModel, opts)
        elif name == "feature_extractor":
            return self.fromPretrained(CLIPFeatureExtractor, opts)
        elif name == "upscaler":
            return self.fromPretrained(NoiseLevelAndTextConditionedUpscaler, opts)
        else:
            raise ValueError(f"Unknown model {name}")   

    def buildNakedPipeline(self, opts, extras = None):
        model = opts.get("model", None)
        is_local = model and len(model) > 1 and model[0] == "@"

        if is_local:
            pipeline = self._nakedFromLoaded(opts)
        else:
            pipeline = self._weightRetry(self._nakedFromWeights, opts)
        
        if extras:
            args = {**pipeline.__dict__, **extras}
            return SN(**args)
        else:
            return pipeline

    def buildPipeline(self, engine, naked = False):
        extra_kwargs={}

        for name, opts in engine.get("overrides", {}).items():
            if isinstance(opts, str): opts = { "model": opts }

            if name == "clip":
                extra_kwargs["clip_model"] = self.buildModel({**opts, "model": opts["model"] + "/clip_model"}, "clip_model")
                extra_kwargs["feature_extractor"] = self.buildModel({**opts, "model": opts["model"] + "/feature_extractor"}, "feature_extractor")
            else:
                extra_kwargs[name] = self.buildModel(opts, name)

        if naked:
            return self.buildNakedPipeline(engine, extra_kwargs)
        
        else:
            pipeline = None

            klass = engine.get("class", "UnifiedPipeline")

            if klass == "StableDiffusionPipeline":
                pipeline = self.fromPretrained(StableDiffusionPipeline, engine, extra_kwargs)

            elif klass == "UnifiedPipeline":
                pipeline = self.fromPretrained(UnifiedPipeline, engine, extra_kwargs)

            elif klass == "UpscalerPipeline":
                pipeline = self.fromPretrained(UpscalerPipeline, engine, extra_kwargs)

            else:
                raise Exception(f'Unknown engine class "{klass}"')

            if engine.get("options", False):
                try:
                    pipeline.set_options(engine.get("options"))
                except:
                    raise ValueError(f"Engine {engine['id']} has options, but created pipeline rejected them")
            
            return pipeline

    def loadModel(self, modelid):
        if modelid in self._models:
            return self._models[modelid]

        for engine in self.engines:
            if not engine.get("enabled", True): continue

            otherid = engine.get("model_id", None)
            if otherid is not None and otherid == modelid:
                print(f"    - Model {modelid}...")

                type = engine.get("type", "pipeline")
                if type == "pipeline": 
                    self._models[modelid] = self.buildPipeline(engine, naked=True)
                elif type == "clip":
                    self._models[modelid] = SN(
                        clip_model = self.buildModel(engine, "clip_model"),
                        feature_extractor = self.buildModel(engine, "feature_extractor"),
                    )
                else:
                    self._models[modelid] = self.buildModel(engine)

                return self._models[modelid]

        raise EnvironmentError(f"Model {modelid} referenced does not exist")

    def loadPipelines(self):

        print("Loading engines...")

        for engine in self.engines:
            if not engine.get("enabled", True): continue

            # Models are loaded on demand (so we don't load models that aren't referenced)
            modelid = engine.get("model_id", None)
            if modelid is not None: continue

            engineid = engine["id"]
            if engine.get("default", False): self._default = engineid

            print(f"  - Engine {engineid}...")
            pipeline = self.buildPipeline(engine)

            self._pipelines[engineid] = PipelineWrapper(
                id=engineid,
                mode=self._mode,
                pipeline=pipeline
            )

        if self.batchMode.autodetect:
            self.batchMode.run_autodetect(self)

    def getStatus(self):
        return {
            engine["id"]: engine["id"] in self._pipelines 
            for engine in self.engines 
            if engine.get("id", False) and engine.get("enabled", False)
        }

    def getPipe(self, id=None):
        """
        Get and activate a pipeline
        TODO: Better activate / deactivate logic. Right now we just keep a max of one pipeline active.
        """

        if id is None: id = self._default

        # If we're already active, just return it
        if self._active and id == self._active.id: return self._active

        # Otherwise deactivate it
        if self._active: self._active.deactivate()

        self._active = self._pipelines[id]
        self._active.activate()

        return self._active
            


