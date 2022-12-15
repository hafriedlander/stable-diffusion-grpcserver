import functools
import importlib
import inspect
import json
import math
import os
import tempfile
import traceback
from copy import deepcopy
from fnmatch import fnmatch
from functools import cache
from types import SimpleNamespace as SN
from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

import accelerate
import generation_pb2
import huggingface_hub
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    ModelMixin,
    PNDMScheduler,
    pipelines,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.utils import deprecate
from huggingface_hub.file_download import http_get
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from sdgrpcserver.k_diffusion import sampling as k_sampling
from sdgrpcserver.pipeline.kschedulers import *
from sdgrpcserver.pipeline.safety_checkers import FlagOnlySafetyChecker
from sdgrpcserver.pipeline.schedulers.sample_dpmpp_2m import sample_dpmpp_2m
from sdgrpcserver.pipeline.schedulers.scheduling_ddim import DDIMScheduler
from sdgrpcserver.pipeline.unified_pipeline import (
    SCHEDULER_NOISE_TYPE,
    UnifiedPipelineImageType,
    UnifiedPipelinePromptType,
)

DEFAULT_LIBRARIES = {
    "StableDiffusionPipeline": "stable_diffusion",
    "UnifiedPipeline": "sdgrpcserver.pipeline.unified_pipeline",
    "UpscalerPipeline": "sdgrpcserver.pipeline.upscaler_pipeline",
}

TYPE_CLASSES = {
    "vae": "diffusers.AutoencoderKL",
    "unet": "diffusers.UNet2DConditionModel",
    "inpaint_unet": "diffusers.UNet2DConditionModel",
    "clip_model": "transformers.CLIPModel",
    "feature_extractor": "transformers.CLIPFeatureExtractor",
    "tokenizer": "transformers.CLIPTokenizer",
    "clip_tokenizer": "transformers.CLIPTokenizer",
    "text_encoder": "transformers.CLIPTextModel",
    "inpaint_text_encoder": "transformers.CLIPTextModel",
    "upscaler": "sdgrpcserver.pipeline.upscaler_pipeline.NoiseLevelAndTextConditionedUpscaler",
}


class ProgressBarWrapper(object):
    class InternalTqdm(tqdm):
        def __init__(self, progress_callback, stop_event, suppress_output, iterable):
            self._progress_callback = progress_callback
            self._stop_event = stop_event
            super().__init__(iterable, disable=suppress_output)

        def update(self, n=1):
            displayed = super().update(n)
            if displayed and self._progress_callback:
                self._progress_callback(**self.format_dict)
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
        return ProgressBarWrapper.InternalTqdm(
            self._progress_callback, self._stop_event, self._suppress_output, iterable
        )


class EngineMode(object):
    def __init__(self, vram_optimisation_level=0, enable_cuda=True, enable_mps=False):
        self._vramO = vram_optimisation_level
        self._enable_cuda = enable_cuda
        self._enable_mps = enable_mps

    @property
    def device(self):
        self._hasCuda = (
            self._enable_cuda
            and getattr(torch, "cuda", False)
            and torch.cuda.is_available()
        )
        self._hasMps = (
            self._enable_mps
            and getattr(torch.backends, "mps", False)
            and torch.backends.mps.is_available()
        )
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
            if pixels <= self.points[0][0]:
                return self.points[0][1]

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
        torch.cuda.set_per_process_memory_fraction(1 - self.safety_margin)

        pipe = manager.getPipe()
        params = SN(
            height=512,
            width=512,
            cfg_scale=7.5,
            sampler=generation_pb2.SAMPLER_DDIM,
            eta=0,
            steps=8,
            strength=1,
            seed=-1,
        )

        l = 32  # Starting value - 512x512 fails inside PyTorch at 32, no amount of VRAM can help

        pixels = []
        batchmax = []

        for x in range(512, resmax, resstep):
            params.width = x
            print(f"Determining max batch for {x}")
            # Quick binary search
            r = l  # Start with the max from the previous run
            l = 1

            while l < r - 1:
                b = (l + r) // 2
                print(f"Trying {b}")
                try:
                    pipe.generate(["A Crocodile"] * b, params, suppress_output=True)
                except Exception as e:
                    r = b
                else:
                    l = b

            print(f"Max for {x} is {l}")

            pixels.append(params.width * params.height)
            batchmax.append(l)

            if l == 1:
                print(f"Max res is {x}x512")
                break

        self.points = list(zip(pixels, batchmax))
        print(
            "To save these for next time, use these for batch_points:",
            json.dumps(self.points),
        )

        torch.cuda.set_per_process_memory_fraction(1.0)


class PipelineWrapper(object):
    def __init__(self, id, mode, pipeline):
        self._id = id
        self._mode = mode

        self._pipeline = pipeline
        self._previous = None

        if self.mode.attention_slice:
            self._pipeline.enable_attention_slicing("auto")
            self._pipeline.enable_vae_slicing()
        else:
            self._pipeline.disable_attention_slicing()
            self._pipeline.disable_vae_slicing()

        if self.mode.cpu_offload:
            for key, value in self._pipeline.__dict__.items():
                if isinstance(value, torch.nn.Module):
                    accelerate.cpu_offload(value, self.mode.device)

        self.prediction_type = getattr(
            self._pipeline.scheduler, "prediction_type", "epsilon"
        )

        self._plms = self._prepScheduler(
            PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                skip_prk_steps=True,
                steps_offset=1,
            )
        )
        self._klms = self._prepScheduler(
            LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        )
        self._ddim = self._prepScheduler(
            DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type=self.prediction_type,
            )
        )
        self._euler = self._prepScheduler(
            EulerDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        )
        self._eulera = self._prepScheduler(
            EulerAncestralDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        )
        self._dpm2 = self._prepScheduler(
            DPM2DiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        )
        self._dpm2a = self._prepScheduler(
            DPM2AncestralDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        )
        self._heun = self._prepScheduler(
            HeunDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        )

        self._dpmspp1 = self._prepScheduler(
            DPMSolverMultistepScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                solver_order=1,
                prediction_type=self.prediction_type,
            )
        )
        self._dpmspp2 = self._prepScheduler(
            DPMSolverMultistepScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                solver_order=2,
                prediction_type=self.prediction_type,
            )
        )
        self._dpmspp3 = self._prepScheduler(
            DPMSolverMultistepScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                solver_order=3,
                prediction_type=self.prediction_type,
            )
        )

        # Common set of samplers that can handle epsilon or v_prediction unets
        self._samplers = {
            generation_pb2.SAMPLER_DDIM: self._ddim,
            generation_pb2.SAMPLER_DPMSOLVERPP_1ORDER: self._dpmspp1,
            generation_pb2.SAMPLER_DPMSOLVERPP_2ORDER: self._dpmspp2,
            generation_pb2.SAMPLER_DPMSOLVERPP_3ORDER: self._dpmspp3,
            generation_pb2.SAMPLER_K_LMS: k_sampling.sample_lms,  # self._klms
            generation_pb2.SAMPLER_K_EULER: k_sampling.sample_euler,  # self._euler
            generation_pb2.SAMPLER_K_EULER_ANCESTRAL: k_sampling.sample_euler_ancestral,  # self._eulera
            generation_pb2.SAMPLER_K_DPM_2: k_sampling.sample_dpm_2,  # self._dpm2
            generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL: k_sampling.sample_dpm_2_ancestral,  # self._dpm2a
            generation_pb2.SAMPLER_K_HEUN: k_sampling.sample_heun,  # self._heun
            generation_pb2.SAMPLER_DPM_FAST: k_sampling.sample_dpm_fast,
            generation_pb2.SAMPLER_DPM_ADAPTIVE: k_sampling.sample_dpm_adaptive,
            generation_pb2.SAMPLER_DPMSOLVERPP_2S_ANCESTRAL: k_sampling.sample_dpmpp_2s_ancestral,
            generation_pb2.SAMPLER_DPMSOLVERPP_SDE: k_sampling.sample_dpmpp_sde,
            generation_pb2.SAMPLER_DPMSOLVERPP_2M: functools.partial(
                sample_dpmpp_2m, warmup_lms=True, ddim_cutoff=0.1
            ),
        }

        # If we're not using a v_prediction unet, add in the samplers that can only handle epsilon too
        if self.prediction_type != "v_prediction":
            self._samplers = {
                **self._samplers,
                generation_pb2.SAMPLER_DDPM: self._plms,
            }

    def _prepScheduler(self, scheduler):
        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        return scheduler

    @property
    def id(self):
        return self._id

    @property
    def mode(self):
        return self._mode

    def activate(self):
        if self.mode.cpu_offload:
            return

        if self._previous is not None:
            raise Exception("Activate called without previous deactivate")

        self._previous = {}

        module_names, *_ = self._pipeline.extract_init_dict(dict(self._pipeline.config))
        for name in module_names.keys():
            module = getattr(self._pipeline, name)
            if isinstance(module, torch.nn.Module):
                self._previous[name] = module
                setattr(self._pipeline, name, deepcopy(module).to(self.mode.device))

    def deactivate(self):
        if self.mode.cpu_offload:
            return

        if self._previous is None:
            raise Exception("Deactivate called without previous activate")

        module_names, *_ = self._pipeline.extract_init_dict(dict(self._pipeline.config))
        for name in module_names.keys():
            module = self._previous.get(name, None)
            if module and isinstance(module, torch.nn.Module):
                setattr(self._pipeline, name, module)

        self._previous = None

        if self.mode.device == "cuda":
            torch.cuda.empty_cache()

    def get_samplers(self):
        return self._samplers

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
        clip_guidance_base: Optional[str] = None,
        # Sampler control
        sampler: generation_pb2.DiffusionSampler = None,
        scheduler=None,
        eta: Optional[float] = None,
        churn: Optional[float] = None,
        churn_tmin: Optional[float] = None,
        churn_tmax: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        karras_rho: Optional[float] = None,
        scheduler_noise_type: Optional[SCHEDULER_NOISE_TYPE] = "normal",
        num_inference_steps: int = 50,
        # Providing these changes from txt2img into either img2img (no mask) or inpaint (mask) mode
        init_image: Optional[UnifiedPipelineImageType] = None,
        mask_image: Optional[UnifiedPipelineImageType] = None,
        outmask_image: Optional[UnifiedPipelineImageType] = None,
        # The strength of the img2img or inpaint process, if init_image is provided
        strength: float = None,
        # Hires control
        hires_fix=None,
        hires_oos_fraction=None,
        # Tiling control
        tiling=False,
        # Debug control
        debug_latent_tags=None,
        debug_latent_prefix="",
        # Process control
        progress_callback=None,
        stop_event=None,
        suppress_output=False,
    ):
        generator = None

        generator_device = "cpu" if self.mode.device == "mps" else self.mode.device

        if isinstance(seed, Iterable):
            generator = [torch.Generator(generator_device).manual_seed(s) for s in seed]
        elif seed > 0:
            generator = torch.Generator(generator_device).manual_seed(seed)

        if scheduler is None:
            samplers = self.get_samplers()
            if sampler is None:
                scheduler = samplers.items()[0]
            else:
                scheduler = samplers.get(sampler, None)

        if not scheduler:
            raise NotImplementedError("Scheduler not implemented")

        self._pipeline.scheduler = scheduler
        self._pipeline.progress_bar = ProgressBarWrapper(
            progress_callback, stop_event, suppress_output
        )

        pipeline_args = dict(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            clip_guidance_scale=clip_guidance_scale,
            clip_guidance_base=clip_guidance_base,
            prediction_type=self.prediction_type,
            eta=eta,
            churn=churn,
            churn_tmin=churn_tmin,
            churn_tmax=churn_tmax,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            karras_rho=karras_rho,
            scheduler_noise_type=scheduler_noise_type,
            num_inference_steps=num_inference_steps,
            init_image=init_image,
            mask_image=mask_image,
            outmask_image=outmask_image,
            strength=strength,
            hires_fix=hires_fix,
            hires_oos_fraction=hires_oos_fraction,
            tiling=tiling,
            debug_latent_tags=debug_latent_tags,
            debug_latent_prefix=debug_latent_prefix,
            output_type="tensor",
            return_dict=False,
        )

        pipeline_keys = inspect.signature(self._pipeline).parameters.keys()
        self_params = inspect.signature(self.generate).parameters
        for k, v in list(pipeline_args.items()):
            if k not in pipeline_keys:
                if v != self_params[k].default:
                    print(
                        f"Warning: Pipeline doesn't understand argument {k} (set to {v}) - ignoring"
                    )
                del pipeline_args[k]

        images = self._pipeline(**pipeline_args)

        return images


def clone_model(model, share_parameters=True, share_buffers=True):
    """
    Copies a model so you get a different set of instances, but they share
    all their parameters and buffers
    """

    # Start by deep cloning the model
    clone = deepcopy(model)

    # If this isn't actually a model, return the deepcopy as is
    if not isinstance(model, torch.nn.Module):
        return clone

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


class ModelSet(SN):
    def update(self, other: dict | SN):
        if isinstance(other, dict):
            self.__dict__.update(other)
        else:  # isinstance(other, SN)
            self.__dict__.update(other.__dict__)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class EngineManager(object):
    def __init__(
        self,
        engines,
        weight_root="./weights",
        refresh_models=None,
        refresh_on_error=False,
        mode=EngineMode(),
        nsfw_behaviour="block",
        batchMode=BatchMode(),
    ):
        self.engines = engines
        self._default = None

        self._models = {}
        self._pipelines = {}

        self._activeId = None
        self._active = None

        self._weight_root = weight_root
        self._refresh_models = refresh_models
        self._refresh_on_error = refresh_on_error

        self._mode = mode
        self._batchMode = batchMode
        self._nsfw = nsfw_behaviour
        self._token = os.environ.get("HF_API_TOKEN", True)

    @property
    def mode(self):
        return self._mode

    @property
    def batchMode(self):
        return self._batchMode

    def _get_weight_path(
        self, spec: dict, force_recheck: bool = False, force_redownload: bool = False
    ) -> str:
        # TODO: Break this up, it's too long

        usefp16 = self.mode.fp16 and spec.get("has_fp16", True)

        local_path = spec.get("local_model_fp16" if usefp16 else "local_model", None)
        model_path = spec.get("model", None)
        subfolder = spec.get("subfolder", None)
        use_auth_token = self._token if spec.get("use_auth_token", False) else False
        urls = spec.get("urls", None)

        # Keep a list of the things we tried that failed, so we can report them all in one go later
        # in the case that we weren't able to load it any way at all
        failures = ["Loading model failed, because:"]

        if local_path:
            test_path = (
                local_path
                if os.path.isabs(local_path)
                else os.path.join(self._weight_root, local_path)
            )
            test_path = os.path.normpath(test_path)
            if os.path.isdir(test_path):
                return test_path
            else:
                failures.append(f"    - Local path '{test_path}' doesn't exist")
        else:
            failures.append(
                "    - No local path for "
                + ("fp16" if usefp16 else "fp32")
                + " model was provided"
            )

        if model_path:
            # We always download the file ourselves, rather than passing model path through to from_pretrained
            # This lets us control the behaviour if the model fails to download
            extra_kwargs = {}

            ignore_patterns = spec.get("ignore_patterns", [])
            if isinstance(ignore_patterns, str):
                ignore_patterns = [ignore_patterns]
            extra_kwargs["ignore_patterns"] = ignore_patterns + ["*.ckpt"]

            if subfolder:
                extra_kwargs["allow_patterns"] = [f"{subfolder}*"]
            if use_auth_token:
                extra_kwargs["use_auth_token"] = use_auth_token
            if usefp16:
                extra_kwargs["revision"] = "fp16"

            if force_redownload:
                try:
                    repo_info = next(
                        (
                            repo
                            for repo in huggingface_hub.scan_cache_dir().repos
                            if repo.repo_id == model_path
                        )
                    )
                    hashes = [revision.commit_hash for revision in repo_info.revisions]
                    huggingface_hub.scan_cache_dir().delete_revisions(*hashes).execute()
                except:
                    pass

            cache_path = None
            attempt_download = force_recheck or force_redownload

            try:
                # Try getting the cached path without connecting to internet
                cache_path = huggingface_hub.snapshot_download(
                    model_path, repo_type="model", local_files_only=True, **extra_kwargs
                )
            except FileNotFoundError:
                attempt_download = True
            except:
                failures.append(
                    "    - Couldn't query local HuggingFace cache. Error was:"
                )
                failures.append(traceback.format_exc())

            if self._refresh_models:
                attempt_download = attempt_download or any(
                    (
                        True
                        for pattern in self._refresh_models
                        if fnmatch(model_path, pattern)
                    )
                )

            if attempt_download:
                try:
                    cache_path = huggingface_hub.snapshot_download(
                        model_path, repo_type="model", **extra_kwargs
                    )
                except:
                    if cache_path:
                        print(
                            f"Couldn't refresh cache for {model_path}. Using existing cache."
                        )
                    else:
                        failures.append(
                            "    - Downloading from HuggingFace failed. Error was:"
                        )
                        failures.append(traceback.format_exc())

            if cache_path:
                return cache_path

        else:
            failures.append("    - No remote model name was provided")

        if urls:
            id = urls["id"]
            cache_path = os.path.join(sd_cache_home, id)
            os.makedirs(cache_path, exist_ok=True)

            try:
                for name, url in urls.items():
                    if name == "id":
                        continue
                    full_name = os.path.join(cache_path, name)
                    if os.path.exists(full_name):
                        continue

                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=cache_path, delete=False
                    ) as temp_file:
                        http_get(url, temp_file)
                        os.replace(temp_file.name, full_name)
            except:
                failures.append("    - Download failed. Error was:")
                failures.append(traceback.format_exc())
            else:
                return cache_path

        raise EnvironmentError("\n".join(failures))

    def _import_class(self, fqclass_name: str | tuple[str, str]):
        # You can pass in either a (dot seperated) string or a tuple of library, class
        if isinstance(fqclass_name, str):
            *library_name, class_name = fqclass_name.split(".")
            library_name = ".".join(library_name)
        else:
            library_name, class_name = fqclass_name

        if not library_name:
            library_name = DEFAULT_LIBRARIES.get(class_name, None)

        if not library_name:
            raise EnvironmentError(
                f"Don't know the library name for class {class_name}"
            )

        # Is `library_name` a submodule of diffusers.pipelines?
        is_pipeline_module = hasattr(pipelines, library_name)

        if is_pipeline_module:
            # If so, look it up from there
            pipeline_module = getattr(pipelines, library_name)
            class_obj = getattr(pipeline_module, class_name)
        else:
            # else we just import it from the library.
            library = importlib.import_module(library_name)
            class_obj = getattr(library, class_name, None)

            # Backwards compatibility - if config asks for transformers.CLIPImageProcessor
            # and we don't have it, use transformers.CLIPFeatureExtractor, that's the old name
            if not class_obj:
                if (
                    library_name == "transformers"
                    and class_name == "CLIPImageProcessor"
                ):
                    class_obj = getattr(library, "CLIPFeatureExtractor", None)

            if not class_obj:
                raise EnvironmentError(
                    f"Config attempts to import {library}.{class_name} that doesn't appear to exist"
                )

        return class_obj

    def _load_model_from_weights(
        self,
        weight_path: str,
        name: str,
        fqclass_name: str | tuple[str, str] | None = None,
    ):
        if fqclass_name is None:
            fqclass_name = TYPE_CLASSES.get(name, None)

        if fqclass_name is None:
            raise EnvironmentError(
                f"Type {name} does not specify a class, and there is no default set for it."
            )

        class_obj = self._import_class(fqclass_name)

        load_method_names = ["from_pretrained", "from_config"]
        load_candidates = [getattr(class_obj, name, None) for name in load_method_names]
        load_method = [m for m in load_candidates if m is not None][0]

        loading_kwargs = {}

        if self.mode.fp16 and issubclass(class_obj, torch.nn.Module):
            loading_kwargs["torch_dtype"] = torch.float16

        is_diffusers_model = issubclass(class_obj, ModelMixin)
        is_transformers_model = issubclass(class_obj, PreTrainedModel)

        if is_diffusers_model or is_transformers_model:
            loading_kwargs["low_cpu_mem_usage"] = True

        # check if the module is in a subdirectory
        sub_path = os.path.join(weight_path, name)
        if os.path.isdir(sub_path):
            weight_path = sub_path

        model = load_method(weight_path, **loading_kwargs)
        model._source = weight_path
        return model

    def _load_modelset_from_weights(self, weight_path, whitelist=None, blacklist=None):
        config_dict = DiffusionPipeline.load_config(weight_path)

        if isinstance(whitelist, str):
            whitelist = [whitelist]
        if whitelist:
            whitelist = set(whitelist)
        if isinstance(blacklist, str):
            blacklist = [blacklist]
        if blacklist:
            blacklist = set(blacklist)

        pipeline = {}

        class_items = [
            item for item in config_dict.items() if isinstance(item[1], list)
        ]

        for name, fqclass_name in class_items:
            if whitelist and name not in whitelist:
                continue
            if blacklist and name in blacklist:
                continue
            if fqclass_name[1] is None:
                pipeline[name] = None
                continue

            if name == "safety_checker":
                if self._nsfw == "flag":
                    fqclass_name = (
                        "sdgrpcserver.pipeline.safety_checkers.FlagOnlySafetyChecker"
                    )
                elif self._nsfw == "ignore":
                    pipeline[name] = None
                    continue

            pipeline[name] = self._load_model_from_weights(
                weight_path, name, fqclass_name
            )

        return ModelSet(**pipeline)

    def _load_from_weights(
        self, spec: dict, force_redownload: bool = False
    ) -> ModelSet:
        # Determine if this set of weights is a pipeline, a clip
        type = spec.get("type", "pipeline")

        weight_path = self._get_weight_path(spec, force_redownload=force_redownload)

        # A pipeline has a top-level json file that describes a set of models
        if type == "pipeline":
            models = self._load_modelset_from_weights(
                weight_path,
                whitelist=spec.get("whitelist"),
                blacklist=spec.get("blacklist"),
            )
        # `clip` type is a special case that loads the same weights into two different models
        elif type == "clip":
            models = {
                "clip_model": self._load_model_from_weights(weight_path, "clip_model"),
                "feature_extractor": self._load_model_from_weights(
                    weight_path, "feature_extractor"
                ),
            }
        # Otherwise load the individual model
        else:
            models = {
                type: self._load_model_from_weights(
                    weight_path, type, spec.get("class")
                )
            }

        return models if isinstance(models, ModelSet) else ModelSet(**models)

    def _load_from_weights_with_retry(self, spec: dict) -> ModelSet:
        try:
            return self._load_from_weights(spec)
        except Exception as e:
            print(e)
            pass

        return self._load_from_weights(spec, force_redownload=True)

    def _load_from_reference(self, modelid: str):
        modelid, *submodel = modelid.split("/")
        if submodel:
            if len(submodel) > 1:
                raise EnvironmentError(
                    f"Can't have multiple sub-model references ({modelid}/{'/'.join(submodel)})"
                )
            submodel = submodel[0]

        print(f"    - Model {modelid}...")

        # If we've previous loaded this model, just return the same model
        if modelid in self._models:
            model = self._models[modelid]

        else:
            # Otherwise find the specification that matches the model_id reference
            spec = [
                spec
                for spec in self.engines
                if spec.get("enabled", True)
                and "model_id" in spec
                and spec["model_id"] == modelid
            ]

            if not spec:
                raise EnvironmentError(f"Model {modelid} referenced does not exist")

            # And load it, storing in cache before continuing
            self._models[modelid] = model = self._load_model(spec[0])

        return getattr(model, submodel) if submodel else model

    def _load_model(self, spec):
        model = spec.get("model", None)

        # Call the correct subroutine based on source to build the model
        if isinstance(model, str) and model.startswith("@"):
            model = self._load_from_reference(model[1:])
        else:
            if self._refresh_on_error:
                model = self._load_from_weights_with_retry(spec)
            else:
                model = self._load_from_weights(spec)

        overrides = spec.get("overrides", None)

        if overrides:
            for name, override in overrides.items():
                if isinstance(override, str):
                    override = {"model": override}

                override_spec = {**override, "type": name}
                override_model = self._load_model(override_spec)

                if isinstance(override_model, SN):
                    model.__dict__.update(override_model.__dict__)
                else:
                    setattr(model, name, override_model)

        return model

    def _instantiate_pipeline(self, engine, model, extra_kwargs):
        fqclass_name = engine.get("class", "UnifiedPipeline")
        class_obj = self._import_class(fqclass_name)

        available = set(model.__dict__.keys())

        class_init_params = inspect.signature(class_obj.__init__).parameters
        expected = set(class_init_params.keys()) - set(["self"])

        required = set(
            [
                name
                for name, param in class_init_params.items()
                if param.default is inspect._empty and name != "self"
            ]
        )

        # optional = expected - required

        if required - available:
            raise EnvironmentError(
                "Model definition did not provide model(s) the pipeline requires. Missing: "
                + repr(required - available)
            )

        modules = {k: clone_model(model[k]) for k in expected & available}

        if False:
            # Debug print source of each model
            max_len = max([len(n) for n in modules.keys()])
            for n, m in modules.items():
                print(f"{n.rjust(max_len, ' ')} | {'None' if m is None else m._source}")

        modules = {**modules, **extra_kwargs}
        return class_obj(**modules)

    def _load_engine(self, engine):
        model = self._load_model(engine)
        pipeline = self._instantiate_pipeline(engine, model, {})

        pipeline_options = engine.get("options", False)

        if pipeline_options:
            try:
                pipeline.set_options(pipeline_options)
            except Exception:
                raise ValueError(
                    f"Engine {engine['id']} has options, but created pipeline rejected them"
                )

        return pipeline

    def loadPipelines(self):

        print("Loading engines...")

        for engine in self.engines:
            if not engine.get("enabled", True):
                continue

            # Models are loaded on demand (so we don't load models that aren't referenced)
            modelid = engine.get("model_id", None)
            if modelid is not None:
                continue

            engineid = engine["id"]
            if engine.get("default", False):
                self._default = engineid

            print(f"  - Engine {engineid}...")
            pipeline = self._load_engine(engine)

            self._pipelines[engineid] = PipelineWrapper(
                id=engineid, mode=self._mode, pipeline=pipeline
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

        if id is None:
            id = self._default

        # If we're already active, just return it
        if self._active and id == self._active.id:
            return self._active

        # Otherwise deactivate it
        if self._active:
            self._active.deactivate()
            # Explicitly mark as not active, in case there's an error later
            self._active = None

        self._active = self._pipelines[id]
        self._active.activate()

        return self._active
