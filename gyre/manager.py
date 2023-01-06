import gc
import glob
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
import queue
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from queue import Queue
from types import SimpleNamespace as SN
from typing import Iterable, Optional, Union
from urllib.parse import urlparse

import generation_pb2
import huggingface_hub
import torch
from diffusers import ModelMixin, UNet2DConditionModel, pipelines
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline, is_safetensors_compatible
from diffusers.utils import deprecate
from huggingface_hub.file_download import http_get
from tqdm.auto import tqdm
from transformers import CLIPModel, PreTrainedModel

from gyre.constants import sd_cache_home
from gyre.pipeline.model_utils import GPUExclusionSet, clone_model
from gyre.pipeline.samplers import build_sampler_set
from gyre.pipeline.unified_pipeline import (
    SCHEDULER_NOISE_TYPE,
    UnifiedPipelineImageType,
    UnifiedPipelinePromptType,
)

DEFAULT_LIBRARIES = {
    "StableDiffusionPipeline": "stable_diffusion",
    "UnifiedPipeline": "gyre.pipeline.unified_pipeline",
    "UpscalerPipeline": "gyre.pipeline.upscaler_pipeline",
    "DiffusersDepthPipeline": "gyre.pipeline.depth.diffusers_depth_pipeline",
    "MidasDepthPipeline": "gyre.pipeline.depth.midas_depth_pipeline",
    "MidasModelWrapper": "gyre.pipeline.depth.midas_model_wrapper",
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
    "upscaler": "gyre.pipeline.upscaler_pipeline.NoiseLevelAndTextConditionedUpscaler",
    "depth_estimator": "transformers.DPTForDepthEstimation",
    "midas_depth_estimator": "MidasModelWrapper",
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
    def unet_exclusion(self):
        return self.device == "cuda" and self._vramO > 2

    @property
    def allexceptclip_exclusion(self):
        return self.device == "cuda" and self._vramO > 3

    @property
    def all_exclusion(self):
        return self.device == "cuda" and self._vramO > 4


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
                    with manager.with_engine() as pipe:
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


class PipelineWrapper:
    def __init__(self, id, mode, pipeline):
        self._id = id
        self._mode = mode

        self._pipeline = pipeline
        self._previous = None

    @property
    def id(self):
        return self._id

    @property
    def mode(self):
        return self._mode

    def pipeline_modules(self):
        pipeline_module_helper = getattr(self._pipeline, "pipeline_modules", None)

        if pipeline_module_helper:
            for name, module in pipeline_module_helper():
                yield name, module

        else:
            module_names, *_ = self._pipeline.extract_init_dict(
                dict(self._pipeline.config)
            )
            for name in module_names.keys():
                module = getattr(self._pipeline, name)
                if isinstance(module, torch.nn.Module):
                    yield name, module

    def _delay(self, name, module):
        return False

    def activate(self, device):
        if self._previous is not None:
            raise Exception("Activate called without previous deactivate")

        self._previous = {}

        exclusion_set = GPUExclusionSet(1)

        for name, module in self.pipeline_modules():
            self._previous[name] = module

            # Clone from CPU to either CUDA or Meta with a hook to move to CUDA
            cloned = clone_model(
                module,
                device,
                exclusion_set=exclusion_set if self._delay(name, module) else None,
            )

            # And set it on the pipeline
            setattr(self._pipeline, name, cloned)

    def deactivate(self):
        if self._previous is None:
            raise Exception("Deactivate called without previous activate")

        for name, module in self.pipeline_modules():
            setattr(self._pipeline, name, self._previous.get(name))

        self._previous = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        return self._pipeline(*args, **kwargs)


class GeneratePipelineWrapper(PipelineWrapper):
    def __init__(self, id, mode, pipeline):
        super().__init__(id, mode, pipeline)

        if self.mode.attention_slice:
            self._pipeline.enable_attention_slicing("auto")
            self._pipeline.enable_vae_slicing()
        else:
            self._pipeline.disable_attention_slicing()
            self._pipeline.disable_vae_slicing()

        self.prediction_type = getattr(
            self._pipeline.scheduler, "prediction_type", "epsilon"
        )

        self._samplers = build_sampler_set(
            self._pipeline.scheduler.config,
            include_diffusers=True,
            include_kdiffusion=True,
        )

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

    def _delay(self, name, module):
        # Should we delay moving this to CUDA until forward is called?
        if self.mode.all_exclusion:
            return True
        elif self.mode.allexceptclip_exclusion:
            if not isinstance(module, CLIPModel):
                return True
        elif self.mode.unet_exclusion:
            if isinstance(module, UNet2DConditionModel):
                return True

        return False

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
        depth_map: Optional[UnifiedPipelineImageType] = None,
        # The strength of the img2img or inpaint process, if init_image is provided
        strength: float = None,
        # Lora
        lora=None,
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
                scheduler = list(samplers.values())[0]
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
            depth_map=depth_map,
            strength=strength,
            lora=lora,
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


class ModelSet(SN):
    def update(self, other: dict | SN):
        if isinstance(other, dict):
            self.__dict__.update(other)
        else:  # isinstance(other, SN)
            self.__dict__.update(other.__dict__)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class DeviceQueueSlot:
    device: torch.device
    pipeline: DiffusionPipeline | None = None


class EngineNotFoundError(Exception):
    pass


class EngineNotReadyError(Exception):
    pass


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
        ram_monitor=None,
    ):
        self.engines = engines
        self._defaults = {}

        # Models that are explictly loaded with a model_id and can be referenced
        self._models = {}
        # Models for each engine
        self._engine_models = {}

        self._activeId = None
        self._active = None

        self._weight_root = weight_root
        self._refresh_models = refresh_models
        self._refresh_on_error = refresh_on_error

        self._mode = mode
        self._batchMode = batchMode
        self._nsfw = nsfw_behaviour
        self._token = os.environ.get("HF_API_TOKEN", True)

        self._ram_monitor = ram_monitor

        self._device_queue = Queue()
        self._available_pipelines: dict[str, Queue] = {}

        for i in range(torch.cuda.device_count()):
            self._device_queue.put(DeviceQueueSlot(device=torch.device("cuda", i)))

    @property
    def mode(self):
        return self._mode

    @property
    def batchMode(self):
        return self._batchMode

    def _get_local_path(self, spec, fp16=False):
        key = "local_model_fp16" if fp16 else "local_model"
        path = spec.get(key)
        # Throw error if no such key in spec
        if not path:
            raise ValueError(f"No local model field `{key}` was provided")
        # Add path to weight root if not absolute
        if not os.path.isabs(path):
            path = os.path.join(self._weight_root, path)
        # Normalise
        path = os.path.normpath(path)
        # Throw error if result isn't a directory
        if not os.path.isdir(path):
            raise ValueError(f"Path '{path}' isn't a directory")

        return path

    def _get_hf_path(self, spec, local_only=True):
        extra_kwargs = {}

        model_path = spec.get("model")

        # If no model_path is provided, don't try and download
        if not model_path:
            raise ValueError("No remote model name was provided")

        fp16 = self.mode.fp16 and spec.get("has_fp16", True)
        subfolder = spec.get("subfolder", None)
        use_auth_token = self._token if spec.get("use_auth_token", False) else False

        if use_auth_token:
            extra_kwargs["use_auth_token"] = use_auth_token
        if fp16:
            extra_kwargs["revision"] = "fp16"

        try:
            # If we're not loading from local_only, do some extra logic to avoid downloading
            # other unusused large files in the repo unnessecarily (like .ckpt files and
            # the .safetensors version of .ckpt files )
            if not local_only:
                # Get a list of files, split into path and extension
                repo_info = huggingface_hub.model_info(model_path, **extra_kwargs)
                repo_files = [os.path.splitext(f.rfilename) for f in repo_info.siblings]
                # Sort by extension (grouping fails if not correctly sorted)
                repo_files.sort(key=lambda x: x[1])
                # Turn into a dictionary of { extension: set_of_files }
                grouped = {
                    k: {f[0] for f in v}
                    for k, v in itertools.groupby(repo_files, lambda x: x[1])
                }

                ignore_patterns = spec.get("ignore_patterns", [])
                if isinstance(ignore_patterns, str):
                    ignore_patterns = [ignore_patterns]

                has_ckpt = ".ckpt" in grouped
                has_bin = ".bin" in grouped
                has_safe = ".safetensors" in grouped
                if has_safe and has_ckpt:
                    # If we have ckpt and safetensors files, don't consider matching safetensors
                    has_safe = bool(grouped[".safetensors"] - grouped[".ckpt"])

                    # Exclude safetensors that match ckpt files
                    ignore_patterns += [
                        f"{f}.safetensors"
                        for f in grouped[".safetensors"] & grouped[".ckpt"]
                    ]

                if not has_bin and not has_safe:
                    if has_ckpt:
                        raise EnvironmentError(
                            "Repo {model_path} only contains .ckpt files. We can only load Diffusers structured models."
                        )
                    else:
                        raise EnvironmentError(
                            "Repo {model_path} doesn't appear to contain any model files."
                        )

                # Only use safetensors if either we have safetensor equivalents for every bin
                safe_only = has_bin and is_safetensors_compatible(repo_info)
                # Or if the spec explicitly says to
                safe_only = safe_only or spec.get("safe_only", False)

                if safe_only:
                    ignore_patterns += ["*.bin"]

                # Always exclude .ckpt and .pt files
                ignore_patterns += ["*.ckpt", "*.pt"]

                if ignore_patterns:
                    extra_kwargs["ignore_patterns"] = ignore_patterns
                if subfolder:
                    extra_kwargs["allow_patterns"] = [f"{subfolder}*"]

            return huggingface_hub.snapshot_download(
                model_path,
                repo_type="model",
                local_files_only=local_only,
                **extra_kwargs,
            )
        except Exception as e:
            if local_only:
                raise ValueError("Couldn't query local HuggingFace cache." + str(e))
            else:
                raise ValueError("Downloading from HuggingFace failed." + str(e))

    def _get_hf_forced_path(self, spec):
        model_path = spec.get("model")

        # If no model_path is provided, don't try and download
        if not model_path:
            raise ValueError("No remote model name was provided")

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

        return self._get_hf_path(spec, local_only=False)

    def _get_url_path(self, spec, local_only=True):
        urls = spec.get("urls")

        if not urls:
            raise ValueError("No URL was provided")

        if isinstance(urls, str):
            id = hashlib.sha1(urls.encode("utf-8")).hexdigest()
            _, filename = os.path.split(urlparse(urls).path)
            urls = {filename: urls}
        else:
            id = urls["id"]
            urls = {k: v for k, v in urls.items() if k != "id"}

        cache_path = os.path.join(sd_cache_home, id)
        temp_path = os.path.join(sd_cache_home, "temp")

        if os.path.isdir(cache_path):
            exists = {
                name: os.path.isfile(os.path.join(cache_path, name))
                for name in urls.keys()
            }
            if all(exists.values()):
                return cache_path
            elif local_only:
                raise ValueError(
                    f"Items missing from cache: {[name for name, exist in exists.items() if not exist]}"
                )
        elif local_only:
            raise ValueError("No local cache for URL")

        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)

        for name, url in urls.items():
            full_name = os.path.join(cache_path, name)
            if os.path.exists(full_name):
                continue

            temp_name = None
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=temp_path, delete=False
            ) as temp_file:
                http_get(url, temp_file)
                temp_name = temp_file.name

            if temp_name:
                os.replace(temp_name, full_name)

        return cache_path

    def _get_weight_path_candidates(self, spec: dict):
        candidates = []

        def add_candidate(callable, *args, **kwargs):
            candidates.append((callable, args, kwargs))

        model_path = spec.get("model")
        matches_refresh = (
            self._refresh_models
            and model_path
            and any(
                (
                    True
                    for pattern in self._refresh_models
                    if fnmatch(model_path, pattern)
                )
            )
        )

        # 1st: If this model should explicitly be refreshed, try refreshing from...
        if matches_refresh:
            # HuggingFace
            add_candidate(self._get_hf_path, local_only=False)
            # Or an explicit URL
            add_candidate(self._get_url_path, local_only=False)
        # 2nd: If we're in fp16 mode, try loading the fp16-specific local model
        if self.mode.fp16:
            add_candidate(self._get_local_path, fp16=True)
        # 3rd: Try loading the general local model
        add_candidate(self._get_local_path, fp16=False)
        # 4th: Try loading from the existing HuggingFace cache
        add_candidate(self._get_hf_path, local_only=True)
        # 5th: Try loading from an already-downloaded explicit URL
        add_candidate(self._get_url_path, local_only=True)
        # 6th: If this model wasn't explicitly flagged to be refreshed, try anyway
        if not matches_refresh:
            add_candidate(self._get_hf_path, local_only=False)
            add_candidate(self._get_url_path, local_only=False)
        # 7th: If configured so, try a forced empty-cache-and-reload from HuggingFace
        if self._refresh_on_error:
            add_candidate(self._get_hf_forced_path)

        return candidates

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
        config_dict = DiffusionPipeline.load_config(weight_path, local_files_only=True)

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
                        "gyre.pipeline.safety_checkers.FlagOnlySafetyChecker"
                    )
                elif self._nsfw == "ignore":
                    pipeline[name] = None
                    continue

            pipeline[name] = self._load_model_from_weights(
                weight_path, name, fqclass_name
            )

        return ModelSet(**pipeline)

    def _load_from_weights(self, spec: dict, weight_path: str) -> ModelSet:
        # Determine if this set of weights is a pipeline, a clip
        type = spec.get("type", "pipeline")

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

    def _load_from_weight_candidates(self, spec: dict) -> tuple[ModelSet, str]:
        candidates = self._get_weight_path_candidates(spec)

        failures = []

        for callback, args, kwargs in candidates:
            weight_path = None
            try:
                weight_path = callback(spec, *args, **kwargs)
                models = self._load_from_weights(spec, weight_path)
                return models, weight_path
            except ValueError as e:
                if str(e) not in failures:
                    failures.append(str(e))
            except Exception as e:
                if weight_path:
                    errstr = (
                        f"Error when trying to load weights from {weight_path}. "
                        + str(e)
                    )
                    if errstr not in failures:
                        failures.append(errstr)
                else:
                    raise e

        if "id" in spec:
            name = f"engine {spec['id']}"
        else:
            name = f"model {spec['model_id']}"

        raise EnvironmentError(
            "\n  - ".join([f"Failed to load {name}. Failed attempts:"] + failures)
        )

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
        elif isinstance(model, str) and model.startswith("!empty"):
            model = ModelSet()
        else:
            model, _ = self._load_from_weight_candidates(spec)

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

    def _build_pipeline_for_engine(self, spec):
        model = self._engine_models.get(spec["id"])
        if not model:
            raise EngineNotReadyError("Not ready yet")

        pipeline = self._instantiate_pipeline(spec, model, {})

        pipeline_options = spec.get("options", False)

        if pipeline_options:
            try:
                pipeline.set_options(pipeline_options)
            except Exception:
                raise ValueError(
                    f"Engine {spec['id']} has options, but created pipeline rejected them"
                )

        task = spec.get("task", "generate")
        if task == "generate":
            wrap_class = GeneratePipelineWrapper
        else:
            wrap_class = PipelineWrapper

        return wrap_class(id=spec["id"], mode=self._mode, pipeline=pipeline)

    def loadPipelines(self):

        print("Loading engines...")

        for engine in self.engines:
            if not engine.get("enabled", True):
                continue

            # If this isn't an engine (but a model, or a depth extractor, skip)
            if "id" not in engine:
                continue

            engineid = engine["id"]
            if engine.get("default", False):
                self._defaults[engine.get("task", "generate")] = engineid

            print(f"  - Engine {engineid}...")
            self._engine_models[engineid] = self._load_model(engine)

        if self.batchMode.autodetect:
            self.batchMode.run_autodetect(self)

    def _fixcfg(self, model, key, test, value):
        if hasattr(model.config, key) and test(getattr(model.config, key)):
            print("Fixing", model._source)
            new_config = dict(model.config)
            new_config[key] = value
            model._internal_dict = FrozenDict(new_config)

    def _save_model_as_safetensor(self, spec):
        # What's the local model attribute in the spec?
        local_model_attr = "local_model_fp16" if self.mode.fp16 else "local_model"

        _id = spec.get("model_id", spec.get("id"))
        type = spec.get("type", "pipeline")
        outpath = spec.get(local_model_attr, None)

        if not outpath:
            raise EnvironmentError(
                f"Can't save safetensor for {type} {_id} if {local_model_attr} not set"
            )

        if not os.path.isabs(outpath):
            outpath = os.path.join(self._weight_root, outpath)

        print(f"Saving {type} {_id} to {outpath}")

        # Load the model
        models, inpath = self._load_from_weight_candidates(spec)

        if type == "pipeline":
            for name, model in models.items():
                if not model:
                    continue

                # Fix model issues before saving
                if name == "scheduler":
                    self._fixcfg(model, "steps_offset", lambda x: x != 1, 1)
                elif name == "unet":
                    self._fixcfg(model, "sample_size", lambda x: x < 64, 64)

                subpath = os.path.join(outpath, name)
                print(f"  Submodule {name} to {subpath}")
                model.save_pretrained(save_directory=subpath, safe_serialization=True)

            if not os.path.samefile(inpath, outpath):
                shutil.copyfile(
                    os.path.join(inpath, "model_index.json"),
                    os.path.join(outpath, "model_index.json"),
                )
        elif type == "clip":
            models.clip_model.save_pretrained(
                save_directory=outpath, safe_serialization=True
            )
            if not os.path.samefile(inpath, outpath):
                for cfg_file in glob.glob(os.path.join(inpath, "*.json")):
                    shutil.copy(cfg_file, outpath)
        else:
            model = list(models.values())[0]
            model.save_pretrained(save_directory=outpath, safe_serialization=True)

    def _find_specs(
        self,
        id: str | Iterable[str] | None = None,
        model_id: str | Iterable[str] | None = None,
    ):
        if id and model_id:
            raise ValueError("Must provide only one of id or model_id")
        if not id and not model_id:
            raise ValueError("Must provide one of id or model_id")

        key = "id" if id else "model_id"
        val = id if id else model_id
        assert val
        val = (val,) if isinstance(val, str) else val

        return (
            spec
            for spec in self.engines
            if key in spec
            and any((True for pattern in val if fnmatch(spec.get(key), pattern)))
        )

    def _find_spec(
        self,
        id: str | Iterable[str] | None = None,
        model_id: str | Iterable[str] | None = None,
    ):
        res = self._find_specs(id=id, model_id=model_id)
        return next(res, None)

    def save_models_as_safetensor(self, patterns):
        specs = self._find_specs(model_id=patterns)

        for spec in specs:
            self._save_model_as_safetensor(spec)

        print("Done")

    def _find_referenced_weightspecs(self, spec):
        referenced = []
        model = spec.get("model")

        if model and model[0] == "@":
            model_id, *_ = model[1:].split("/")
            model_spec = self._find_spec(model_id=model_id)
            referenced += self._find_referenced_weightspecs(model_spec)
        else:
            referenced.append(spec)

        overrides = spec.get("overrides")

        if overrides:
            for _, override in overrides.items():
                if isinstance(override, str):
                    override = {"model": override}
                referenced += self._find_referenced_weightspecs(override)

        return referenced

    def save_engine_as_safetensor(self, patterns):
        specs = self._find_specs(id=patterns)

        involved = []

        for spec in specs:
            involved += self._find_referenced_weightspecs(spec)

        unique = {
            f"e/{spec['id']}" if "id" in spec else f"m/{spec['model_id']}": spec
            for spec in involved
        }

        for spec in unique.values():
            self._save_model_as_safetensor(spec)

        print("Done")

    def getStatus(self):
        return {
            engine["id"]: engine["id"] in self._engine_models
            for engine in self.engines
            if engine.get("id", False) and engine.get("enabled", False)
        }

    def _return_pipeline_to_pool(self, slot):
        assert slot.pipeline, "No pipeline to return to pool"

        # Get the current slot pipeline
        pipeline = slot.pipeline

        # Deactivate and remove it from the slot
        slot.pipeline.deactivate()
        slot.pipeline = None

        # Return it to the pool (creating a pool if needed)
        pool = self._available_pipelines.setdefault(pipeline.id, Queue())
        pool.put(pipeline)

    def _get_pipeline_from_pool(self, slot, id):
        assert not slot.pipeline, "Cannot allocate pipeline to full device slot"

        # Get the pool. If none available, return
        pool = self._available_pipelines.get(id)
        if not pool:
            return None

        # Try getting a pipeline from the pool. Again, if none available, just return
        try:
            pipeline = pool.get(block=False)
        except queue.Empty:
            return None

        # Assign the pipeline to the slot and activate
        slot.pipeline = pipeline
        slot.pipeline.activate(slot.device)

        return pipeline

    @contextmanager
    def with_engine(self, id=None, task=None):
        """
        Get and activate a pipeline
        TODO: Better activate / deactivate logic. Right now we just keep a max of one pipeline active.
        """

        if id is None:
            id = self._defaults[task if task else "generate"]

        if id is None:
            raise EngineNotFoundError("No engine ID provided and no default is set.")

        # Get the engine spec
        spec = self._find_spec(id=id)
        if not spec or not spec.get("enabled", False):
            raise EngineNotFoundError(f"Engine ID {id} doesn't exist or isn't enabled.")

        if task is not None and task != spec.get("task", "generate"):
            raise ValueError(
                f"Engine ID {id} is for task '{spec.get('task', 'generate')}' not '{task}'"
            )

        # Get device queue slot
        slot = self._device_queue.get()

        # Get pipeline (create if all pipelines for the id are busy)

        # If a pipeline is already active on this device slot, check if it's the right
        # one. If not, deactivate it and clear it
        if slot.pipeline and slot.pipeline.id != id:
            old_id = slot.pipeline.id
            self._return_pipeline_to_pool(slot)

            if self._ram_monitor:
                print(f"Existing pipeline {old_id} deactivated")
                self._ram_monitor.print()

        # If there's no pipeline on this device slot yet, find it (creating it
        # if all the existing pipelines are busy)
        if not slot.pipeline:
            existing = True
            self._get_pipeline_from_pool(slot, id)

            if not slot.pipeline:
                existing = False
                slot.pipeline = self._build_pipeline_for_engine(spec)
                slot.pipeline.activate(slot.device)

            if self._ram_monitor:
                print(f"{'Existing' if existing else 'New'} pipeline {id} activated")
                self._ram_monitor.print()

        try:
            # Do the work
            yield slot.pipeline
        finally:
            # Release device handle
            self._device_queue.put(slot)

        # All done
