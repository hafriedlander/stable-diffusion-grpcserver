import functools
import inspect

import generation_pb2
from diffusers import DPMSolverMultistepScheduler, LMSDiscreteScheduler, PNDMScheduler

from sdgrpcserver.k_diffusion import sampling as k_sampling
from sdgrpcserver.pipeline.kschedulers import (
    DPM2AncestralDiscreteScheduler,
    DPM2DiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
)
from sdgrpcserver.pipeline.schedulers.sample_dpmpp_2m import sample_dpmpp_2m
from sdgrpcserver.pipeline.schedulers.scheduling_ddim import DDIMScheduler

DIFFUSERS_SAMPLERS = {
    generation_pb2.SAMPLER_DDIM: (DDIMScheduler, {}),
    generation_pb2.SAMPLER_DDPM: (PNDMScheduler, {"skip_prk_steps": True}),
    generation_pb2.SAMPLER_K_LMS: (LMSDiscreteScheduler, {}),
    generation_pb2.SAMPLER_K_EULER: (EulerDiscreteScheduler, {}),
    generation_pb2.SAMPLER_K_EULER_ANCESTRAL: (EulerAncestralDiscreteScheduler, {}),
    generation_pb2.SAMPLER_K_DPM_2: (DPM2DiscreteScheduler, {}),
    generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL: (DPM2AncestralDiscreteScheduler, {}),
    generation_pb2.SAMPLER_K_HEUN: (HeunDiscreteScheduler, {}),
    generation_pb2.SAMPLER_DPMSOLVERPP_1ORDER: (
        DPMSolverMultistepScheduler,
        {"solver_order": 1},
    ),
    generation_pb2.SAMPLER_DPMSOLVERPP_2ORDER: (
        DPMSolverMultistepScheduler,
        {"solver_order": 2},
    ),
    generation_pb2.SAMPLER_DPMSOLVERPP_3ORDER: (
        DPMSolverMultistepScheduler,
        {"solver_order": 3},
    ),
}

KDIFFUSION_SAMPLERS = {
    generation_pb2.SAMPLER_K_LMS: k_sampling.sample_lms,
    generation_pb2.SAMPLER_K_EULER: k_sampling.sample_euler,
    generation_pb2.SAMPLER_K_EULER_ANCESTRAL: k_sampling.sample_euler_ancestral,
    generation_pb2.SAMPLER_K_DPM_2: k_sampling.sample_dpm_2,
    generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL: k_sampling.sample_dpm_2_ancestral,
    generation_pb2.SAMPLER_K_HEUN: k_sampling.sample_heun,
    generation_pb2.SAMPLER_DPM_FAST: k_sampling.sample_dpm_fast,
    generation_pb2.SAMPLER_DPM_ADAPTIVE: k_sampling.sample_dpm_adaptive,
    generation_pb2.SAMPLER_K_DPMPP_2S_ANCESTRAL: k_sampling.sample_dpmpp_2s_ancestral,
    generation_pb2.SAMPLER_K_DPMPP_SDE: k_sampling.sample_dpmpp_sde,
    generation_pb2.SAMPLER_K_DPMPP_2M: functools.partial(
        sample_dpmpp_2m, warmup_lms=True, ddim_cutoff=0.1
    ),
    # These are deprecated, as there are now official enums above from the Stability-AI API
    generation_pb2.SAMPLER_DPMSOLVERPP_2S_ANCESTRAL: k_sampling.sample_dpmpp_2s_ancestral,
    generation_pb2.SAMPLER_DPMSOLVERPP_SDE: k_sampling.sample_dpmpp_sde,
    generation_pb2.SAMPLER_DPMSOLVERPP_2M: functools.partial(
        sample_dpmpp_2m, warmup_lms=True, ddim_cutoff=0.1
    ),
}


@functools.cache
def sampler_properties(include_diffusers=True, include_kdiffusion=True):
    out = []

    all_noise_types = [
        generation_pb2.SAMPLER_NOISE_NORMAL,
        generation_pb2.SAMPLER_NOISE_BROWNIAN,
    ]
    normal_only = [generation_pb2.SAMPLER_NOISE_NORMAL]

    samplers = {
        **(DIFFUSERS_SAMPLERS if include_diffusers else {}),
        **(KDIFFUSION_SAMPLERS if include_kdiffusion else {}),
    }

    for k, v in samplers.items():
        if callable(v):
            args = set(inspect.signature(v).parameters.keys())

            out.append(
                dict(
                    sampler=k,
                    supports_eta="eta" in args,
                    supports_churn="s_churn" in args,
                    supports_sigma_limits="sigmas" in args or "sigma_min" in args,
                    supports_karras_rho="sigmas" in args,
                    supported_noise_types=all_noise_types
                    if "noise_sampler" in args
                    else normal_only,
                )
            )
        else:
            scheduler_class, kwargs = v
            args = set(inspect.signature(scheduler_class.step).parameters.keys())

            out.append(dict(sampler=k, supports_eta="eta" in args))

    return out


def build_sampler_set(config, include_diffusers=True, include_kdiffusion=True):
    out = {}

    samplers = {
        **(DIFFUSERS_SAMPLERS if include_diffusers else {}),
        **(KDIFFUSION_SAMPLERS if include_kdiffusion else {}),
    }

    for k, v in samplers.items():
        if callable(v):
            out[k] = v
        else:
            scheduler_class, kwargs = v
            class_config = {**config, **kwargs}
            # Filter config to what the specific scheduler accepts
            ctor_keys = inspect.signature(scheduler_class.__init__).parameters.keys()
            class_config = {k: v for k, v in class_config.items() if k in ctor_keys}
            # And construct
            out[k] = scheduler_class(**class_config)

    return out
