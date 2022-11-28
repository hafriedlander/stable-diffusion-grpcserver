import inspect
from typing import Callable

import torch
from sdgrpcserver.k_diffusion import sampling as k_sampling, utils as k_utils, external as k_external

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from sdgrpcserver.pipeline.kschedulers.scheduling_utils import KSchedulerMixin

from sdgrpcserver.pipeline.randtools import *
from sdgrpcserver.pipeline.unet_wrappers.types import *

SCHEDULER_NOISE_TYPE = Literal["brownian", "normal"]

class SchedulerCallback(Protocol):
    def __call__(self, i: int, t: int, latents: PX0Tensor) -> None:
        pass

class DiffusersSchedulerCommon(Protocol):
    alphas_cumprod: Tensor
    timesteps: Tensor
    init_noise_sigma: float
    def set_timesteps(self, num_inference_steps: int, device: str|torch.device): pass
    @abstractmethod
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, **step_kwargs): pass
    @abstractmethod
    def scale_model_input(self, sample: torch.FloatTensor, timestep: int|None) -> torch.FloatTensor: pass
    @abstractmethod
    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.FloatTensor) -> torch.FloatTensor: pass

class CommonScheduler:

    # -- These methods are called by the pipeline itself

    def __init__(
        self, 
        scheduler: Union[SchedulerMixin, KSchedulerMixin, Callable],
        generators: List[torch.Generator],
        device: torch.device,
        dtype: torch.dtype,
        vae,
        callback: Optional[SchedulerCallback] = None,
        callback_steps: int = 1
    ):
        self.scheduler = scheduler
        self.generators = generators
        self.device = device
        self.dtype = dtype
        self.vae = vae

        self.eps_unets = None

        self.callback = callback
        self.callback_steps = callback_steps

    def set_eps_unets(
        self,
        eps_unets: List[NoisePredictionUNet]
    ):
        if getattr(self, "unet", False):
            raise RuntimeError("Can't set eps_unet once set_timesteps has been called")
        self.eps_unets = eps_unets

    def set_timesteps(
        self, 
        num_inference_steps: int,
        start_offset: int = None,
        strength: float = None,
        sigma_min: float = None,
        sigma_max: float = None,
        karras_rho: float = None,
        eta: float = None,
        churn: float = None,
        churn_tmin: float = None,
        churn_tmax: float = None,
        noise_type: SCHEDULER_NOISE_TYPE = "normal",
        prediction_type = "epsilon"
    ):
        raise NotImplementedError("Subclass to implement")

    def loop(
        self, 
        latents: Tensor,
        progress_wrapper
    ) -> Tensor:
        raise NotImplementedError("Subclass to implement")

    # -- These methods are for unet wrappers and modes

    def prepare_initial_latents(
        self, 
        latents: Tensor
    ) -> Tensor:
        raise NotImplementedError("Subclass to implement")

    def scale_latents(
        self,
        latents: Tensor,
        t: ScheduleTimestep
    ):
        raise NotImplementedError("Subclass to implement")

    def add_noise(
        self,
        latents: Tensor,
        noise: Tensor,
        t: ScheduleTimestep,
    ) -> Tensor:
        raise NotImplementedError("Subclass to implement")

    # -- Internal utility functions. 

    # TODO: I suspect this isn't needed, (and k-diffusion comes with similar)
    def match_shape(self, values: Tensor, broadcast_array: Tensor):
        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]

        return values.to(broadcast_array.device)

    def t_to_index(self, timestep):
        timesteps = self.scheduler.timesteps
        timesteps = timesteps.to(timestep.device)

        dists = timestep - timesteps
        return dists.abs().argmin().item()


class DiffusersSchedulerBase(CommonScheduler):

    scheduler: DiffusersSchedulerCommon

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        scheduler_step_args = set(inspect.signature(self.scheduler.step).parameters.keys())

        self.accepts_eta = "eta" in scheduler_step_args
        self.accepts_generator = "generator" in scheduler_step_args
        self.accepts_noise_predictor = "noise_predictor" in scheduler_step_args

        self.unet: Optional[DiffusersSchedulerUNet] = None

    def set_timesteps(
        self, 
        num_inference_steps: int,
        start_offset: int|None = None,
        strength: float|None = None,
        sigma_min: float|None = None,
        sigma_max: float|None = None,
        karras_rho: float|None = None,
        eta: float|None = None,
        churn: float|None = None,
        churn_tmin: float|None = None,
        churn_tmax: float|None = None,
        noise_type: SCHEDULER_NOISE_TYPE = "normal",
        prediction_type = "epsilon"
    ):
        if self.eps_unets is None:
            raise ValueError("Epsilon unet needs to be set before timesteps")

        if sigma_min is not None or sigma_max is not None:
            print("Warning: This scheduler doen't accept sigma_min or sigma_max. Ignoring.")
        if karras_rho is not None:
            print("Warning: This scheduler doesn't accept karras_rho. Ignoring.")
        if churn is not None or churn_tmin is not None or churn_tmax is not None:
            print("Warning: This scheduler doesn't accept churn. Ignoring.")
        if eta is not None and not self.accepts_eta:
            print("Warning: This scheduler doesn't accept eta. Ignoring.")
        if noise_type != "normal":
            print("Warning: This scheduler only accepts normal noise. Ignoring.")

        self.eta = eta
        self.num_inference_steps = num_inference_steps

        if strength is not None:
            if start_offset is not None:
                raise ValueError("Can't pass both start_offset and strength to set_timesteps")

            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            self.start_offset = max(num_inference_steps - init_timestep + offset, 0)
        elif start_offset is not None:
            self.start_offset = start_offset
        else:
            self.start_offset = 0

        # Wrap the scheduler. TODO: Better place than here?
        self.unets = [self.wrap_unet(eps_unet) for eps_unet in self.eps_unets]

        self.scheduler.set_timesteps(num_inference_steps)
        self.start_timestep = self.scheduler.timesteps[self.start_offset]

    def wrap_scaled_unet(self, unet: NoisePredictionUNet) -> ScalingUNet:
        def wrapped_unet(latents: UnscaledXtTensor, t: ScheduleTimestep) -> PredictedNoiseTensor:
            scaled_latents: PrescaledXtTensor = self.scheduler.scale_model_input(latents, t)
            return unet(scaled_latents, t)

        return wrapped_unet

    @abstractmethod
    def predict_x0(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        t: ScheduleTimestep
    ) -> PX0Tensor:
        pass

    def wrap_unet(self, unet: NoisePredictionUNet) -> DiffusersSchedulerUNet:
        step_kwargs = {}
        if self.accepts_eta and self.eta is not None: step_kwargs["eta"] = self.eta
        if self.accepts_generator: step_kwargs["generator"] = self.generators[0]
        if self.accepts_noise_predictor: step_kwargs["noise_predictor"] = unet

        def wrapped_unet(latents, t, u):
            # predict the noise residual
            noise_pred = unet(latents, t)

            if self.callback:
                i = self.t_to_index(t)
                if i % self.callback_steps == 0:
                    self.callback(i, t, self.predict_x0(latents, noise_pred, t))

            return self.scheduler.step(noise_pred, t, latents, **step_kwargs).prev_sample
        
        return wrapped_unet

    def loop(self, latents: UnscaledXtTensor, progress_wrapper):
        unet = self.unet
        timesteps_tensor = self.scheduler.timesteps[self.start_offset:].to(self.device)

        for i, t in enumerate(progress_wrapper(timesteps_tensor)):
            u = i / len(timesteps_tensor)
            u = max(min(u, 0.999), 0)
            latents = unet(latents, t, u=u)

        return latents

    def prepare_initial_latents(
        self, 
        latents: Tensor
    ) -> Tensor:
        return latents * self.scheduler.init_noise_sigma

    def add_noise(
        self,
        latents: Tensor,
        noise: Tensor,
        t: ScheduleTimestep
    ) -> Tensor:
        return self.scheduler.add_noise(latents, noise, torch.tensor([t], device=self.device))

    def scale_latents(
        self,
        latents: Tensor,
        t: ScheduleTimestep
    ):
        return self.scheduler.scale_model_input(latents, t)           

class DiffusersScheduler(DiffusersSchedulerBase):

    scheduler: SchedulerMixin

    def predict_x0(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        t: ScheduleTimestep
    ):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        return pred_original_sample

        # From Diffusers CLIP pipeline, this blends some of the original latent back in
        # Why? Don't know, and seems to make result worse. Add if compatibility most important
        fac = torch.sqrt(beta_prod_t)
        return pred_original_sample * (fac) + latents * (1 - fac)    

class DiffusersKScheduler(DiffusersSchedulerBase):

    scheduler: KSchedulerMixin

    def predict_x0(
        self,
        latents: Tensor,
        noise_pred: Tensor,
        t: ScheduleTimestep
    ):
        sigma = self.scheduler.t_to_sigma(t)
        return latents - sigma * noise_pred

class KDiffusionUNetWrapper(k_external.DiscreteEpsDDPMDenoiser):
    def __init__(self, unet: NoisePredictionUNet, alphas_cumprod: Tensor):
        super().__init__(unet, alphas_cumprod, quantize=True)

    def get_eps(self, latents, t, **kwargs):
        return self.inner_model(latents, t)

class KDiffusionVUNetWrapper(k_external.DiscreteVDDPMDenoiser):
    def __init__(self, unet: NoisePredictionUNet, alphas_cumprod: Tensor):
        super().__init__(unet, alphas_cumprod, quantize=True)

    def get_v(self, latents, t, **kwargs):
        return self.inner_model(latents, t)

class KDiffusionPositionTracker:

    def __init__(self, progress_wrapper, sigmas):
        self.i = None
        self.i_max = len(sigmas)-1
        self.progress_wrapper = progress_wrapper
        self.sigmas = sigmas

    # Unet wrapper that adds u (floating point progress, from 0..1)
    def get_u(self, sigma):
        i, i_max = self.i, self.i_max
        # If we're not looping through a fixed range, fall back
        # to guessing position based on sigmas
        if i is None: i = len([s for s in self.sigmas if s >= sigma])

        u = i / i_max
        return max(min(u, 0.999), 0)

    def trange(self, max, **kwargs):
        self.i_max = max
        for j in self.progress_wrapper(range(max)):
            self.i = j
            yield j

    def tqdm(self, **kwargs):
        return self.progress_wrapper(None)

class KDiffusionScheduler(CommonScheduler):

    scheduler: Callable

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        scheduler_keys = inspect.signature(self.scheduler).parameters.keys()
        self.accepts_sigma_min = "sigma_min" in scheduler_keys
        self.accepts_sigmas = "sigmas" in scheduler_keys
        self.accepts_n = "n" in scheduler_keys
        self.accepts_eta = "eta" in scheduler_keys
        self.accepts_s_churn = "s_churn" in scheduler_keys
        self.accepts_noise_sampler = "noise_sampler" in scheduler_keys

        self.unet: Optional[KDiffusionSchedulerUNet] = None

    def get_betas(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> Tensor:
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, device=self.device) ** 2

    def get_alphas(self, betas: Tensor) -> Tensor:
        return 1.0 - betas

    def get_alphas_cumprod(self, alphas: Tensor) -> Tensor:
        return torch.cumprod(alphas, dim=0)

    def set_timesteps(
        self, 
        num_inference_steps: int,
        start_offset: int = None,
        strength: float = None,
        sigma_min: float = None,
        sigma_max: float = None,
        karras_rho: float = None,
        eta: float = None,
        churn: float = None,
        churn_tmin: float = None,
        churn_tmax: float = None,
        noise_type: SCHEDULER_NOISE_TYPE = "normal",
        prediction_type = "epsilon"
    ):
        if self.eps_unets is None:
            raise ValueError("Epsilon unet needs to be set before timesteps")

        if (sigma_min is not None or sigma_max is not None) and not (self.accepts_sigma_min or self.accepts_sigmas):
            print("Warning: This scheduler doen't accept sigma_min or sigma_max. Ignoring.")
        if karras_rho is not None and not self.accepts_sigmas:
            print("Warning: This scheduler doen't accept karras_rho. Ignoring.")
        if churn is not None and not self.accepts_s_churn:
            print("Warning: This scheduler doen't accept churn. Ignoring.")
        if eta is not None and not self.accepts_eta:
            print("Warning: This scheduler doen't accept eta. Ignoring.")

        betas = self.get_betas()
        alphas = self.get_alphas(betas)
        alphas_cumprod = self.get_alphas_cumprod(alphas)

        if prediction_type == "v_prediction": wrapper_klass = KDiffusionVUNetWrapper
        else: wrapper_klass = KDiffusionUNetWrapper

        self.unets = [wrapper_klass(eps_unet, alphas_cumprod) for eps_unet in self.eps_unets]
        self._unet = self.unets[0]

        # clamp sigma_min and sigma_max
        if sigma_min is not None: sigma_min = max(self._unet.sigma_min, sigma_min)
        if sigma_max is not None: sigma_max = min(self._unet.sigma_max, sigma_max)

        # Calculate the Karras schedule if Rho is provided
        if karras_rho is not None:
            # quantize sigma min and max
            if sigma_min is not None: 
                sigma_min = self._unet.t_to_sigma(self._unet.sigma_to_t(torch.tensor(sigma_min, device=self.device))).to('cpu')
            if sigma_max is not None: 
                sigma_max = self._unet.t_to_sigma(self._unet.sigma_to_t(torch.tensor(sigma_max, device=self.device))).to('cpu')

            self.sigmas = k_sampling.get_sigmas_karras(
                n=num_inference_steps,
                sigma_max=sigma_max if sigma_max is not None else self._unet.sigma_max.to('cpu'),
                sigma_min=sigma_min if sigma_min is not None else self._unet.sigma_min.to('cpu'),
                rho=karras_rho,
                device=self.device,
            )

        # Calculate the linear schedule - this is the same as DiscreteSchedule.get_sigmas but it supports truncated schedules
        else:
            t_min = 0
            if sigma_min is not None: t_min = self._unet.sigma_to_t(torch.tensor(sigma_min, device=self.device))
            t_max = len(self._unet.sigmas) - 1
            if sigma_max is not None: t_max = self._unet.sigma_to_t(torch.tensor(sigma_max, device=self.device))

            t = torch.linspace(t_max, t_min, num_inference_steps, device=self.device)
            self.sigmas = k_sampling.append_zero(self._unet.t_to_sigma(t))

        self.eta = eta
        self.churn = churn
        self.churn_tmin = churn_tmin
        self.churn_tmax = churn_tmax
        self.noise_type = noise_type

        self.num_inference_steps = num_inference_steps

        if strength is not None:
            if start_offset is not None:
                raise ValueError("Can't pass both start_offset and strength to set_timesteps")

            init_timestep = int(num_inference_steps * strength)
            init_timestep = min(init_timestep, num_inference_steps)
            self.start_offset = max(num_inference_steps - init_timestep, 0)
        elif start_offset is not None:
            self.start_offset = start_offset
        else:
            self.start_offset = 0

        self.start_timestep = self._unet.sigma_to_t(self.sigmas[self.start_offset])

    def prepare_initial_latents(
        self, 
        latents: Tensor
    ) -> Tensor:
        return latents * self.sigmas[0]

    def scale_latents(
        self,
        latents: Tensor,
        t: ScheduleTimestep
    ):
        sigma = self._unet.t_to_sigma(t)
        _, cin =  self._unet.get_scalings(sigma)
        cin = k_utils.append_dims(cin, latents.ndim)
        cin = cin.to(latents.dtype)
        return latents * cin

    def add_noise(
        self,
        latents: Tensor,
        noise: Tensor,
        t: ScheduleTimestep
    ) -> Tensor:
        sigma = self._unet.t_to_sigma(t)
        sigmas = self.match_shape(sigma, noise)
        return latents + noise * sigmas

    def loop(self, latents, progress_wrapper):
        unet = self.unet

        sigmas = self.sigmas[self.start_offset:].to(self.dtype)

        sigma_min = sigmas[sigmas > 0].min()
        sigma_max = sigmas.max()

        tracker = KDiffusionPositionTracker(progress_wrapper, sigmas)

        # Unet wrapper that adds u (floating point progress, from 0..1)
        def wrapped(latents, sigma):
            u = tracker.get_u(sigma)
            return unet(latents, sigma, u=u)

        k_sampling.trange = tracker.trange
        k_sampling.tqdm = tracker.tqdm

        kwargs = {}
        if self.accepts_eta and self.eta is not None:
            kwargs['eta'] = self.eta
        if self.accepts_s_churn and self.churn is not None:
            kwargs['s_churn'] = self.churn
            if self.churn_tmin is not None: kwargs['s_tmin'] = self.churn_tmin
            if self.churn_tmax is not None: kwargs['s_tmax'] = self.churn_tmax
        if self.accepts_n:
            kwargs['n'] = self.num_inference_steps

        if self.accepts_sigmas:
            kwargs['sigmas'] = sigmas
        else:
            kwargs['sigma_min']=sigma_min
            kwargs['sigma_max']=sigma_max

        if self.accepts_noise_sampler:
            if self.noise_type == "brownian":
                seeds = [
                    torch.randint(0, 2 ** 63 - 1, [], generator=g, device=g.device).item()
                    for g in self.generators
                ]
                kwargs['noise_sampler'] = \
                    k_sampling.BrownianTreeNoiseSampler(latents, sigma_min, sigma_max, seed=seeds)
            else:
                kwargs['noise_sampler'] = \
                    lambda _, __: batched_randn(latents.shape, self.generators, self.device, self.dtype)


        def callback_wrapper(d):
            i, sigma, denoised = d["i"], d["sigma"], d["denoised"]
            if i % self.callback_steps == 0:
                t = self._unet.sigma_to_t(sigma)
                self.callback(i, t, denoised)

        if self.callback:
            kwargs['callback'] = callback_wrapper

        return self.scheduler(wrapped, latents, **kwargs)
