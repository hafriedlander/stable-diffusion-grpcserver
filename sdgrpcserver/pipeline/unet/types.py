from abc import abstractmethod
from typing import NewType, Protocol, overload

from torch import Tensor

# Some types to describe the various structures of unet. First some return types

# An Xt (ie a sample that includes some amount of noise)
XtTensor = Tensor
# The predicted noise in a sample (eps)
EpsTensor = Tensor
# The predicted X0 (i.e Xt - PredictedNoise)
PX0Tensor = Tensor

# Sigma
ScheduleSigma = float | Tensor
# Timestep (from 1000 to 0 usually)
ScheduleTimestep = int | Tensor
# Progress float, range [0..1)
ScheduleProgress = NewType("Progress", float)


# The Core Diffusers UNet
class DiffusersUNetOutput(Protocol):
    sample: EpsTensor


class DiffusersUNet(Protocol):
    @abstractmethod
    def __call__(
        self, latents: XtTensor, t: ScheduleTimestep, encoder_hidden_states: Tensor
    ) -> DiffusersUNetOutput:
        raise NotImplementedError


# A Wrapped UNet where the hidden_state argument inside the wrapping
class NoisePredictionUNet(Protocol):
    @abstractmethod
    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        raise NotImplementedError


# A KDiffusion wrapped UNet
class KDiffusionSchedulerUNet(Protocol):
    @abstractmethod
    def __call__(self, latents: XtTensor, sigma: ScheduleSigma, u: float) -> PX0Tensor:
        raise NotImplementedError


class DiffusersSchedulerUNet(Protocol):
    @abstractmethod
    def __call__(self, latents: XtTensor, t: ScheduleTimestep, u: float) -> XtTensor:
        raise NotImplementedError


class GenericSchedulerUNet:
    @overload
    @abstractmethod
    def __call__(self, latents: XtTensor, sigma: ScheduleSigma, u: float) -> PX0Tensor:
        pass

    @overload
    @abstractmethod
    def __call__(self, latents: XtTensor, t: ScheduleTimestep, u: float) -> XtTensor:
        pass

    @abstractmethod
    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        raise NotImplementedError
