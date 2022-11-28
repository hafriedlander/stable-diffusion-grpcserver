from abc import abstractmethod
from typing import Callable, List, Tuple, Iterable, Optional, Union, Literal, NewType, Protocol, overload
from torch import Tensor

# Some types to describe the various structures of unet. First some return types

# An Xt (ie a sample that includes some amount of noise), not scaled for scheduler purposes yet
UnscaledXtTensor = NewType('UnscaledXtTensor', Tensor)
# An Xt scaled depending on current timestep
PrescaledXtTensor = NewType('PrescaledXtTensor', Tensor)
# And XtTensor, either Pre or Unscaled
XtTensor = Union[UnscaledXtTensor, PrescaledXtTensor]

# The predicted noise in a sample (eps)
PredictedNoiseTensor = NewType('PredictedNoiseTensor', Tensor)
# The predicted X0 (i.e Xt - PredictedNoise)
PX0Tensor = NewType('PX0Tensor', Tensor)

# Sigma
ScheduleSigma = Union[float, Tensor]
# Timestep (from 1000 to 0 usually)
ScheduleTimestep = Union[float, Tensor]
# Progress float, range [0..1)
ScheduleProgress = NewType('Progress', float)


# The Core Diffusers UNet
DiffusersUNet = Callable[[PrescaledXtTensor, ScheduleTimestep, Tensor], PredictedNoiseTensor]
# A Wrapped UNet where the hidden_state argument inside the wrapping
NoisePredictionUNet = Callable[[XtTensor, ScheduleTimestep], PredictedNoiseTensor]
# A UNet that wraps a NoisePredictionUNet to take an XtTensor instead of a ScaledXtTensor
ScalingUNet = Callable[[UnscaledXtTensor, ScheduleTimestep], PredictedNoiseTensor]

# A KDiffusion wrapped UNet
class KDiffusionSchedulerUNet(Protocol):
    @abstractmethod
    def __call__(self, latents: UnscaledXtTensor, sigma: ScheduleSigma, u: float) -> PX0Tensor:
        pass

class DiffusersSchedulerUNet(Protocol):
    @abstractmethod
    def __call__(self, latents: UnscaledXtTensor, t: ScheduleTimestep, u: float) -> XtTensor:
        pass

class GenericSchedulerUNet(KDiffusionSchedulerUNet, DiffusersSchedulerUNet, Protocol):
    @abstractmethod
    def __call__(self, latents: UnscaledXtTensor, __step: Union[ScheduleSigma, ScheduleTimestep], u: float) -> Union[PX0Tensor, XtTensor]:
        pass

