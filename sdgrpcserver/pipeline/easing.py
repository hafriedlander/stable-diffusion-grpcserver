from typing import Literal, Type

from easing_functions import easing

EASING_TYPE = Literal[
    "linear", "quad", "cubic", "quartic", "quintic", "sine", "circular", "expo"
]

EASINGS: dict[EASING_TYPE, Type[easing.EasingBase]] = {
    "linear": easing.LinearInOut,
    "quad": easing.QuadEaseInOut,
    "cubic": easing.CubicEaseInOut,
    "quartic": easing.QuarticEaseInOut,
    "quintic": easing.QuinticEaseInOut,
    "sine": easing.SineEaseInOut,
    "circular": easing.CircularEaseInOut,
    "expo": easing.ExponentialEaseInOut,
}


class Easing:
    def __init__(
        self,
        floor: float,
        start: float,
        end: float,
        easing: EASING_TYPE | Type[easing.EasingBase],
    ):
        self.floor = 0
        self.start = 0.1
        self.end = 0.3

        if isinstance(easing, str):
            easing = EASINGS[easing]

        self.easing = easing(
            end=1 - floor, duration=1 - (start + end)  # type: ignore - easing_functions takes floats just fine
        )

    def interp(self, u: float):
        if u < self.start:
            return self.floor
        if u > 1 - self.end:
            return 1

        return self.floor + self.easing(u - self.start)
