from typing import List, Optional, Sequence

import torch


def batched_rand(
    shape: Sequence[int],
    generators: List[torch.Generator],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:

    if shape[0] % len(generators) != 0:
        raise ValueError(
            f"shape[0] ({shape[0]}) needs to be a multiple of len(generators) ({len(generators)})"
        )

    latents = torch.cat(
        [
            torch.rand(
                (1, *shape[1:]),
                generator=generator,
                device=generator.device,
                dtype=dtype,
            )
            for generator in generators * (shape[0] // len(generators))
        ],
        dim=0,
    )

    return latents.to(device)


def batched_randn(
    shape: Sequence[int],
    generators: List[torch.Generator],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:

    if shape[0] % len(generators) != 0:
        raise ValueError(
            f"shape[0] ({shape[0]}) needs to be a multiple of len(generators) ({len(generators)})"
        )

    latents = torch.cat(
        [
            torch.randn(
                (1, *shape[1:]),
                generator=generator,
                device=generator.device,
                dtype=dtype,
            )
            for generator in generators * (shape[0] // len(generators))
        ],
        dim=0,
    )

    return latents.to(device)


class TorchRandOverride:
    def __init__(self, generators):
        self.generators = generators

    def randn_like(
        self,
        input: torch.Tensor,
        *args,
        dtype: Optional[torch.dtype] = None,
        device=None,
        **kwargs,
    ):
        if input.shape[0] % len(self.generators) != 0:
            if dtype:
                kwargs["dtype"] = dtype
            if device:
                kwargs["device"] = device
            return torch.randn_like(input, *args, **kwargs)

        if device is None:
            device = input.device
        if dtype is None:
            dtype = input.dtype
        return batched_randn(input.shape, self.generators, device, dtype)

    def randint_like(
        self,
        input,
        *args,
        high=None,
        low=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        **kwargs,
    ):
        if len(args) == 1:
            high = args[0]
        elif args:
            low = args[0]
            high = args[1]
        if low is None:
            low = 0

        if input.shape[0] % len(self.generators) != 0:
            print("Skip")
            return torch.randint_like(
                input,
                low=low,
                high=high,
                dtype=dtype,
                layout=layout,
                device=device,
                **kwargs,
            )

        latents = torch.cat(
            [
                torch.randint(
                    size=(1, *input.shape[1:]),
                    low=low,
                    high=high,
                    generator=generator,
                    device=generator.device,
                    dtype=dtype,
                )
                for generator in self.generators
            ],
            dim=0,
        )

        return latents.to(device)

    def __getattr__(self, item):
        return getattr(torch, item)
