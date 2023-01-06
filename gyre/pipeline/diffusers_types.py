from dataclasses import dataclass


@dataclass
class VaeConfig:
    block_out_channels: list[int]


@dataclass
class UnetConfig:
    sample_size: int | None
    attention_head_dim: int | list[int]
