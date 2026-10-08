"""The deliberately small interface used by pretraining code."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class ModelCapabilities:
    varlen: bool
    generation: bool
    sliding_window: bool
    compile: bool = True


@dataclass(frozen=True)
class PretrainOptimizerConfig:
    unembedding_lr: float
    embedding_lr: float
    matrix_lr: float
    scalar_lr: float
    weight_decay: float


@runtime_checkable
class PretrainModel(Protocol):
    config: object

    def init_weights(self) -> None: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        loss_reduction: str = "mean",
    ): ...

    def setup_optimizer(self, **kwargs): ...

    def setup_pretraining_optimizer(self, config: PretrainOptimizerConfig): ...

    def estimate_flops(self) -> int: ...

    def num_scaling_params(self) -> dict[str, int]: ...

    def scaling_parameter_count(self) -> int: ...
