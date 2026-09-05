"""Abstract base class for all randcompress model backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class RandCompressModel(ABC):

    @abstractmethod
    def init_frozen(self, seed: int) -> dict[str, Tensor]:
        """Return named frozen weight tensors (registered as buffers, no grad)."""

    @abstractmethod
    def init_adapters(self, seed: int) -> dict[str, Tensor]:
        """Return named trainable tensors (requires_grad=True).

        If use_hira: keys like 'layer0.W_q.A', 'layer0.W_q.B', ...
        If not use_hira: keys like 'layer0.W_q.B', 'layer0.W_q.A' (plain LoRA).
        Always includes 'output_proj'.
        """

    @abstractmethod
    def forward(
        self,
        frozen: dict[str, Tensor],
        adapters: dict[str, Tensor],
        tokens: Tensor,         # [B, T] int32
        states: Any,
    ) -> tuple[Tensor, Any]:
        """Returns (logits [B, T, oh, ov], new_states)."""

    @abstractmethod
    def init_states(self, batch_size: int, device: torch.device) -> Any:
        """Return zero initial states."""

    @abstractmethod
    def step(
        self,
        frozen: dict[str, Tensor],
        adapters: dict[str, Tensor],
        token: Tensor,          # [B] int32
        state: Any,
        t: int,
    ) -> tuple[Tensor, Any]:
        """Single autoregressive step. Returns (logits [B, oh, ov], new_state)."""

    def count_params(self, adapters: dict[str, Tensor]) -> int:
        return sum(p.numel() for p in adapters.values())
