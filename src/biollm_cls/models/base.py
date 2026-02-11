from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class NeocortexAdapter(nn.Module, ABC):
    """Common neocortical interface used by wake/sleep training code."""

    hidden_size: int
    vocab_size: int

    @abstractmethod
    def forward_to_injection(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_from_injection(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_base(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

