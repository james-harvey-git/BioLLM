from __future__ import annotations

import torch
from torch import nn


class EWCState:
    def __init__(self, lambda_: float, fisher_decay: float = 0.99, device: torch.device | str = "cpu") -> None:
        self.lambda_ = lambda_
        self.fisher_decay = fisher_decay
        self.device = torch.device(device)
        self._fisher: dict[str, torch.Tensor] = {}
        self._anchor: dict[str, torch.Tensor] = {}

    def penalty(self, base_model: nn.Module) -> torch.Tensor:
        if not self._anchor:
            return torch.tensor(0.0, device=self.device)

        total = torch.tensor(0.0, device=self.device)
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            fisher = self._fisher.get(name)
            anchor = self._anchor.get(name)
            if fisher is None or anchor is None:
                continue
            total = total + (fisher * (param - anchor).pow(2)).sum()
        return self.lambda_ * total

    def update_fisher(self, base_model: nn.Module, loss: torch.Tensor) -> None:
        params = [(n, p) for n, p in base_model.named_parameters() if p.requires_grad]
        if not params:
            return
        grads = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
        for (name, param), grad in zip(params, grads):
            if grad is None:
                continue
            g2 = grad.detach().pow(2)
            if name not in self._fisher:
                self._fisher[name] = torch.zeros_like(param, device=param.device)
            self._fisher[name] = self.fisher_decay * self._fisher[name] + (1.0 - self.fisher_decay) * g2

    def snapshot_anchor(self, base_model: nn.Module) -> None:
        self._anchor = {
            name: param.detach().clone() for name, param in base_model.named_parameters() if param.requires_grad
        }
        if not self._fisher:
            self._fisher = {
                name: torch.zeros_like(param, device=param.device)
                for name, param in base_model.named_parameters()
                if param.requires_grad
            }
