from __future__ import annotations

import torch
from torch import nn

from biollm_cls.consolidation.ewc import EWCState


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_ewc_penalty_is_small_at_anchor_and_grows_with_drift() -> None:
    torch.manual_seed(0)
    model = Tiny()
    ewc = EWCState(lambda_=10.0, fisher_decay=0.9)

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    loss = ((model(x) - y) ** 2).mean()
    ewc.update_fisher(model, loss)
    ewc.snapshot_anchor(model)

    p0 = ewc.penalty(model).item()
    assert p0 >= 0.0
    assert p0 < 1e-6

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)

    p1 = ewc.penalty(model).item()
    assert p1 > p0
