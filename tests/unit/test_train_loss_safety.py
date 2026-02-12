from __future__ import annotations

import torch

from biollm_cls.train import _safe_token_ce


def test_safe_token_ce_handles_all_ignored_targets() -> None:
    logits = torch.randn(2, 4, 16, requires_grad=True)
    targets = torch.full((2, 4), -100, dtype=torch.long)

    loss, n_tokens = _safe_token_ce(logits, targets, vocab_size=16)

    assert n_tokens == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_safe_token_ce_matches_cross_entropy_with_supervised_tokens() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, requires_grad=True)
    targets = torch.randint(0, 16, (2, 4), dtype=torch.long)
    targets[0, 0] = -100

    loss, n_tokens = _safe_token_ce(logits, targets, vocab_size=16)
    ref = torch.nn.functional.cross_entropy(logits.view(-1, 16), targets.view(-1), ignore_index=-100)

    assert n_tokens == int((targets != -100).sum().item())
    assert torch.allclose(loss, ref)
