from __future__ import annotations

import pytest

from biollm_cls.config import ModelConfig
from biollm_cls.models.factory import build_neocortex
from biollm_cls.models.neocortex import TinyCausalTransformer


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        provider="tiny",
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        seq_len=16,
        dropout=0.0,
        injection_layer=1,
    )


def test_build_tiny_neocortex() -> None:
    model = build_neocortex(_tiny_cfg())
    assert isinstance(model, TinyCausalTransformer)


def test_unknown_provider_rejected() -> None:
    cfg = _tiny_cfg()
    cfg.provider = "unknown"
    with pytest.raises(ValueError):
        build_neocortex(cfg)
