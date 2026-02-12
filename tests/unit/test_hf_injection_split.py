from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from biollm_cls.config import ModelConfig
from biollm_cls.models.hf_neocortex import HFNeocortexAdapter


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, add_value: float) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))
        self.register_buffer("add", torch.full((1, 1, hidden_size), add_value))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.proj(hidden) + self.add.to(hidden.dtype),)


class _FakeDecoderOnlyLM(nn.Module):
    def __init__(self, layer_path: str, num_layers: int = 4, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._layer_path = layer_path

        layers = nn.ModuleList([_FakeDecoderLayer(hidden_size, float(i + 1)) for i in range(num_layers)])
        self._set_layers(layer_path, layers)

    def _set_layers(self, path: str, layers: nn.ModuleList) -> None:
        current: nn.Module = self
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(current, part):
                setattr(current, part, nn.Module())
            current = getattr(current, part)
        setattr(current, parts[-1], layers)

    def _get_layers(self) -> nn.ModuleList:
        current: object = self
        for part in self._layer_path.split("."):
            current = getattr(current, part)
        assert isinstance(current, nn.ModuleList)
        return current

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> types.SimpleNamespace:
        del use_cache
        hidden = self.embed_tokens(input_ids)
        hidden_states: list[torch.Tensor] | None = [hidden] if output_hidden_states else None

        for layer in self._get_layers():
            layer_out = layer(hidden)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            if hidden_states is not None:
                hidden_states.append(hidden)

        logits = self.lm_head(hidden)
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )


class _FakeAutoModelForCausalLM:
    model_factory = staticmethod(lambda: _FakeDecoderOnlyLM("model.layers"))

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> nn.Module:
        del args, kwargs
        return cls.model_factory()


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, model_factory) -> None:
    _FakeAutoModelForCausalLM.model_factory = staticmethod(model_factory)
    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=_FakeAutoModelForCausalLM)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _hf_cfg(**kwargs) -> ModelConfig:
    return ModelConfig(
        provider="hf",
        vocab_size=32,
        hidden_size=0,
        num_layers=0,
        num_heads=0,
        seq_len=16,
        dropout=0.0,
        injection_layer=-1,
        hf_model_name="fake/model",
        **kwargs,
    )


@pytest.mark.parametrize(
    "layer_path",
    ["model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"],
)
def test_hf_decoder_split_mode_for_supported_paths(
    monkeypatch: pytest.MonkeyPatch,
    layer_path: str,
) -> None:
    _install_fake_transformers(monkeypatch, lambda: _FakeDecoderOnlyLM(layer_path=layer_path, num_layers=4))

    adapter = HFNeocortexAdapter(_hf_cfg(hf_injection_fraction=0.7))
    assert adapter.hf_injection_mode == "decoder_split"
    assert adapter.hf_num_decoder_layers == 4
    assert adapter.hf_injection_layer_idx == 2

    input_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    split_hidden = adapter.forward_to_injection(input_ids)
    split_logits = adapter.forward_from_injection(split_hidden)
    base_logits = adapter.forward_base(input_ids)

    # In split mode, replaying the split hidden through post layers should match base logits.
    assert torch.allclose(split_logits, base_logits, atol=1e-5)


def test_hf_legacy_fallback_for_unsupported_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, lambda: _FakeDecoderOnlyLM(layer_path="decoder.blocks", num_layers=4))
    HFNeocortexAdapter._warned_legacy_once = False

    with pytest.warns(RuntimeWarning):
        adapter = HFNeocortexAdapter(_hf_cfg(hf_allow_legacy_injection_fallback=True))

    assert adapter.hf_injection_mode == "legacy_head_only"
    assert adapter.hf_injection_layer_idx == -1

    input_ids = torch.randint(0, 32, (1, 5), dtype=torch.long)
    hidden = adapter.forward_to_injection(input_ids)
    logits = adapter.forward_from_injection(hidden)
    assert logits.shape == (1, 5, 32)


def test_hf_legacy_fallback_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, lambda: _FakeDecoderOnlyLM(layer_path="decoder.blocks", num_layers=4))

    with pytest.raises(ValueError):
        HFNeocortexAdapter(_hf_cfg(hf_allow_legacy_injection_fallback=False))
