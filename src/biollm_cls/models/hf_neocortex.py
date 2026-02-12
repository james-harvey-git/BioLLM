from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

from biollm_cls.config import ModelConfig
from biollm_cls.models.base import NeocortexAdapter


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    value = dtype_name.lower().strip()
    if value == "auto":
        return "auto"
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported hf_torch_dtype '{dtype_name}'. Use auto|fp16|bf16|fp32.")


class HFNeocortexAdapter(NeocortexAdapter):
    """Hugging Face adapter with split-layer injection for decoder-only CausalLMs."""

    _warned_legacy_once = False

    _SUPPORTED_LAYER_PATHS = (
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    )

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if not cfg.hf_model_name:
            raise ValueError("For model.provider=hf, you must set model.hf_model_name.")

        try:
            from transformers import AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover - covered by runtime smoke
            raise RuntimeError(
                "transformers is required for model.provider=hf. Install dependencies with `uv sync`."
            ) from exc

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.hf_trust_remote_code,
            "local_files_only": cfg.hf_local_files_only,
            "torch_dtype": _resolve_torch_dtype(cfg.hf_torch_dtype),
        }
        if cfg.hf_revision is not None:
            model_kwargs["revision"] = cfg.hf_revision
        if cfg.hf_attn_implementation is not None:
            model_kwargs["attn_implementation"] = cfg.hf_attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(cfg.hf_model_name, **model_kwargs)
        if cfg.hf_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        hf_hidden_size = getattr(self.model.config, "hidden_size", None)
        if hf_hidden_size is None:
            hf_hidden_size = getattr(self.model.config, "n_embd", None)
        if hf_hidden_size is None:
            raise ValueError("Unable to infer hidden size from HF model config.")

        hf_vocab_size = int(getattr(self.model.config, "vocab_size"))
        self.hidden_size = int(hf_hidden_size)
        # Keep memory bounded in this MVP by allowing active vocab truncation via config.
        self.vocab_size = int(min(cfg.vocab_size, hf_vocab_size))
        self._hf_vocab_size = hf_vocab_size

        self.hf_injection_fraction = float(cfg.hf_injection_fraction)
        self.hf_allow_legacy_injection_fallback = bool(cfg.hf_allow_legacy_injection_fallback)

        self.hf_injection_mode = "legacy_head_only"
        self.hf_injection_layer_idx = -1
        self.hf_num_decoder_layers = 0
        self.hf_decoder_layer_path: str | None = None

        self._split_layer: nn.Module | None = None
        self._cached_input_ids: torch.Tensor | None = None

        self._initialize_decoder_split()

    def _trim_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] > self.vocab_size:
            return logits[..., : self.vocab_size]
        return logits

    @staticmethod
    def _get_attr_path(root: Any, path: str) -> Any | None:
        current = root
        for part in path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    def _resolve_decoder_layers(self) -> tuple[nn.ModuleList | list[nn.Module] | None, str | None]:
        for path in self._SUPPORTED_LAYER_PATHS:
            candidate = self._get_attr_path(self.model, path)
            if isinstance(candidate, nn.ModuleList):
                return candidate, path
            if isinstance(candidate, list) and candidate and all(isinstance(m, nn.Module) for m in candidate):
                return candidate, path
        return None, None

    @staticmethod
    def _clamp_split_layers(num_layers: int, fraction: float) -> int:
        target = int(round(num_layers * fraction))
        return max(1, min(num_layers - 1, target))

    @classmethod
    def _warn_legacy_once(cls, message: str) -> None:
        if cls._warned_legacy_once:
            return
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        cls._warned_legacy_once = True

    def _initialize_decoder_split(self) -> None:
        decoder_layers, layer_path = self._resolve_decoder_layers()
        if decoder_layers is None or len(decoder_layers) < 2:
            if not self.hf_allow_legacy_injection_fallback:
                raise ValueError(
                    "Unable to resolve supported decoder stack for HF split injection and "
                    "model.hf_allow_legacy_injection_fallback=false."
                )
            self._warn_legacy_once(
                "Falling back to legacy HF injection mode (final hidden -> lm_head). "
                "Set model.hf_allow_legacy_injection_fallback=false to fail hard."
            )
            return

        self.hf_num_decoder_layers = int(len(decoder_layers))
        pre_layers = self._clamp_split_layers(self.hf_num_decoder_layers, self.hf_injection_fraction)
        self.hf_injection_layer_idx = int(pre_layers - 1)
        self._split_layer = decoder_layers[self.hf_injection_layer_idx]
        self.hf_injection_mode = "decoder_split"
        self.hf_decoder_layer_path = layer_path

    @staticmethod
    def _replace_layer_output(output: Any, hidden: torch.Tensor) -> Any:
        if isinstance(output, torch.Tensor):
            return hidden.to(device=output.device, dtype=output.dtype)

        if isinstance(output, tuple):
            if len(output) == 0:
                return output
            first = output[0]
            if not isinstance(first, torch.Tensor):
                raise ValueError("Expected tensor as first tuple item in decoder layer output.")
            injected = hidden.to(device=first.device, dtype=first.dtype)
            return (injected,) + output[1:]

        if isinstance(output, list):
            if len(output) == 0:
                return output
            first = output[0]
            if not isinstance(first, torch.Tensor):
                raise ValueError("Expected tensor as first list item in decoder layer output.")
            injected = hidden.to(device=first.device, dtype=first.dtype)
            return [injected, *output[1:]]

        raise ValueError(f"Unsupported decoder layer output type for injection: {type(output)!r}")

    def forward_to_injection(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._cached_input_ids = input_ids
        out = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        if self.hf_injection_mode == "decoder_split":
            hidden_states = out.hidden_states
            if hidden_states is None:
                raise ValueError("HF model did not return hidden states with output_hidden_states=True.")
            state_idx = self.hf_injection_layer_idx + 1
            if state_idx >= len(hidden_states):
                raise ValueError(
                    f"Injection hidden state index {state_idx} out of range for hidden_states length {len(hidden_states)}."
                )
            return hidden_states[state_idx]

        return out.hidden_states[-1]

    def forward_from_injection(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.hf_injection_mode == "decoder_split":
            if self._split_layer is None or self._cached_input_ids is None:
                raise ValueError("forward_from_injection called before forward_to_injection in decoder_split mode.")

            handle = self._split_layer.register_forward_hook(
                lambda _module, _inputs, output: self._replace_layer_output(output, hidden)
            )
            try:
                out = self.model(
                    input_ids=self._cached_input_ids,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                handle.remove()
                self._cached_input_ids = None
            return self._trim_logits(out.logits)

        lm_head = self.model.get_output_embeddings()
        if lm_head is None and hasattr(self.model, "lm_head"):
            lm_head = self.model.lm_head
        if lm_head is None:
            raise ValueError("HF model has no output embedding head for logits.")
        logits = lm_head(hidden)
        return self._trim_logits(logits)

    def forward_base(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        return self._trim_logits(out.logits)

    def runtime_info(self) -> dict[str, int | float | str]:
        return {
            "hf_injection_mode": self.hf_injection_mode,
            "hf_injection_layer_idx": int(self.hf_injection_layer_idx),
            "hf_num_decoder_layers": int(self.hf_num_decoder_layers),
            "hf_injection_fraction": float(self.hf_injection_fraction),
        }
