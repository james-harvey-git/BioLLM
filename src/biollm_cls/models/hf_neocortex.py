from __future__ import annotations

from typing import Any

import torch

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
    """
    Hugging Face adapter with a stable interface for the CLS training loop.

    Note: injection is applied at the final hidden representation before the LM head.
    """

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

    def _trim_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] > self.vocab_size:
            return logits[..., : self.vocab_size]
        return logits

    def forward_to_injection(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def forward_from_injection(self, hidden: torch.Tensor) -> torch.Tensor:
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

