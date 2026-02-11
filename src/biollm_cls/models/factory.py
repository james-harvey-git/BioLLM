from __future__ import annotations

from biollm_cls.config import ModelConfig
from biollm_cls.models.base import NeocortexAdapter
from biollm_cls.models.hf_neocortex import HFNeocortexAdapter
from biollm_cls.models.neocortex import TinyCausalTransformer


def build_neocortex(cfg: ModelConfig) -> NeocortexAdapter:
    provider = cfg.provider.lower().strip()
    if provider == "tiny":
        return TinyCausalTransformer(cfg)
    if provider in {"hf", "huggingface"}:
        return HFNeocortexAdapter(cfg)
    raise ValueError(f"Unsupported model.provider '{cfg.provider}'. Use tiny or hf.")

