from __future__ import annotations

import math

import torch
from torch import nn

from biollm_cls.config import ModelConfig
from biollm_cls.models.base import NeocortexAdapter


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCausalTransformer(NeocortexAdapter):
    """Small causal transformer with an explicit injection point for MoE deltas."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def _input_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        return self.drop(self.token_embed(input_ids) + self.pos_embed(pos_ids))

    def forward_to_injection(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._input_embed(input_ids)
        mask = self._causal_mask(input_ids.shape[1], input_ids.device)
        upto = min(self.cfg.injection_layer + 1, len(self.blocks))
        for idx in range(upto):
            x = self.blocks[idx](x, mask)
        return x

    def forward_from_injection(self, hidden: torch.Tensor) -> torch.Tensor:
        mask = self._causal_mask(hidden.shape[1], hidden.device)
        start = min(self.cfg.injection_layer + 1, len(self.blocks))
        x = hidden
        for idx in range(start, len(self.blocks)):
            x = self.blocks[idx](x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward_base(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_to_injection(input_ids)
        return self.forward_from_injection(hidden)

    def forward_with_delta(self, input_ids: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_to_injection(input_ids)
        return self.forward_from_injection(hidden + delta)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def estimate_flops_per_token(self) -> int:
        h = self.cfg.hidden_size
        l = self.cfg.num_layers
        return int(l * (12 * h * h))
