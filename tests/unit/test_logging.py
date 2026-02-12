from __future__ import annotations

from pathlib import Path

import torch

from biollm_cls.logging import MetricsLogger


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l(x)


def test_metrics_logger_local_and_checkpoint_rotation(tmp_path: Path) -> None:
    model = _Tiny()
    hip = _Tiny()
    opt_a = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt_b = torch.optim.AdamW(hip.parameters(), lr=1e-3)

    logger = MetricsLogger(
        output_dir=str(tmp_path),
        use_wandb=False,
        wandb_project="x",
        wandb_mode="offline",
        config={},
        checkpoint_keep_last=2,
        metric_glossary_rows=[
            {
                "metric": "task_loss",
                "scope": "wake-step",
                "direction": "lower",
                "unit": "cross-entropy",
                "definition": "Test metric entry.",
                "formula": "",
            }
        ],
    )
    logger.log({"a": 1.0, "mode": "mixed_replay_pseudo"}, step=1)
    logger.save_checkpoint(1, model, hip, opt_a, opt_b)
    logger.save_checkpoint(2, model, hip, opt_a, opt_b)
    logger.save_checkpoint(3, model, hip, opt_a, opt_b)
    logger.close()

    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.exists()
    assert (tmp_path / "metric_glossary.json").exists()
    assert (tmp_path / "metric_glossary.md").exists()
    ckpts = sorted((tmp_path / "checkpoints").glob("step_*.pt"))
    assert len(ckpts) == 2
