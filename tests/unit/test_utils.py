from __future__ import annotations

from biollm_cls.config import (
    CLSConfig,
    ConsolidationConfig,
    EWCConfig,
    ExpertConfig,
    LoggingConfig,
    ModelConfig,
    RefreshConfig,
    ReplayConfig,
    ReproConfig,
    SchedulerConfig,
    TrainConfig,
)


def make_test_config(output_dir: str, ablation_no_sleep: bool = False, max_steps: int = 260) -> CLSConfig:
    return CLSConfig(
        seed=7,
        device="cpu",
        repro=ReproConfig(deterministic=True, save_config_snapshot=True),
        model=ModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            seq_len=16,
            dropout=0.0,
            injection_layer=1,
        ),
        experts=ExpertConfig(
            num_experts=8,
            expert_hidden=16,
            top_k=2,
            fast_lr=1e-3,
            reward_alpha=0.5,
            routing_reg_weight=0.01,
            utilization_bonus=0.2,
            capacity_penalty=0.2,
            saturation_threshold=0.6,
        ),
        replay=ReplayConfig(capacity=256, batch_size=16),
        consolidation=ConsolidationConfig(
            slow_lr=1e-5,
            wake_base_lr_scale=0.0,
            sleep_min_interval=20,
            sleep_max_interval=80,
            min_sleep_steps=5,
            max_sleep_steps=20,
            pseudo_rehearsal=False,
            refresh=RefreshConfig(kl_threshold=0.2, reset_factor=0.1),
        ),
        ewc=EWCConfig(lambda_=10.0, fisher_decay=0.95),
        scheduler=SchedulerConfig(
            saturation_weight=0.5,
            forgetting_weight=0.3,
            novelty_weight=0.2,
            pressure_threshold=0.2,
        ),
        train=TrainConfig(
            batch_size=4,
            max_steps=max_steps,
            eval_interval=20,
            ablation_no_sleep=ablation_no_sleep,
            reward_from_correctness=True,
        ),
        logging=LoggingConfig(
            use_wandb=False,
            wandb_project="test",
            wandb_mode="offline",
            output_dir=output_dir,
        ),
    )
