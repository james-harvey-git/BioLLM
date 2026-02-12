from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReproConfig:
    deterministic: bool = True
    save_config_snapshot: bool = True


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    seq_len: int
    dropout: float
    injection_layer: int
    provider: str = "tiny"
    hf_model_name: str | None = None
    hf_revision: str | None = None
    hf_trust_remote_code: bool = False
    hf_torch_dtype: str = "auto"
    hf_attn_implementation: str | None = None
    hf_gradient_checkpointing: bool = False
    hf_local_files_only: bool = False


@dataclass
class ExpertConfig:
    num_experts: int
    expert_hidden: int
    top_k: int
    fast_lr: float
    reward_alpha: float
    routing_reg_weight: float
    utilization_bonus: float
    capacity_penalty: float
    saturation_threshold: float


@dataclass
class ReplayConfig:
    capacity: int
    batch_size: int


@dataclass
class RefreshConfig:
    kl_threshold: float
    reset_factor: float


@dataclass
class ConsolidationConfig:
    slow_lr: float
    wake_base_lr_scale: float
    sleep_min_interval: int
    sleep_max_interval: int
    min_sleep_steps: int
    max_sleep_steps: int
    pseudo_rehearsal: bool
    refresh: RefreshConfig


@dataclass
class EWCConfig:
    lambda_: float
    fisher_decay: float


@dataclass
class SchedulerConfig:
    saturation_weight: float
    forgetting_weight: float
    novelty_weight: float
    pressure_threshold: float


@dataclass
class BenchmarkConfig:
    name: str = "toy"
    switch_every: int = 200
    num_tasks: int = 4
    heldout_size: int = 128
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"
    local_path: str | None = None
    eval_local_path: str | None = None
    prompt_field: str = "instruction"
    response_field: str = "output"
    task_field: str | None = None
    task_selection_seed: int = 42
    max_examples: int = 0
    min_examples_per_task: int = 32
    heldout_per_task: int = 32
    tokenizer_name: str | None = None
    tokenizer_trust_remote_code: bool = False
    ignore_prompt_loss: bool = True
    prompt_template: str = "### Instruction:\n{prompt}\n\n### Response:\n"
    enforce_full_vocab: bool = True


@dataclass
class TrainConfig:
    batch_size: int
    max_steps: int
    eval_interval: int
    ablation_no_sleep: bool
    reward_from_correctness: bool
    ablation_disable_hippocampus: bool = False
    amp_enabled: bool = True
    amp_dtype: str = "fp16"
    allow_tf32: bool = True
    matmul_precision: str = "high"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"


@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb_project: str
    wandb_mode: str
    output_dir: str
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "train"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_resume: str | None = None
    wandb_run_id: str | None = None
    wandb_api_key_env_var: str = "WANDB_API_KEY"
    wandb_watch_model: bool = False
    wandb_watch_log_freq: int = 100
    checkpoint_interval: int = 200
    checkpoint_keep_last: int = 3
    upload_checkpoints: bool = True
    upload_config_artifact: bool = True
    upload_metadata_artifact: bool = True


@dataclass
class CLSConfig:
    seed: int
    device: str
    repro: ReproConfig
    model: ModelConfig
    benchmark: BenchmarkConfig
    experts: ExpertConfig
    replay: ReplayConfig
    consolidation: ConsolidationConfig
    ewc: EWCConfig
    scheduler: SchedulerConfig
    train: TrainConfig
    logging: LoggingConfig

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CLSConfig":
        return CLSConfig(
            seed=data["seed"],
            device=data["device"],
            repro=ReproConfig(**data["repro"]),
            model=ModelConfig(**data["model"]),
            benchmark=BenchmarkConfig(**data["benchmark"]),
            experts=ExpertConfig(**data["experts"]),
            replay=ReplayConfig(**data["replay"]),
            consolidation=ConsolidationConfig(
                refresh=RefreshConfig(**data["consolidation"]["refresh"]),
                **{k: v for k, v in data["consolidation"].items() if k != "refresh"},
            ),
            ewc=EWCConfig(**data["ewc"]),
            scheduler=SchedulerConfig(**data["scheduler"]),
            train=TrainConfig(**data["train"]),
            logging=LoggingConfig(**data["logging"]),
        )
