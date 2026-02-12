from __future__ import annotations

from biollm_cls.benchmarks.base import ContinualBenchmark
from biollm_cls.benchmarks.continual_instruction import ContinualInstructionBenchmark
from biollm_cls.benchmarks.continual_toy import ContinualToyBenchmark
from biollm_cls.config import BenchmarkConfig, ModelConfig


def build_benchmark(
    benchmark_cfg: BenchmarkConfig,
    model_cfg: ModelConfig,
    runtime_vocab_size: int,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> ContinualBenchmark:
    name = benchmark_cfg.name.lower().strip()
    if name == "toy":
        return ContinualToyBenchmark(
            vocab_size=runtime_vocab_size,
            seq_len=seq_len,
            batch_size=batch_size,
            switch_every=benchmark_cfg.switch_every,
            num_tasks=benchmark_cfg.num_tasks,
            heldout_size=benchmark_cfg.heldout_size,
            seed=seed,
        )

    if name == "instruction":
        if model_cfg.provider.lower().strip() not in {"hf", "huggingface"}:
            raise ValueError("benchmark.name=instruction currently requires model.provider=hf.")

        return ContinualInstructionBenchmark(
            seq_len=seq_len,
            batch_size=batch_size,
            runtime_vocab_size=runtime_vocab_size,
            model_hf_name=model_cfg.hf_model_name or "",
            switch_every=benchmark_cfg.switch_every,
            dataset_name=benchmark_cfg.dataset_name,
            dataset_config=benchmark_cfg.dataset_config,
            dataset_split=benchmark_cfg.dataset_split,
            local_path=benchmark_cfg.local_path,
            eval_local_path=benchmark_cfg.eval_local_path,
            prompt_field=benchmark_cfg.prompt_field,
            response_field=benchmark_cfg.response_field,
            task_field=benchmark_cfg.task_field,
            task_selection_seed=benchmark_cfg.task_selection_seed,
            num_tasks=benchmark_cfg.num_tasks,
            max_examples=benchmark_cfg.max_examples,
            min_examples_per_task=benchmark_cfg.min_examples_per_task,
            heldout_per_task=benchmark_cfg.heldout_per_task,
            tokenizer_name=benchmark_cfg.tokenizer_name,
            tokenizer_trust_remote_code=benchmark_cfg.tokenizer_trust_remote_code,
            ignore_prompt_loss=benchmark_cfg.ignore_prompt_loss,
            prompt_template=benchmark_cfg.prompt_template,
            enforce_full_vocab=benchmark_cfg.enforce_full_vocab,
            seed=seed,
        )

    raise ValueError(f"Unsupported benchmark.name '{benchmark_cfg.name}'. Use toy or instruction.")
