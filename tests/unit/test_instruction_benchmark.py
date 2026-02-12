from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

from biollm_cls.benchmarks.continual_instruction import ContinualInstructionBenchmark
from biollm_cls.models.base import NeocortexAdapter


class _FakeTokenizer:
    vocab_size = 64

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def add_special_tokens(self, _tokens: dict[str, str]) -> None:
        return None

    def __call__(self, text: str, add_special_tokens: bool, truncation: bool, max_length: int) -> dict[str, list[int]]:
        token_ids = [2 + (ord(c) % 31) for c in text][: max_length - 1]
        if add_special_tokens:
            token_ids = token_ids + [self.eos_token_id]
        return {"input_ids": token_ids[:max_length]}


class _FakeTransformersModule:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
            return _FakeTokenizer()


class _TinyModel(NeocortexAdapter):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward_to_injection(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)

    def forward_from_injection(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def forward_base(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_from_injection(self.forward_to_injection(input_ids))


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_instruction_benchmark_uses_eval_local_path(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformersModule())

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"

    train_rows = []
    eval_rows = []
    for task in ("task_alpha", "task_beta"):
        for i in range(6):
            train_rows.append(
                {
                    "instruction": f"train {task} #{i}",
                    "output": f"out {i}",
                    "task": task,
                }
            )
        for i in range(3):
            eval_rows.append(
                {
                    "instruction": f"eval {task} #{i}",
                    "output": f"eval_out {i}",
                    "task": task,
                }
            )

    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    benchmark = ContinualInstructionBenchmark(
        seq_len=16,
        batch_size=2,
        runtime_vocab_size=64,
        model_hf_name="fake-model",
        switch_every=5,
        local_path=str(train_path),
        eval_local_path=str(eval_path),
        prompt_field="instruction",
        response_field="output",
        task_field="task",
        num_tasks=2,
        min_examples_per_task=2,
        heldout_per_task=2,
        enforce_full_vocab=False,
        seed=11,
    )

    assert len(benchmark.task_ids) == 2
    for task_id in benchmark.task_ids:
        assert len(benchmark.train_by_task[task_id]) == 6
        assert len(benchmark.heldout_by_task[task_id]) == 3


def test_evaluate_task_set_returns_expected_fields(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformersModule())

    data_path = tmp_path / "data.jsonl"
    rows = []
    for task in ("one", "two"):
        for i in range(8):
            rows.append({"instruction": f"{task} prompt {i}", "output": f"resp {i}", "task": task})
    _write_jsonl(data_path, rows)

    benchmark = ContinualInstructionBenchmark(
        seq_len=12,
        batch_size=2,
        runtime_vocab_size=64,
        model_hf_name="fake-model",
        local_path=str(data_path),
        task_field="task",
        num_tasks=2,
        min_examples_per_task=2,
        heldout_per_task=2,
        enforce_full_vocab=False,
    )

    model = _TinyModel(vocab_size=64)
    out = benchmark.evaluate_task_set(model, torch.device("cpu"), benchmark.task_ids)

    assert sorted(out.keys()) == sorted(benchmark.task_ids)
    for value in out.values():
        assert value.n_tokens >= 0
        assert 0.0 <= value.token_acc <= 1.0
        assert value.loss >= 0.0


def test_encoding_reserves_supervised_tokens_for_long_prompts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformersModule())

    data_path = tmp_path / "long_prompt.jsonl"
    rows = []
    very_long_prompt = " ".join(["prompt"] * 200)
    for i in range(12):
        rows.append(
            {
                "instruction": f"{very_long_prompt} {i}",
                "output": "ok",
                "task": "coverage_task",
            }
        )
    _write_jsonl(data_path, rows)

    benchmark = ContinualInstructionBenchmark(
        seq_len=16,
        batch_size=2,
        runtime_vocab_size=64,
        model_hf_name="fake-model",
        local_path=str(data_path),
        task_field="task",
        num_tasks=1,
        min_examples_per_task=2,
        heldout_per_task=2,
        enforce_full_vocab=False,
    )

    assert benchmark.task_ids
    for split in (benchmark.train_by_task, benchmark.heldout_by_task):
        for task_id in benchmark.task_ids:
            for ex in split[task_id]:
                assert int((ex.target_ids != -100).sum().item()) > 0
