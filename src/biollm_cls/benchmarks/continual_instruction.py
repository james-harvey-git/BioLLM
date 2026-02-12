from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from biollm_cls.benchmarks.base import BenchmarkBatch
from biollm_cls.models.base import NeocortexAdapter


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor


class ContinualInstructionBenchmark:
    """Continual-learning benchmark backed by instruction-response examples."""

    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        runtime_vocab_size: int,
        model_hf_name: str,
        switch_every: int = 200,
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        dataset_split: str = "train",
        local_path: str | None = None,
        prompt_field: str = "instruction",
        response_field: str = "output",
        task_field: str | None = None,
        num_tasks: int = 4,
        max_examples: int = 0,
        min_examples_per_task: int = 32,
        heldout_per_task: int = 32,
        tokenizer_name: str | None = None,
        tokenizer_trust_remote_code: bool = False,
        ignore_prompt_loss: bool = True,
        prompt_template: str = "### Instruction:\n{prompt}\n\n### Response:\n",
        enforce_full_vocab: bool = True,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.runtime_vocab_size = runtime_vocab_size
        self.switch_every = switch_every
        self.num_tasks = num_tasks
        self._rng = random.Random(seed)
        self.ignore_prompt_loss = ignore_prompt_loss
        self.prompt_template = prompt_template

        if not model_hf_name and not tokenizer_name:
            raise ValueError("Instruction benchmark requires model_hf_name or tokenizer_name.")

        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers is required for instruction benchmark.") from exc

        tok_name = tokenizer_name or model_hf_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=tokenizer_trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        tok_vocab = int(self.tokenizer.vocab_size)
        if enforce_full_vocab and runtime_vocab_size < tok_vocab:
            raise ValueError(
                f"runtime vocab_size ({runtime_vocab_size}) is smaller than tokenizer vocab ({tok_vocab}). "
                "Set model.vocab_size high enough or disable benchmark.enforce_full_vocab."
            )

        records = self._load_records(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            local_path=local_path,
            max_examples=max_examples,
        )
        task_buckets = self._bucket_by_task(
            records=records,
            prompt_field=prompt_field,
            response_field=response_field,
            task_field=task_field,
            num_tasks=num_tasks,
        )

        self.task_ids = sorted(task_buckets.keys())
        if not self.task_ids:
            raise ValueError("No valid instruction data found after preprocessing.")

        self.train_by_task: dict[int, list[EncodedExample]] = {}
        self.heldout_by_task: dict[int, list[EncodedExample]] = {}
        for task_id, rows in task_buckets.items():
            self._rng.shuffle(rows)
            if len(rows) < max(2, min_examples_per_task):
                continue

            holdout_n = min(heldout_per_task, max(1, len(rows) // 5))
            if len(rows) - holdout_n <= 0:
                continue

            heldout_rows = rows[:holdout_n]
            train_rows = rows[holdout_n:]
            if len(train_rows) < 1:
                continue

            self.train_by_task[task_id] = [self._encode(prompt, response) for prompt, response in train_rows]
            self.heldout_by_task[task_id] = [self._encode(prompt, response) for prompt, response in heldout_rows]

        self.task_ids = sorted(self.train_by_task.keys())
        if not self.task_ids:
            raise ValueError("No tasks with enough train/heldout data. Reduce min_examples_per_task.")
        self._task_count = len(self.task_ids)

    def _load_records(
        self,
        dataset_name: str | None,
        dataset_config: str | None,
        dataset_split: str,
        local_path: str | None,
        max_examples: int,
    ) -> list[dict[str, Any]]:
        if local_path:
            return self._load_local_records(local_path, max_examples=max_examples)
        if dataset_name:
            try:
                from datasets import load_dataset
            except Exception as exc:
                raise RuntimeError(
                    "datasets package is required when benchmark.dataset_name is set. "
                    "Install dependencies with uv sync."
                ) from exc
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
            out: list[dict[str, Any]] = []
            for idx, row in enumerate(ds):
                out.append(dict(row))
                if max_examples > 0 and idx + 1 >= max_examples:
                    break
            return out
        raise ValueError("Instruction benchmark requires either benchmark.local_path or benchmark.dataset_name.")

    @staticmethod
    def _load_local_records(local_path: str, max_examples: int) -> list[dict[str, Any]]:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Instruction dataset path not found: {local_path}")

        out: list[dict[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fp:
                for idx, line in enumerate(fp):
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                    if max_examples > 0 and idx + 1 >= max_examples:
                        break
            return out

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for idx, row in enumerate(payload):
                    out.append(dict(row))
                    if max_examples > 0 and idx + 1 >= max_examples:
                        break
                return out
            raise ValueError("JSON file must contain a list of records.")

        raise ValueError("Unsupported local dataset file type. Use .jsonl or .json")

    def _bucket_by_task(
        self,
        records: list[dict[str, Any]],
        prompt_field: str,
        response_field: str,
        task_field: str | None,
        num_tasks: int,
    ) -> dict[int, list[tuple[str, str]]]:
        buckets: dict[int, list[tuple[str, str]]] = {}
        for rec in records:
            if prompt_field not in rec or response_field not in rec:
                continue
            prompt = str(rec[prompt_field]).strip()
            response = str(rec[response_field]).strip()
            if not prompt or not response:
                continue

            if task_field and task_field in rec:
                raw_task = str(rec[task_field])
                task_id = abs(hash(raw_task)) % max(1, num_tasks)
            else:
                task_id = abs(hash(prompt)) % max(1, num_tasks)

            buckets.setdefault(task_id, []).append((prompt, response))
        return buckets

    def _encode(self, prompt: str, response: str) -> EncodedExample:
        prefix = self.prompt_template.format(prompt=prompt)
        full_text = prefix + response

        prefix_ids = self.tokenizer(prefix, add_special_tokens=True, truncation=True, max_length=self.seq_len + 1)[
            "input_ids"
        ]
        full_ids = self.tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.seq_len + 1)[
            "input_ids"
        ]

        if len(full_ids) < 2:
            eos_id = self.tokenizer.eos_token_id
            if eos_id is None:
                eos_id = self.tokenizer.pad_token_id
            full_ids = [full_ids[0], eos_id]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        prompt_len = max(0, min(len(prefix_ids), len(full_ids)) - 1)
        if self.ignore_prompt_loss and prompt_len > 0:
            for i in range(prompt_len):
                target_ids[i] = -100

        pad_id = int(self.tokenizer.pad_token_id)
        if len(input_ids) < self.seq_len:
            pad_n = self.seq_len - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_n
            target_ids = target_ids + [-100] * pad_n
        else:
            input_ids = input_ids[: self.seq_len]
            target_ids = target_ids[: self.seq_len]

        # If benchmark vocab is smaller than tokenizer vocab and strict mode is off,
        # remap ids into the active vocab for compatibility with runtime heads.
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        if self.runtime_vocab_size > 0:
            input_tensor = input_tensor % self.runtime_vocab_size

        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        if self.runtime_vocab_size > 0:
            mask = target_tensor != -100
            target_tensor[mask] = target_tensor[mask] % self.runtime_vocab_size

        return EncodedExample(input_ids=input_tensor, target_ids=target_tensor)

    def current_task(self, step: int) -> int:
        return self.task_ids[(step // self.switch_every) % self._task_count]

    def sample_batch(self, step: int, batch_size: int | None = None) -> BenchmarkBatch:
        bsz = batch_size or self.batch_size
        task_id = self.current_task(step)
        rows = self.train_by_task[task_id]
        samples = [rows[self._rng.randrange(0, len(rows))] for _ in range(bsz)]
        return BenchmarkBatch(
            input_ids=torch.stack([s.input_ids for s in samples], dim=0),
            target_ids=torch.stack([s.target_ids for s in samples], dim=0),
            task_id=task_id,
        )

    def evaluate_old_loss(self, model: NeocortexAdapter, device: torch.device, current_task: int) -> float:
        old_tasks = [tid for tid in self.task_ids if tid != current_task]
        if not old_tasks:
            return 0.0

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for task_id in old_tasks:
                rows = self.heldout_by_task.get(task_id, [])
                if not rows:
                    continue
                inputs = torch.stack([r.input_ids for r in rows], dim=0).to(device)
                targets = torch.stack([r.target_ids for r in rows], dim=0).to(device)
                logits = model.forward_base(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, self.runtime_vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )
                losses.append(float(loss.item()))
        model.train()
        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))

    @staticmethod
    def reward_from_correctness(logits: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = targets != -100
            if not mask.any():
                return 0.0
            acc = (pred[mask] == targets[mask]).float().mean().item()
        return 2.0 * acc - 1.0
