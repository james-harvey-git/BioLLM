from __future__ import annotations

import json
import random
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from biollm_cls.benchmarks.base import BenchmarkBatch, TaskEvalResult
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
        eval_local_path: str | None = None,
        prompt_field: str = "instruction",
        response_field: str = "output",
        task_field: str | None = None,
        task_selection_seed: int = 42,
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

        train_records = self._load_records(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            local_path=local_path,
            max_examples=max_examples,
        )
        train_buckets = self._bucket_by_task(
            records=train_records,
            prompt_field=prompt_field,
            response_field=response_field,
            task_field=task_field,
            num_tasks=num_tasks,
        )
        if not train_buckets:
            raise ValueError("No valid instruction data found after preprocessing.")

        selected_task_labels = self._select_task_labels(
            sorted(train_buckets.keys()),
            num_tasks=num_tasks,
            task_selection_seed=task_selection_seed,
        )

        self.task_label_to_id = {label: idx for idx, label in enumerate(selected_task_labels)}
        self.task_id_to_label = {idx: label for label, idx in self.task_label_to_id.items()}

        eval_buckets: dict[str, list[tuple[str, str]]] = {}
        if eval_local_path is not None:
            eval_records = self._load_local_records(eval_local_path, max_examples=0)
            eval_buckets = self._bucket_by_task(
                records=eval_records,
                prompt_field=prompt_field,
                response_field=response_field,
                task_field=task_field,
                num_tasks=num_tasks,
            )

        self.train_by_task: dict[int, list[EncodedExample]] = {}
        self.heldout_by_task: dict[int, list[EncodedExample]] = {}

        for task_label in selected_task_labels:
            rows = list(train_buckets.get(task_label, []))
            self._rng.shuffle(rows)
            if len(rows) < max(1, min_examples_per_task):
                continue

            if eval_local_path is not None:
                train_rows = rows
                heldout_rows = list(eval_buckets.get(task_label, []))
                if not heldout_rows:
                    holdout_n = min(heldout_per_task, max(1, len(rows) // 5))
                    heldout_rows = rows[:holdout_n]
                    train_rows = rows[holdout_n:]
            else:
                holdout_n = min(heldout_per_task, max(1, len(rows) // 5))
                if len(rows) - holdout_n <= 0:
                    continue
                heldout_rows = rows[:holdout_n]
                train_rows = rows[holdout_n:]

            if not train_rows or not heldout_rows:
                continue

            task_id = self.task_label_to_id[task_label]
            self.train_by_task[task_id] = [self._encode(prompt, response) for prompt, response in train_rows]
            self.heldout_by_task[task_id] = [self._encode(prompt, response) for prompt, response in heldout_rows]

        self.task_ids = sorted(self.train_by_task.keys())
        if not self.task_ids:
            raise ValueError("No tasks with enough train/eval data. Reduce min_examples_per_task.")
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
    ) -> dict[str, list[tuple[str, str]]]:
        buckets: dict[str, list[tuple[str, str]]] = {}
        for rec in records:
            if prompt_field not in rec or response_field not in rec:
                continue

            prompt = self._normalize_text(str(rec[prompt_field]))
            response = self._normalize_text(str(rec[response_field]))
            if not prompt or not response:
                continue

            if task_field and task_field in rec:
                task_label = self._normalize_text(str(rec[task_field]))
                if not task_label:
                    continue
            else:
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                stable = int(digest[:16], 16)
                task_label = f"task_{stable % max(1, num_tasks)}"

            buckets.setdefault(task_label, []).append((prompt, response))
        return buckets

    @staticmethod
    def _select_task_labels(task_labels: list[str], num_tasks: int, task_selection_seed: int) -> list[str]:
        if not task_labels:
            return []
        sorted_labels = sorted(task_labels)
        if len(sorted_labels) <= num_tasks:
            return sorted_labels
        rng = random.Random(task_selection_seed)
        rng.shuffle(sorted_labels)
        return sorted_labels[:num_tasks]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

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

    def _evaluate_examples(
        self,
        model: NeocortexAdapter,
        device: torch.device,
        rows: list[EncodedExample],
    ) -> TaskEvalResult:
        if not rows:
            return TaskEvalResult(loss=0.0, token_acc=0.0, n_tokens=0)

        total_nll = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start : start + self.batch_size]
                inputs = torch.stack([r.input_ids for r in chunk], dim=0).to(device)
                targets = torch.stack([r.target_ids for r in chunk], dim=0).to(device)
                logits = model.forward_base(inputs)

                mask = targets != -100
                n_tokens = int(mask.sum().item())
                if n_tokens <= 0:
                    continue

                loss_sum = F.cross_entropy(
                    logits.view(-1, self.runtime_vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                preds = logits.argmax(dim=-1)
                total_nll += float(loss_sum.item())
                total_correct += int((preds[mask] == targets[mask]).sum().item())
                total_tokens += n_tokens

        if total_tokens <= 0:
            return TaskEvalResult(loss=0.0, token_acc=0.0, n_tokens=0)

        return TaskEvalResult(
            loss=float(total_nll / total_tokens),
            token_acc=float(total_correct / total_tokens),
            n_tokens=total_tokens,
        )

    def evaluate_task_set(
        self,
        model: NeocortexAdapter,
        device: torch.device,
        task_ids: list[int],
    ) -> dict[int, TaskEvalResult]:
        model.eval()
        out: dict[int, TaskEvalResult] = {}
        with torch.no_grad():
            for task_id in task_ids:
                rows = self.heldout_by_task.get(task_id, [])
                out[int(task_id)] = self._evaluate_examples(model, device, rows)
        model.train()
        return out

    def evaluate_old_loss(self, model: NeocortexAdapter, device: torch.device, current_task: int) -> float:
        old_tasks = [tid for tid in self.task_ids if tid != current_task]
        if not old_tasks:
            return 0.0

        evals = self.evaluate_task_set(model, device, old_tasks)
        losses = [r.loss for r in evals.values() if r.n_tokens > 0]
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
