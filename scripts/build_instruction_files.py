from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


VOWELS = set("aeiou")
WORDS = [
    "neuron",
    "memory",
    "sleep",
    "cortex",
    "hippocampus",
    "expert",
    "router",
    "reward",
    "plasticity",
    "consolidation",
    "pattern",
    "buffer",
    "signal",
    "gradient",
    "token",
    "context",
    "sequence",
    "prediction",
    "novelty",
    "capacity",
]


@dataclass
class Example:
    instruction: str
    output: str
    task: str


def _sample_words(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(WORDS) for _ in range(n)]


def make_sort_example(rng: random.Random, lo: int, hi: int) -> Example:
    nums = [rng.randint(lo, hi) for _ in range(5)]
    out = " ".join(str(x) for x in sorted(nums))
    return Example(instruction=f"Sort numbers ascending: {' '.join(str(x) for x in nums)}", output=out, task="sort")


def make_reverse_example(rng: random.Random, lo: int, hi: int) -> Example:
    n = rng.randint(lo, hi)
    token = f"item{n}"
    return Example(instruction=f"Reverse string: {token}", output=token[::-1], task="reverse")


def make_arithmetic_example(rng: random.Random, lo: int, hi: int) -> Example:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    c = rng.randint(lo, hi)
    out = a + b - c
    return Example(
        instruction=f"Compute exactly: {a} + {b} - {c}",
        output=str(out),
        task="arithmetic",
    )


def make_word_count_example(rng: random.Random, lo: int, hi: int) -> Example:
    n = rng.randint(4, 12)
    words = _sample_words(rng, n)
    phrase = " ".join(words)
    return Example(
        instruction=f"Count words in this phrase: {phrase}",
        output=str(n),
        task="word_count",
    )


def make_vowel_count_example(rng: random.Random, lo: int, hi: int) -> Example:
    n = rng.randint(4, 12)
    words = _sample_words(rng, n)
    text = " ".join(words)
    count = sum(1 for ch in text.lower() if ch in VOWELS)
    return Example(
        instruction=f"How many vowels are in: {text}",
        output=str(count),
        task="vowel_count",
    )


def make_upper_example(rng: random.Random, lo: int, hi: int) -> Example:
    n = rng.randint(2, 5)
    words = _sample_words(rng, n)
    text = " ".join(words)
    return Example(
        instruction=f"Convert to UPPERCASE: {text}",
        output=text.upper(),
        task="uppercase",
    )


def make_last_token_example(rng: random.Random, lo: int, hi: int) -> Example:
    n = rng.randint(3, 6)
    words = _sample_words(rng, n)
    text = " ".join(words)
    return Example(
        instruction=f"Return only the last token in: {text}",
        output=words[-1],
        task="last_token",
    )


def make_parity_example(rng: random.Random, lo: int, hi: int) -> Example:
    x = rng.randint(lo, hi)
    return Example(
        instruction=f"Is {x} odd or even? Reply with one word.",
        output="even" if x % 2 == 0 else "odd",
        task="parity",
    )


def generate_split(
    *,
    seed: int,
    n_per_task: int,
    lo: int,
    hi: int,
) -> list[Example]:
    rng = random.Random(seed)
    makers = [
        make_sort_example,
        make_reverse_example,
        make_arithmetic_example,
        make_word_count_example,
        make_vowel_count_example,
        make_upper_example,
        make_last_token_example,
        make_parity_example,
    ]

    rows: list[Example] = []
    for make in makers:
        for _ in range(n_per_task):
            rows.append(make(rng, lo, hi))
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(
                json.dumps(
                    {
                        "instruction": row.instruction,
                        "output": row.output,
                        "task": row.task,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local instruction train/eval files.")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--train-per-task", type=int, default=120)
    parser.add_argument("--eval-per-task", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_rows = generate_split(seed=args.seed, n_per_task=args.train_per_task, lo=0, hi=200)
    eval_rows = generate_split(seed=args.seed + 1, n_per_task=args.eval_per_task, lo=500, hi=900)

    train_path = out_dir / "instructions_train.jsonl"
    eval_path = out_dir / "instructions_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"Wrote {len(train_rows)} rows -> {train_path}")
    print(f"Wrote {len(eval_rows)} rows -> {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
