"""Benchmark generators."""

from biollm_cls.benchmarks.base import BenchmarkBatch, ContinualBenchmark
from biollm_cls.benchmarks.continual_instruction import ContinualInstructionBenchmark
from biollm_cls.benchmarks.continual_toy import ContinualToyBenchmark
from biollm_cls.benchmarks.factory import build_benchmark

__all__ = [
    "BenchmarkBatch",
    "ContinualBenchmark",
    "ContinualInstructionBenchmark",
    "ContinualToyBenchmark",
    "build_benchmark",
]
