from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepStats:
    saturation_ratio: float
    forgetting_signal: float
    novelty_rate: float


class SleepPressureController:
    def __init__(
        self,
        saturation_weight: float,
        forgetting_weight: float,
        novelty_weight: float,
        pressure_threshold: float,
        min_interval: int,
        max_interval: int,
        min_sleep_steps: int,
        max_sleep_steps: int,
    ) -> None:
        self.saturation_weight = saturation_weight
        self.forgetting_weight = forgetting_weight
        self.novelty_weight = novelty_weight
        self.pressure_threshold = pressure_threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.min_sleep_steps = min_sleep_steps
        self.max_sleep_steps = max_sleep_steps

        self.current_pressure = 0.0
        self.last_sleep_step = 0

    def update(self, stats: StepStats) -> None:
        pressure = (
            self.saturation_weight * stats.saturation_ratio
            + self.forgetting_weight * stats.forgetting_signal
            + self.novelty_weight * stats.novelty_rate
        )
        self.current_pressure = max(0.0, float(pressure))

    def should_sleep(self, step: int) -> bool:
        since_last = step - self.last_sleep_step
        if since_last >= self.max_interval:
            return True
        if since_last < self.min_interval:
            return False
        return self.current_pressure >= self.pressure_threshold

    def recommended_sleep_steps(self) -> int:
        span = self.max_sleep_steps - self.min_sleep_steps
        if span <= 0:
            return self.min_sleep_steps
        ratio = min(1.0, self.current_pressure / max(self.pressure_threshold, 1e-8))
        return int(round(self.min_sleep_steps + ratio * span))

    def mark_slept(self, step: int) -> None:
        self.last_sleep_step = step
        self.current_pressure = 0.0
