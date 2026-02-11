from __future__ import annotations

from biollm_cls.control.scheduler import SleepPressureController, StepStats


def test_scheduler_triggers_on_pressure_after_min_interval() -> None:
    s = SleepPressureController(
        saturation_weight=0.5,
        forgetting_weight=0.3,
        novelty_weight=0.2,
        pressure_threshold=0.4,
        min_interval=10,
        max_interval=100,
        min_sleep_steps=5,
        max_sleep_steps=20,
    )
    s.update(StepStats(saturation_ratio=1.0, forgetting_signal=1.0, novelty_rate=1.0))
    assert not s.should_sleep(5)
    assert s.should_sleep(12)


def test_scheduler_triggers_on_max_interval_even_low_pressure() -> None:
    s = SleepPressureController(
        saturation_weight=0.5,
        forgetting_weight=0.3,
        novelty_weight=0.2,
        pressure_threshold=0.9,
        min_interval=10,
        max_interval=30,
        min_sleep_steps=5,
        max_sleep_steps=20,
    )
    s.update(StepStats(saturation_ratio=0.0, forgetting_signal=0.0, novelty_rate=0.0))
    assert s.should_sleep(30)


def test_recommended_sleep_steps_scales_with_pressure() -> None:
    s = SleepPressureController(
        saturation_weight=1.0,
        forgetting_weight=0.0,
        novelty_weight=0.0,
        pressure_threshold=1.0,
        min_interval=1,
        max_interval=10,
        min_sleep_steps=5,
        max_sleep_steps=15,
    )
    s.update(StepStats(saturation_ratio=0.0, forgetting_signal=0.0, novelty_rate=0.0))
    low = s.recommended_sleep_steps()
    s.update(StepStats(saturation_ratio=1.0, forgetting_signal=0.0, novelty_rate=0.0))
    high = s.recommended_sleep_steps()
    assert high >= low
