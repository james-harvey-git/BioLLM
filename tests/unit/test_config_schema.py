from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from biollm_cls.config_schema import validate_hydra_cfg


def test_hydra_compose_and_pydantic_validate() -> None:
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="config")
    validated = validate_hydra_cfg(cfg)
    assert validated.model.hidden_size == 256


def test_unknown_key_rejected() -> None:
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="config")
    as_dict = OmegaConf.to_container(cfg, resolve=True)
    as_dict["train"]["unknown_field"] = 123

    with pytest.raises(ValidationError):
        validate_hydra_cfg(OmegaConf.create(as_dict))


def test_rtx4070_preset_composes_and_validates() -> None:
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="config", overrides=["train=rtx4070", "device=cuda"])
    validated = validate_hydra_cfg(cfg)
    assert validated.train.batch_size == 24
    assert validated.device == "cuda"


def test_instruction_benchmark_preset_composes_and_validates() -> None:
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="config", overrides=["benchmark=instruction", "model=hf_causal_lm"])
    validated = validate_hydra_cfg(cfg)
    assert validated.benchmark.name == "instruction"
    assert validated.model.provider == "hf"
    assert validated.benchmark.eval_local_path is None
    assert validated.model.hf_injection_fraction == 0.7
    assert validated.model.hf_allow_legacy_injection_fallback is True


def test_qwen_0_5b_12gb_train_preset_composes_and_validates() -> None:
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="config", overrides=["train=qwen_0_5b_12gb", "device=cuda"])
    validated = validate_hydra_cfg(cfg)
    assert validated.train.batch_size == 2
    assert validated.train.amp_enabled is True
    assert validated.device == "cuda"


def test_ni8_pilot_and_continual_preset_composes() -> None:
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "benchmark=ni8_pilot",
                "train=continual_pilot_4070",
                "model=hf_causal_lm",
            ],
        )
    validated = validate_hydra_cfg(cfg)
    assert validated.benchmark.eval_local_path == "data/ni8/eval.jsonl"
    assert validated.benchmark.task_selection_seed == 42
    assert validated.train.ablation_disable_hippocampus is False
    assert validated.consolidation.pseudo_ratio == 0.25
    assert validated.consolidation.fisher_use_capability_mix is True
