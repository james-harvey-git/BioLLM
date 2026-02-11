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
