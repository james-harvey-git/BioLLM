from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra

from biollm_cls.config_schema import dataclass_to_dict, validate_hydra_cfg
from biollm_cls.train import run_training


def _compose_config(overrides: list[str]):
    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def _train(overrides: list[str]) -> int:
    hydra_cfg = _compose_config(overrides)
    cfg = validate_hydra_cfg(hydra_cfg)

    out_dir = Path(cfg.logging.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validated_config.json").write_text(
        json.dumps(dataclass_to_dict(cfg), indent=2),
        encoding="utf-8",
    )

    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="BioLLM CLS MVP")
    parser.add_argument("command", choices=["train"], help="Action to run")
    args, extra = parser.parse_known_args(argv)

    if args.command == "train":
        return _train(extra)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
