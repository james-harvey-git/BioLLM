from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

class MetricsLogger:
    def __init__(
        self,
        output_dir: str,
        use_wandb: bool,
        wandb_project: str,
        wandb_mode: str,
        config: dict[str, Any],
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_group: str | None = None,
        wandb_job_type: str = "train",
        wandb_tags: list[str] | None = None,
        wandb_notes: str | None = None,
        wandb_resume: str | None = None,
        wandb_run_id: str | None = None,
        wandb_api_key_env_var: str = "WANDB_API_KEY",
        checkpoint_keep_last: int = 3,
        upload_checkpoints: bool = True,
        upload_config_artifact: bool = True,
        upload_metadata_artifact: bool = True,
    ) -> None:
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_path / "metrics.jsonl"
        self._fp = self.metrics_file.open("a", encoding="utf-8")
        self.upload_checkpoints = upload_checkpoints
        self.upload_config_artifact = upload_config_artifact
        self.upload_metadata_artifact = upload_metadata_artifact
        self.checkpoint_keep_last = checkpoint_keep_last

        self._wandb_module = None
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb_module = wandb
                mode = wandb_mode
                if mode != "offline" and not os.environ.get(wandb_api_key_env_var):
                    mode = "offline"
                    self._append_warning(
                        f"{wandb_api_key_env_var} is not set, falling back to offline W&B mode."
                    )

                self._wandb = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    mode=mode,
                    config=config,
                    name=wandb_run_name,
                    group=wandb_group,
                    job_type=wandb_job_type,
                    tags=wandb_tags,
                    notes=wandb_notes,
                    id=wandb_run_id,
                    resume=wandb_resume,
                )
                self._wandb.define_metric("step")
                self._wandb.define_metric("*", step_metric="step")
            except Exception as exc:
                self._append_warning(f"W&B init failed; continuing without W&B. Error: {exc}")
                self._wandb = None

    def _append_warning(self, msg: str) -> None:
        warning_file = self.output_path / "wandb_warnings.log"
        with warning_file.open("a", encoding="utf-8") as fp:
            fp.write(msg + "\n")

    def watch(self, model: torch.nn.Module, log_freq: int = 100) -> None:
        if self._wandb is None:
            return
        try:
            self._wandb.watch(model, log="gradients", log_freq=log_freq)
        except Exception as exc:
            self._append_warning(f"W&B watch failed: {exc}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        payload = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        self._fp.write(json.dumps(payload) + "\n")
        self._fp.flush()
        if self._wandb is not None:
            self._wandb.log(payload)

    def log_summary(self, summary: dict[str, float]) -> None:
        if self._wandb is None:
            return
        for key, value in summary.items():
            self._wandb.summary[key] = float(value)

    def log_artifact_file(
        self,
        path: str | Path,
        artifact_name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
    ) -> None:
        file_path = Path(path)
        if self._wandb is None or not file_path.exists():
            return
        aliases = aliases or ["latest"]
        try:
            artifact = self._wandb_module.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(str(file_path), name=file_path.name)
            self._wandb.log_artifact(artifact, aliases=aliases)
        except Exception as exc:
            self._append_warning(f"W&B artifact upload failed for {file_path}: {exc}")

    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        hippocampus: torch.nn.Module,
        base_optimizer: torch.optim.Optimizer,
        fast_optimizer: torch.optim.Optimizer,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        ckpt_path = self.checkpoint_dir / f"step_{step:07d}.pt"
        payload: dict[str, Any] = {
            "step": step,
            "base_model": model.state_dict(),
            "hippocampus": hippocampus.state_dict(),
            "base_optimizer": base_optimizer.state_dict(),
            "fast_optimizer": fast_optimizer.state_dict(),
        }
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, ckpt_path)

        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        if self.checkpoint_keep_last > 0 and len(checkpoints) > self.checkpoint_keep_last:
            for old in checkpoints[: len(checkpoints) - self.checkpoint_keep_last]:
                old.unlink(missing_ok=True)

        if self._wandb is not None and self.upload_checkpoints:
            run_id = getattr(self._wandb, "id", "run")
            self.log_artifact_file(
                ckpt_path,
                artifact_name=f"{run_id}-checkpoint",
                artifact_type="model",
                aliases=["latest", f"step-{step}"],
            )
        return ckpt_path

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            if self._wandb is not None:
                self._wandb.finish()
