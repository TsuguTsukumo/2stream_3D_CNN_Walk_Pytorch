from __future__ import annotations

from dataclasses import dataclass

from pytorch_lightning import LightningDataModule, LightningModule

try:
    from project.dataloader.data_loader import WalkDataModule
    from project.dataloader.data_loader_multi import MultiData
    from project.trainer.early_fusion import EarlyFusionTrainer
    from project.trainer.late_fusion import LateFusionTrainer
    from project.trainer.single import SingleTrainer
    from project.trainer.slow_fusion import SlowFusionTrainer
except ModuleNotFoundError:
    from dataloader.data_loader import WalkDataModule
    from dataloader.data_loader_multi import MultiData
    from trainer.early_fusion import EarlyFusionTrainer
    from trainer.late_fusion import LateFusionTrainer
    from trainer.single import SingleTrainer
    from trainer.slow_fusion import SlowFusionTrainer


@dataclass(frozen=True)
class ExperimentSpec:
    display_name: str
    trainer_cls: type[LightningModule]
    data_module_cls: type[LightningDataModule]


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "single": ExperimentSpec(
        display_name="Single",
        trainer_cls=SingleTrainer,
        data_module_cls=WalkDataModule,
    ),
    "early_fusion": ExperimentSpec(
        display_name="Early Fusion",
        trainer_cls=EarlyFusionTrainer,
        data_module_cls=MultiData,
    ),
    "late_fusion": ExperimentSpec(
        display_name="Late Fusion",
        trainer_cls=LateFusionTrainer,
        data_module_cls=MultiData,
    ),
    "slow_fusion": ExperimentSpec(
        display_name="Slow Fusion",
        trainer_cls=SlowFusionTrainer,
        data_module_cls=MultiData,
    ),
}


def get_experiment_spec(experiment_name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[experiment_name]
    except KeyError as exc:
        available_experiments = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {available_experiments}."
        ) from exc


def create_experiment_components(
    hparams,
) -> tuple[LightningModule, LightningDataModule]:
    spec = get_experiment_spec(hparams.train.experiment)
    return spec.trainer_cls(hparams), spec.data_module_cls(hparams)
