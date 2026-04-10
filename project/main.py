#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

try:
    from project.experiment_factory import (
        create_experiment_components,
        get_experiment_spec,
    )
except ModuleNotFoundError:
    from experiment_factory import create_experiment_components, get_experiment_spec


def build_logger(log_path: str, fold: str) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=log_path,
        name=fold,
    )


def build_callbacks() -> list:
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/acc_epoch",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    early_stopping = EarlyStopping(
        monitor="val/acc_epoch",
        patience=10,
        mode="max",
    )

    return [model_check_point, early_stopping]


def build_lightning_trainer(
    hparams: DictConfig,
    fold: str,
    include_callbacks: bool = True,
) -> Trainer:
    return Trainer(
        devices=hparams.device.device,
        strategy="auto",
        accelerator="gpu",
        num_sanity_val_steps=0,
        max_epochs=hparams.train.max_epochs,
        logger=build_logger(hparams.train.log_path, fold),
        check_val_every_n_epoch=1,
        callbacks=build_callbacks() if include_callbacks else [],
        fast_dev_run=hparams.train.fast_dev_run,
    )


def run_fold(hparams: DictConfig) -> None:
    seed_everything(42, workers=True)

    experiment_spec = get_experiment_spec(hparams.train.experiment)
    logging.info(experiment_spec.display_name)

    lightning_module, data_module = create_experiment_components(hparams)
    pl_trainer = build_lightning_trainer(
        hparams=hparams,
        fold=str(hparams.train.current_fold),
        include_callbacks=hparams.train.run_mode == "fit",
    )

    if hparams.train.run_mode == "fit":
        pl_trainer.fit(lightning_module, data_module)
        return

    if hparams.train.run_mode == "test":
        ckpt_path = hparams.train.ckpt_path
        if ckpt_path in (None, "", "null"):
            raise ValueError("`train.ckpt_path` must be set when `train.run_mode=test`.")

        pl_trainer.test(
            model=lightning_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
        return

    raise ValueError(f"Unsupported run_mode: {hparams.train.run_mode}")


def run_cross_validation(config: DictConfig) -> None:
    for fold_index in range(config.train.fold):
        print("#" * 50)
        print(f"Start {fold_index}")
        print("#" * 50)

        config.train.current_fold = f"fold{fold_index}"
        run_fold(config)

    print("#" * 50)


def run_single_fold(config: DictConfig) -> None:
    print("#" * 50)
    print(f"Run {config.train.current_fold}")
    print("#" * 50)
    run_fold(config)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config.yaml",
)
def init_params(config: DictConfig) -> None:
    if config.train.run_mode == "fit":
        run_cross_validation(config)
        return

    if config.train.run_mode == "test":
        run_single_fold(config)
        return

    raise ValueError(f"Unsupported run_mode: {config.train.run_mode}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
