#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/main.py
Project: /workspace/project/project
Created Date: Thursday January 30th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 30th 2025 2:16:42 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import logging

import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# dataloader
from project.dataloader.data_loader import WalkDataModule
from project.dataloader.data_loader_multi import MultiData

# compare experiment
from project.trainer.late_fusion import LateFusionTrainer
from project.trainer.early_fusion import EarlyFusionTrainer
from project.trainer.slow_fusion import SlowFusionTrainer
from project.trainer.single import SingleTrainer

import hydra
from omegaconf import DictConfig


def train(hparams: DictConfig):

    fold = hparams.train.current_fold
    # set seed
    seed_everything(42, workers=True)

    if hparams.train.experiment == "late_fusion":
        logging.info("Late Fusion")
        trainer = LateFusionTrainer(hparams)

    elif hparams.train.experiment == "slow_fusion":
        logging.info("Slow Fusion")
        trainer = SlowFusionTrainer(hparams)

    elif hparams.train.experiment == "early_fusion":
        logging.info("Early Fusion")
        trainer = EarlyFusionTrainer(hparams)

    elif hparams.train.experiment == "single":
        logging.info("Single")
        trainer = SingleTrainer(hparams)

    else:
        logging.error("No such expert: %s" % hparams.train.experiment)
        assert False

    # select the data module
    if hparams.train.experiment == "single":
        data_module = WalkDataModule(hparams)

    elif hparams.train.experiment in ["late_fusion", "early_fusion", "slow_fusion"]:
        data_module = MultiData(hparams)

    else:
        logging.error("No such expert: %s" % hparams.train.experiment)
        assert False

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=str(fold),  # here should be str type.
    )

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/acc",
        patience=3,
        mode="max",
    )

    pl_trainer = Trainer(
        devices=[
            int(hparams.train.gpu_num),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,
        #   log_every_n_steps=100,
        check_val_every_n_epoch=1,
        callbacks=[
            model_check_point,
            early_stopping,
        ],
        #   deterministic=True
    )

    pl_trainer.fit(trainer, data_module)

    pl_trainer.test(trainer, data_module, ckpt_path="best")


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #############
    # K Fold CV
    #############

    DATA_PATH_A = config.data.ap_data_path
    DATA_PATH_B = config.data.lat_data_path

    # get the fold number
    print("DATA_PATH_A:", DATA_PATH_A)
    fold_num_a = os.listdir(DATA_PATH_A)
    print("DATA_PATH_B:", DATA_PATH_B)
    fold_num_b = os.listdir(DATA_PATH_B)
    fold_num_a.sort()
    fold_num_b.sort()
    print("fold_num_a:", fold_num_a)
    print("fold_num_b:", fold_num_b)

    store_Acc_Dict = {}
    sum_list = []

    for fold in fold_num_a:
        #################
        # start k Fold CV
        #################

        print("#" * 50)
        print("Strat %s" % fold)
        print("#" * 50)

        config.train.current_fold = fold

        train(config)

    print("#" * 50)
    print("different fold Acc:")
    print(store_Acc_Dict)
    print("Final avg Acc is: %s" % (sum(sum_list) / len(sum_list)))


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
