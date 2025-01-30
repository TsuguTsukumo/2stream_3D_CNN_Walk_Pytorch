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

from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from dataloader.data_loader import WalkDataModule
from project.train import WalkVideoClassificationLightningModule
from argparse import ArgumentParser

import pytorch_lightning
import hydra
from omegaconf import DictConfig


def get_parameters():
    """
    The parameters for the model training, can be called out via the --h menu
    """
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["resnet", "csn", "r2plus1d", "x3d", "slowfast", "c2d", "i3d"],
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--version", type=str, default="test", help="the version of logger, such data"
    )
    parser.add_argument(
        "--model_class_num", type=int, default=1, help="the class num of model"
    )
    parser.add_argument(
        "--model_depth",
        type=int,
        default=50,
        choices=[50, 101, 152],
        help="the depth of used model",
    )

    # Training setting
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="numer of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size for the dataloader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="dataloader for load video"
    )
    parser.add_argument(
        "--clip_duration", type=int, default=1, help="clip duration for the video"
    )
    parser.add_argument(
        "--uniform_temporal_subsample_num",
        type=int,
        default=8,
        help="num frame from the clip duration",
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=0,
        choices=[0, 1],
        help="the gpu number whicht to train",
    )

    # ablation experment
    # different fusion method
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="slow_fusion",
        choices=["single_frame", "early_fusion", "late_fusion", "slow_fusion"],
        help="select the different fusion method from ['single_frame', 'early_fusion', 'late_fusion']",
    )

    # Transfor_learning
    parser.add_argument(
        "--transfor_learning",
        action="store_true",
        help="if use the transformer learning",
    )
    parser.add_argument(
        "--fix_layer",
        type=str,
        default="all",
        choices=["all", "head", "stem_head", "stage_head"],
        help="select the ablation study within the choices ['all', 'head', 'stem_head', 'stage_head'].",
    )

    # TTUR
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate for optimizer"
    )
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)

    # TODO:Path,　before detection
    parser.add_argument(
        "--data_path_a",
        type=str,
        default="/workspace/data/Cross_Validation/ex_20250116_ap_organized",
        help="meta dataset path",
    )
    parser.add_argument(
        "--data_path_b",
        type=str,
        default="/workspace/data/Cross_Validation/ex_20250116_lat_organized",
        help="meta dataset path",
    )

    parser.add_argument("--split_data_path", type=str)

    # TODO: change this path, after detection
    parser.add_argument(
        "--split_pad_data_path",
        type=str,
        default="/workspace/data/Cross_Validation/ex_20250122_lat",
        help="split and pad dataset with detection method.",
    )
    parser.add_argument(
        "--seg_data_path",
        type=str,
        default="/workspace/data/Cross_Validation/ex_20250122_lat",
        help="segmentation dataset with mediapipe, with 5 fold cross validation.",
    )

    parser.add_argument(
        "--log_path", type=str, default="./logs", help="the lightning logs saved path"
    )

    # using pretrained
    parser.add_argument(
        "--pretrained_model",
        type=bool,
        default=False,
        help="if use the pretrained model for training.",
    )

    # add the parser to ther Trainer
    # parser = Trainer.add_argparse_args(parser)

    return parser.parse_known_args()


def train(hparams: DictConfig, fold: str):
    # set seed
    seed_everything(42, workers=True)

    classification_module = WalkVideoClassificationLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=str(fold),  # here should be str type.
    )

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
        patience=3,
        mode="max",
    )

    trainer = Trainer(
        devices=[
            hparams.gpu_num,
        ],
        accelerator="gpu",
        max_epochs=hparams.max_epochs,
        logger=tb_logger,
        #   log_every_n_steps=100,
        check_val_every_n_epoch=1,
        callbacks=[
            model_check_point,
            early_stopping,
        ],
        #   deterministic=True
    )

    trainer.fit(classification_module, data_module)

    trainer.test(classification_module, data_module, ckpt_path="best")


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #############
    # K Fold CV
    #############

    DATA_PATH_A = config.ap_data_path
    DATA_PATH_B = config.lat_data_path

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

        config.fold = fold

        train(config)

    print("#" * 50)
    print("different fold Acc:")
    print(store_Acc_Dict)
    print("Final avg Acc is: %s" % (sum(sum_list) / len(sum_list)))


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    init_params()
