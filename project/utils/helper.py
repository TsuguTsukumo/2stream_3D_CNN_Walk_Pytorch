#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/utils/helper.py
Project: /workspace/project/project/utils
Created Date: Friday January 31st 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday February 1st 2025 8:45:45 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryConfusionMatrix,
)

def save_inference(all_pred: list, all_label: list, fold: str, save_path: str):
    """save the inference results to .pt file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        fold (str): fold number.
        save_path (str): save path.
    """       

    pred = torch.tensor(all_pred)
    label = torch.tensor(all_label)

    # save the results
    save_path = Path(save_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path} / {fold}"
    )

def save_metrics(all_pred: list, all_label: list, fold: str, save_path: str, num_class: int):
    """save the metrics to .txt file.

    Args:
        all_pred (list): all the predict result.
        all_label (list): all the label result.
        fold (str): the fold number.
        save_path (str): the path to save the metrics.
        num_class (int): number of class.
    """    

    save_path = Path(save_path) / "metrics.txt"
    all_pred = torch.tensor(all_pred)
    all_label = torch.tensor(all_label)

    _accuracy = BinaryAccuracy()
    _precision = BinaryPrecision()
    _recall = BinaryRecall()
    _f1_score = BinaryF1Score()
    _auroc = BinaryAUROC()
    _confusion_matrix = BinaryConfusionMatrix()

    logging.info("*" * 100)
    logging.info("accuracy: %s" % _accuracy(all_pred, all_label))
    logging.info("precision: %s" % _precision(all_pred, all_label))
    logging.info("recall: %s" % _recall(all_pred, all_label))
    logging.info("f1_score: %s" % _f1_score(all_pred, all_label))
    logging.info("aurroc: %s" % _auroc(all_pred, all_label.long()))
    logging.info("confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))
    logging.info("#" * 100)

    with open(save_path, "a") as f:
        f.writelines(f"Fold {fold}\n")
        f.writelines(f"accuracy: {_accuracy(all_pred, all_label)}\n")
        f.writelines(f"precision: {_precision(all_pred, all_label)}\n")
        f.writelines(f"recall: {_recall(all_pred, all_label)}\n")
        f.writelines(f"f1_score: {_f1_score(all_pred, all_label)}\n")
        f.writelines(f"aurroc: {_auroc(all_pred, all_label.long())}\n")
        f.writelines(f"confusion_matrix: {_confusion_matrix(all_pred, all_label)}\n")
        f.writelines("#" * 100)
        f.writelines("\n")

def save_CM(all_pred: list, all_label: list, save_path: str, num_class: int, fold: str):
    """save the confusion matrix to file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        save_path (Path): the path to save the confusion matrix.
        num_class (int): the number of class.
        fold (str): the fold number.
    """    

    save_path = Path(save_path) / "CM"
    all_pred = torch.Tensor(all_pred)
    all_label = torch.Tensor(all_label)

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    _confusion_matrix = BinaryConfusionMatrix()

    logging.info("_confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))

    # set the font and title
    plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})

    confusion_matrix_data = _confusion_matrix(all_pred, all_label).cpu().numpy() / 10

    axis_labels = ["ASD", "non-ASD",]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=100,
    )
    plt.title(f"{fold} (%)", fontsize=30)
    plt.ylabel("Actual Label", fontsize=30)
    plt.xlabel("Predicted Label", fontsize=30)

    plt.savefig(
        save_path / f"{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    logging.info(
        f"save the confusion matrix into {save_path}/fold{fold}_confusion_matrix.png"
    )