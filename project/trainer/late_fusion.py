#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/trainer/early_fusion copy.py
Project: /workspace/project/project/trainer
Created Date: Friday January 31st 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 31st 2025 7:31:17 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import logging

import torch
import torch.nn.functional as F

from project.models.make_model import late_fusion

from pytorch_lightning import LightningModule

# from torchmetrics.classification import (
#     MulticlassAccuracy,
#     MulticlassPrecision,
#     MulticlassRecall,
#     MulticlassF1Score,
#     MulticlassConfusionMatrix
# )

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

from project.utils.helper import save_inference, save_metrics, save_CM

class LateFusionTrainer(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model
        self.img_size = hparams.data.img_size

        self.lr=hparams.optimizer.lr
        self.num_classes = hparams.model.model_class_num
        self.uniform_temporal_subsample_num = hparams.data.uniform_temporal_subsample_num

        self.model = late_fusion(hparams)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # select the metrics
        # self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        # self._precision = MulticlassPrecision(num_classes=self.num_classes)
        # self._recall = MulticlassRecall(num_classes=self.num_classes)
        # self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        # self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self._accuracy = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._recall = BinaryRecall()
        self._f1_score = BinaryF1Score()
        self._confusion_matrix = BinaryConfusionMatrix()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        '''
        
        # * Check the info from two date 
        self._check_info(batch=batch)

        label = batch['ap']['label']

        # input and label
        video_a = batch['ap']['video'] # b, t, c, h, w
        video_b = batch['lat']['video'] # b, t, c, h, w
        
        # * predict the video frames
        preds = self.model(video_a, video_b)
        
        # when torch.size([1]), not squeeze.
        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        train_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)
        precision = self._precision(preds_sigmoid, label)
        val_f1 = self._f1_score(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        self.log_dict({'train/f1:': val_f1, 'train/loss': train_loss, 'train/acc': accuracy, 'train/precision': precision}, on_step=True, on_epoch=True, batch_size=label.size(0))

        return train_loss

    def validation_step(self, batch, batch_idx):
        '''
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss 
            accuract: selected accuracy result.
        '''

        # * Check the info from two date 
        self._check_info(batch=batch)

        label = batch['ap']['label']

        # input and label
        video_a = batch['ap']['video'] # b, t, c, h, w
        video_b = batch['lat']['video'] # b, t, c, h, w
        
        # * predict the video frames
        with torch.no_grad():
            preds = self.model(video_a, video_b)
        
        # when torch.size([1]), not squeeze.
        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)
        precision = self._precision(preds_sigmoid, label)
        val_f1 = self._f1_score(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val/f1:': val_f1, 'val/loss': val_loss, 'val/acc': accuracy, 'val/precision': precision}, on_step=True, on_epoch=True, batch_size=label.size(0))
        
    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""
        self.test_outputs = []
        self.test_pred_list = []
        self.test_label_list = []

        logging.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logging.info("test end")

    def test_step(self, batch: torch.Tensor, batch_idx: int):

        # * Check the info from two date 
        self._check_info(batch=batch)

        label = batch['ap']['label']

        # input and label
        video_a = batch['ap']['video'] # b, t, c, h, w
        video_b = batch['lat']['video'] # b, t, c, h, w
        
        # * predict the video frames
        with torch.no_grad():
            preds = self.model(video_a, video_b)
        
        # when torch.size([1]), not squeeze.
        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        test_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)
        precision = self._precision(preds_sigmoid, label)
        fi_score = self._f1_score(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        self.log_dict(
            {
                "test/acc": accuracy,
                "test/precision": precision,
                "test/f1_score": fi_score,
            },
            on_epoch=True, on_step=True, batch_size=label.size(0)
        )

        return preds_sigmoid

    def on_test_batch_end(
        self,
        outputs: list[torch.Tensor],
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """hook function for test batch end

        Args:
            outputs (torch.Tensor | logging.Mapping[str, Any] | None): current output from batch.
            batch (Any): the data of current batch.
            batch_idx (int): the index of current batch.
            dataloader_idx (int, optional): the index of all dataloader. Defaults to 0.
        """

        perds = outputs
        label = batch['ap']['label'].float().squeeze()

        self.test_outputs.append(outputs)
        # tensor to list
        for i in perds.tolist():
            self.test_pred_list.append(i)
        for i in label.tolist():
            self.test_label_list.append(i)

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""

        # save inference
        save_inference(
            self.test_pred_list,
            self.test_label_list,
            fold=self.logger.name,
            save_path=self.hparams.hparams.train.log_path,
        )
        # save metrics
        save_metrics(
            self.test_pred_list,
            self.test_label_list,
            fold=self.logger.name,
            save_path=self.hparams.hparams.train.log_path,
            num_class=self.num_classes,
        )
        # save confusion matrix
        save_CM(
            self.test_pred_list,
            self.test_label_list,
            save_path=self.hparams.hparams.train.log_path,
            num_class=self.num_classes,
            fold=self.logger.name,
        )

        # save CAM
        # save_CAM(self.test_pred_list, self.test_label_list, self.num_classes)

        logging.info("test epoch end")


    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val/loss",
            },
        }

    def _check_info(self, batch: dict) -> None:
        """check the info from data loader.
        because prepare the two data, lat and ap, so should check the video name, tensor shape and label, first

        Args:
            batch (dict): the dataloader info, include dict['a'[info], 'b'[info]]
        """        

        # * Check the info from two date 
        video_info_a = batch['ap']
        video_info_b = batch['lat']
        
        # * check tensor shape 
        video_a = video_info_a["video"]
        video_b = video_info_b["video"]
        
        assert video_a.shape == video_b.shape

        # * check label 
        label_a = video_info_a["label"]
        label_b = video_info_b["label"]
        
        assert label_a.shape ==  label_b.shape 
        
        for i in range(len(label_a)):
            assert label_a[i] == label_b[i]
        
        # * check video name 
        video_name_a = video_info_a["video_name"]
        video_name_b = video_info_b["video_name"]

        assert len(video_name_a) == len(video_name_b)
        
        for i in range(len(video_name_a)):
            assert video_name_a[i] == video_name_b[i] 
        