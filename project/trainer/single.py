#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/trainer/early_fusion copy 2.py
Project: /workspace/project/project/trainer
Created Date: Friday January 31st 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 31st 2025 7:31:19 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from project.models.make_model import single

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

import matplotlib.pyplot as plt 
import seaborn as sns
import torchmetrics

class SingleTrainer(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model
        self.img_size = hparams.data.img_size

        self.lr=hparams.optimizer.lr
        self.num_class = hparams.model.model_class_num
        self.uniform_temporal_subsample_num = hparams.data.uniform_temporal_subsample_num

        self.fusion_method = hparams.train.experiment
        
        if self.fusion_method == 'slow_fusion':
            self.model = MakeVideoModule(hparams)

            # select the network structure 
            if self.model_type == 'resnet':
                self.model=self.model.make_walk_resnet()

        elif self.fusion_method == 'early_fusion':
            self.model = early_fusion(hparams)
        elif self.fusion_method == 'late_fusion':
            self.model = late_fusion(hparams)
        # else:
        #     raise ValueError('no choiced model selected, get {self.fusion_method}')

        # self.transfor_learning = hparams.transfor_learning

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # select the metrics
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
        
        # input and label
        self._check_info(batch=batch)
        label = batch['a']['label']
        
        # input and label
        video_a = batch['a']['video'] # b, c, t, h, w
        video_b = batch['b']['video'] # b, c, t, h, w
        
        fusion_video = torch.cat([video_a, video_b], dim=2) # b, t, c, h, w 
        fusion_video = fusion_video.transpose(2, 1) # b, t, c, h, w > b, c, t, h, w 

        if self.fusion_method == 'single': 
            # for single frame
            label = batch['label'].detach()

            # when batch > 1, for multi label, to repeat label in (bxt)
            label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        else:
            label = batch['a']['label'] # b, class_num

        # classification task
        y_hat = self.model(fusion_video)

        # when torch.size([1]), not squeeze.
        if y_hat.size()[0] != 1 or len(y_hat.size()) != 1 :
            y_hat = y_hat.squeeze(dim=-1)
            
            y_hat_sigmoid = torch.sigmoid(y_hat)
        
        else:
            y_hat_sigmoid = torch.sigmoid(y_hat)

        loss = F.binary_cross_entropy_with_logits(y_hat, label.float())
        # soft_margin_loss = F.soft_margin_loss(y_hat_sigmoid, label.float())

        accuracy = self._accuracy(y_hat_sigmoid, label)
        precision = self._precision(y_hat_sigmoid, label)

        self.log_dict({'train_loss': loss, 'train_acc': accuracy, 'train_precision': precision}, on_step=True, on_epoch=True)

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     '''
    #     after validattion_step end.

    #     Args:
    #         outputs (list): a list of the train_step return value.
    #     '''
        
    #     # log epoch metric
    #     # self.log('train_acc_epoch', self.accuracy)
    #     pass

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
        label = batch['a']['label']

        # input and label
        video_a = batch['a']['video'] # b, t, c, h, w
        video_b = batch['b']['video'] # b, t, c, h, w

        # TODO: fuse the video data with lat and ap.
        fusion_video = torch.cat([video_a, video_b], dim=2) # b, t, c, h, w 
        fusion_video = fusion_video.transpose(2, 1) # b, t, c, h, w > b, c, t, h, w 
        self.model.eval()
        
        # pred the video frames
        with torch.no_grad():
            preds = self.model(fusion_video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1 :
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        val_f1 = self.f1_score(preds_sigmoid, label)
        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val_f1:': val_f1, 'val_loss': val_loss, 'val_acc': accuracy, 'val_precision': precision}, on_step=True, on_epoch=True)
        
        return accuracy

    def validation_epoch_end(self, outputs):
        pass
        
        # val_metric = torch.stack(outputs, dim=0)

        # final_acc = (torch.sum(val_metric) / len(val_metric)).item()

        # print('Epoch: %s, avgAcc: %s' % (self.current_epoch, final_acc))

        # self.ACC[self.current_epoch] = final_acc

    def on_validation_end(self) -> None:
        pass
            
    def test_step(self, batch, batch_idx):
        '''
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        '''

        # input and label
        video = batch['video'].detach() # b, c, t, h, w

        if self.fusion_method == 'single': 
            label = batch['label'].detach()

            # when batch > 1, for multi label, to repeat label in (bxt)
            label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        else:
            label = batch['label'].detach() # b, class_num

        self.model.eval()

        # pred the video frames
        with torch.no_grad():
            preds = self.model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1 :
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'test_loss': val_loss, 'test_acc': accuracy, 'test_precision': precision}, on_step=False, on_epoch=True)

        return {
            'pred': preds_sigmoid.tolist(),
            'label': label.tolist()
        }
        
    def test_epoch_end(self, outputs):
        #todo try to store the pred or confusion matrix
        pred_list = []
        label_list = []

        for i in outputs:
            for number in i['pred']:
                if number > 0.5:
                    pred_list.append(1)
                else:
                    pred_list.append(0)
            for number in i['label']:
                label_list.append(number)

        pred = torch.tensor(pred_list)
        label = torch.tensor(label_list)

        cm = confusion_matrix(label, pred)
        ax = sns.heatmap(cm, annot=True, fmt="3d")

        ax.set_title('confusion matrix')
        ax.set(xlabel="pred class", ylabel="ground truth")
        ax.xaxis.tick_top()
        plt.show()
        plt.savefig('test.png')

        return cm 

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
                "monitor": "val_loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type
    
    def _check_info(self, batch: dict) -> None:
        """check the info from data loader.
        because prepare the two data, lat and ap, so should check the video name, tensor shape and label, first

        Args:
            batch (dict): the dataloader info, include dict['a'[info], 'b'[info]]
        """        

        # * Check the info from two date 
        video_info_a = batch['a']
        video_info_b = batch['b']
        
        # print('vedeoname_a', video_info_a["video_name"])
        # print('vedeoname_b', video_info_b["video_name"])
        
        # * check tensor shape 
        video_a = video_info_a["video"]
        video_b = video_info_b["video"]
        
        assert video_a.shape == video_b.shape

        # * check label 
        label_a = video_info_a["label"]
        label_b = video_info_b["label"]
        
        # print('label_a:', label_a)
        # print('label_b:', label_b)

        assert label_a.shape ==  label_b.shape 
        
        for i in range(len(label_a)):
            assert label_a[i] == label_b[i]
        
        # * check video name 
        video_name_a = video_info_a["video_name"]
        video_name_b = video_info_b["video_name"]

        assert len(video_name_a) == len(video_name_b)
        
        for i in range(len(video_name_a)):
            assert video_name_a[i] == video_name_b[i] 
        