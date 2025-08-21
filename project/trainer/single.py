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

        self.model = single(hparams)
        
        self.save_hyperparameters()

        # select the metrics
        self._accuracy = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._recall = BinaryRecall()
        self._f1_score = BinaryF1Score()
        self._confusion_matrix = BinaryConfusionMatrix()
        
        self.test_preds_sigmoid_list = []
        self.test_labels_list = []
        self.test_raw_preds_list = []
        
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
        # self._check_info(batch=batch)
        
        label = batch['label'].detach() # b, class_num
        label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        video = batch['video'].detach()  # b, t, c, h, w
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        # print(f"Shape of 'video' tensor received by trainer: {video.shape}")
        
        current_batch_size = video.shape[0]
        
        # classification task
        preds = self.model(video)
        
        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)
        
        train_loss = F.binary_cross_entropy_with_logits(preds, label.float())
        # soft_margin_loss = F.soft_margin_loss(y_hat_sigmoid, label.float())

        accuracy = self._accuracy(preds_sigmoid, label)
        precision = self._precision(preds_sigmoid, label)
        val_f1 = self._f1_score(preds_sigmoid, label)

        # self.log_dict({'train_loss': train_loss, 'train_acc': accuracy, 'train_precision': precision}, on_step=True, on_epoch=True)
        # self.log_dict({'train_loss': train_loss, 'train_acc': accuracy, 'train_precision': precision}, on_epoch=True, sync_dist=True, batch_size=current_batch_size)
        # self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        # self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        # self.log("train_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        self.log_dict({'train/f1:': val_f1, 'train/loss': train_loss, 'train/acc': accuracy, 'train/precision': precision}, on_step=True, on_epoch=True, batch_size=label.size(0))
        # self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)


        return train_loss

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
        # self._check_info(batch=batch)
        label = batch['label']
        label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        # input and label
        video = batch['video']
        video = video.permute(0, 2, 1, 3, 4).contiguous()# b, t, c, h, w
        
        current_batch_size = video.shape[0]
        # print(f"Shape of 'video' tensor received by trainer: {video.shape}")
        # pred the video frames
        
        with torch.no_grad():
            preds = self.model(video)

        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)
        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        val_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        val_f1 = self._f1_score(preds_sigmoid, label)
        # print(f"[DEBUG] Epoch: {self.current_epoch}, Val Loss: {val_loss.item():.4f}, Val Acc: {accuracy.item():.4f}")
        # log the val loss and val acc, in step and in epoch.
        # self.log_dict({'val_f1:': val_f1, 'val_loss': val_loss, 'val_acc': accuracy, 'val_precision': precision}, on_step=True, on_epoch=True, batch_size=current_batch_size)
        # self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        # self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size= current_batch_size)
        # self.log("val_f1", val_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size= current_batch_size)
        # self.log("val_precission", precision, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size= current_batch_size)
        self.log_dict({'val/f1:': val_f1, 'val/loss': val_loss, 'val/acc': accuracy, 'val/precision': precision}, on_step=True, on_epoch=True, batch_size=label.size(0))

        
        # return val_loss, accuracy

    def on_validation_epoch_end(self):
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
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        label = batch['label'].detach()
        # when batch > 1, for multi label, to repeat label in (bxt)
        label = label.repeat_interleave(self.uniform_temporal_subsample_num).squeeze()

        self.model.eval()

        # pred the video frames
        with torch.no_grad():
            preds = self.model(video)
        
        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        test_loss = F.binary_cross_entropy_with_logits(preds, label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'test_loss': test_loss, 'test_acc': accuracy, 'test_precision': precision}, on_step=False, on_epoch=True)

        return {
            'pred': preds_sigmoid.tolist(),
            'label': label.tolist()
        }
        
    def on_test_epoch_end(self):
        '''
        テストエポックの最後に呼び出されるフック
        test_stepで保存した全ての予測とラベルを集計し、最終的なメトリックを計算してログに記録
        '''
        # 保存された全ての予測とラベルを結合
        all_preds_sigmoid = torch.cat(self.test_preds_sigmoid_list)
        all_labels = torch.cat(self.test_labels_list)
        all_raw_preds = torch.cat(self.test_raw_preds_list) # 保存した生の予測値を使用

        # 最終損失の計算 (生の予測値を使用するのが数値的に安定)
        final_test_loss = F.binary_cross_entropy_with_logits(all_raw_preds, all_labels.float())

        # 最終的なメトリックの計算 (全データセットに対して一度に)
        final_accuracy = self._accuracy(all_preds_sigmoid, all_labels)
        final_precision = self._precision(all_preds_sigmoid, all_labels)
        final_f1 = self._f1_score(all_preds_sigmoid, all_labels)
        final_confusion_matrix_tensor = self._confusion_matrix(all_preds_sigmoid, all_labels)

        # 最終的なメトリックをログに記録
        self.log_dict({
            'final_test/loss': final_test_loss,
            'final_test/acc': final_accuracy,
            'final_test/precision': final_precision,
            'final_test/f1': final_f1,
            # 混同行列はTensorBoardなどに直接プロットする場合が多い
        }, on_step=False, on_epoch=True)

        # [追加]: 混同行列の描画と保存ロジックをここに移動
        # torchmetricsのConfusionMatrixは直接プロット機能を持っている場合が多い
        # あるいはnumpyに変換してmatplotlib/seabornで描画
        cm_numpy = final_confusion_matrix_tensor.cpu().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_numpy, annot=True, fmt="d", cmap="Blues") # fmt="d"で整数表示
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('test_confusion_matrix.png')
        plt.close() # メモリ解放のためにプロットを閉じる

        # 次のテスト実行のためにリストをクリア
        self.test_preds_sigmoid_list.clear()
        self.test_labels_list.clear()
        self.test_raw_preds_list.clear() # 生の予測値リストもクリア
    
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
        #video_info_a = batch['ap']
        #video_info_b = batch['lat']
        
        # print('vedeoname_a', video_info_a["video_name"])
        # print('vedeoname_b', video_info_b["video_name"])
        
        # * check tensor shape 
        #video_a = video_info_a["video"]
        # video_b = video_info_b["video"]
        
        # assert video_a.shape == video_b.shape

        # * check label 
        
        #label_a = video_info_a["label"]
        # label_b = video_info_b["label"]
        
        # print('label_a:', label_a)
        # print('label_b:', label_b)

        # assert label_a.shape ==  label_b.shape 
        
        # for i in range(len(label_a)):
        #     assert label_a[i] == label_b[i]
        
        # * check video name 
        # video_name_a = video_info_a["video_name"]
        # video_name_b = video_info_b["video_name"]

        # assert len(video_name_a) == len(video_name_b)
        
        # for i in range(len(video_name_a)):
        #     assert video_name_a[i] == video_name_b[i] 
        