#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/models/make_model.py
Project: /workspace/project/project/models
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
'''

import torch
import torch.nn as nn

class single(nn.Module):

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num

        self.transfor_learning = hparams.transfor_learning

        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfor_learning)

        self.resnet_model.fc = torch.nn.Linear(2048, self.model_class_num, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        b, c, t, h, w = img.size()
        # make frame to batch image, wich (b*t, c, h, w)
        batch_imgs = img.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

        output = self.resnet_model(batch_imgs)

        return output

class early_fusion(nn.Module):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model_class_num

        self.transfor_learning = hparams.transfor_learning
        self.uniform_temporal_subsample_num = hparams.uniform_temporal_subsample_num

        # change the resnet work structure
        self.resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=self.transfor_learning)

        self.resnet_model.conv1 = torch.nn.Conv2d(3 * self.uniform_temporal_subsample_num, out_channels=64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_model.fc = torch.nn.Linear(2048, self.model_class_num, bias=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        b, c, t, h, w = img.size()

        batch_imgs = img.permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)

        output = self.resnet_model(batch_imgs)

        return output

class late_fusion(nn.Module):

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model.model_class_num

        # self.transfor_learning = hparams.transfor_learning

        model_ap = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        model_lat = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        # late fusion model
        self.first_model = model_ap.blocks[:-1]
        self.second_model = model_lat.blocks[:-1]

        self.head = model_ap.blocks[-1]
        self.head.proj = nn.Linear(2048 * 2, self.model_class_num)

    def forward(self, video_ap: torch.Tensor, video_lat: torch.Tensor) -> torch.Tensor:

        # transporse the video tensor
        video_ap = video_ap.permute(0, 2, 1, 3, 4) # b, t, c, h, w > b, c, t, h, w
        video_lat = video_lat.permute(0, 2, 1, 3, 4) # b, t, c, h, w > b, c, t, h, w

        for layer in self.first_model:
            video_ap = layer(video_ap)

        for layer in self.second_model:
            video_lat = layer(video_lat)    

        cat_feat = torch.cat((video_ap, video_lat), dim = 1) # b, c

        # output = self.head(cat_feat)

        # head 
        cat_feat = self.head(cat_feat)

        return cat_feat

class slow_fast(nn.Module):

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class

        self.model_ap = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model_lat = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        self.model_ap.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
        self.model_lat.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
    
    def forward(self, video_ap: torch.Tensor, video_lat: torch.Tensor) -> torch.Tensor:
        
        # transporse the
        video_ap = video_ap.permute(0, 2, 1, 3, 4)
        video_lat = video_lat.permute(0, 2, 1, 3, 4)

        output_ap = self.model_ap(video_ap)
        output_lat = self.model_lat(video_lat)

        return output_ap, output_lat