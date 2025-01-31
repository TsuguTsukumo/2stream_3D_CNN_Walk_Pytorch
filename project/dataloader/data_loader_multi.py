#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/dataloader/data_loader_multi.py
Project: /workspace/project/project/dataloader
Created Date: Thursday January 30th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 30th 2025 2:09:48 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
)

from torchvision.transforms.v2 import functional as F, Transform
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     Normalize,
#     RandomShortSideScale,
#     ShortSideScale,
#     UniformTemporalSubsample,
#     Div255,
#     create_video_transform,
# )

from typing import Any, Callable, Dict, Optional, Type, Tuple
from pytorch_lightning import LightningDataModule
import os

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.utilities.combined_loader import CombinedLoader

from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data import make_clip_sampler

from pytorchvideo.data.labeled_video_dataset import (
    LabeledVideoDataset,
    labeled_video_dataset,
)

class UniformTemporalSubsample(Transform):
    """Uniformly subsample ``num_samples`` indices from the temporal dimension of the video.

    Videos are expected to be of shape ``[..., T, C, H, W]`` where ``T`` denotes the temporal dimension.

    When ``num_samples`` is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        num_samples (int): The number of equispaced samples to be selected
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = inpt.permute(1, 0, 2, 3) # [C, T, H, W] -> [T, C, H, W]
        return self._call_kernel(F.uniform_temporal_subsample, inpt, self.num_samples)


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return x / 255.0


def WalkDataset(
    data_path_ap: str,
    data_path_lat: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
) -> Tuple[LabeledVideoDataset, LabeledVideoDataset]:
    """
    A helper function to create "LabeledVideoDataset" object for the Walk dataset.

    Args:
        data_path (str): Path to the data. The path defines how the data should be read. For a directory, the directory structure defines the classes (i.e. each subdirectory is class).
        clip_sampler (ClipSampler): Defines how clips should be sampled from each video. See the clip sampling documentation for more information.
        video_sampler (Type[torch.utils.data.Sampler], optional): Sampler for the internal video container. Defaults to torch.utils.data.RandomSampler.
        transform (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]], optional): This callable is evaluated on the clip output before the clip is returned. Defaults to None.
        video_path_prefix (str, optional): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. Defaults to "".
        decode_audio (bool, optional): If True, also decode audio from video. Defaults to False. Notice that, if Ture will trigger the stack error.
        decoder (str, optional): Defines what type of decoder used to decode a video. Defaults to "pyav".

    Returns:
        Tuple[LabeledVideoDataset, LabeledVideoDataset]: Two dataset objects for the two video paths
    """
    dataset_1 = labeled_video_dataset(
        data_path_ap,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )

    dataset_2 = labeled_video_dataset(
        data_path_lat,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )

    return dataset_1, dataset_2


class MultiData(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        # use this for dataloader
        self._TRAIN_PATH_1 = opt.data.ap_data_path
        self._TRAIN_PATH_2 = opt.data.lat_data_path

        self._BATCH_SIZE = opt.data.batch_size
        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # frame rate
        self._CLIP_DURATION = opt.data.clip_duration
        self.uniform_temporal_subsample_num = opt.data.uniform_temporal_subsample_num

        self.current_fold = opt.train.current_fold

        self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # uniform clip T frames from the given n sec video.
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                            # dived the pixel from [0, 255] tp [0, 1], to save computing resources.
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            # Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            # RandomShortSideScale(min_size=256, max_size=320),
                            # RandomCrop(self._IMG_SIZE),
                            # ShortSideScale(self._IMG_SIZE),
                            # RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # if stage == "fit" or stage == None:
        if stage in ("fit", None):
            self.train_dataset_1, self.train_dataset_2 = WalkDataset(
                data_path_ap=os.path.join(self._TRAIN_PATH_1, self.current_fold, "train"),
                data_path_lat=os.path.join(self._TRAIN_PATH_2, self.current_fold, "train"),
                clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
                video_sampler=torch.utils.data.SequentialSampler,
                transform=self.train_transform,
            )

        if stage in ("fit", "validate", None):
            self.val_dataset_1, self.val_dataset_2 = WalkDataset(
                data_path_ap=os.path.join(self._TRAIN_PATH_1, self.current_fold, "val"),
                data_path_lat=os.path.join(self._TRAIN_PATH_2, self.current_fold, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                video_sampler=torch.utils.data.SequentialSampler,
                transform=self.train_transform,
            )

        if stage in ("predict", "test", None):
            self.test_dataset_1, self.test_dataset_2 = WalkDataset(
                data_path_ap=os.path.join(self._TRAIN_PATH_1, self.current_fold, "val"),
                data_path_lat=os.path.join(self._TRAIN_PATH_2, self.current_fold, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                video_sampler=torch.utils.data.SequentialSampler,
                transform=self.train_transform,
            )

    def train_dataloader(self) -> DataLoader:
        combined_loader = {
            "ap": DataLoader(
                self.train_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            "lat": DataLoader(
                self.train_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),

        }
        return CombinedLoader(
            combined_loader,
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> DataLoader:

        combined_loader = {
            "ap": DataLoader(
                self.val_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            "lat": DataLoader(
                self.val_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
        }
        return CombinedLoader(
            combined_loader,
            mode="max_size_cycle",
        )
    
    def test_dataloader(self) -> DataLoader:
        
        combined_loader = {
            "ap": DataLoader(
                self.test_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            "lat": DataLoader(
                self.test_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
        }
        return CombinedLoader(
            combined_loader,
            mode="max_size_cycle",
        )