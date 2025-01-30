'''
a pytorch lightning data module based dataloader, for train/val/test dataset prepare.

'''

# %%
import matplotlib.pylab as plt

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    Div255,
    create_video_transform,
)

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule
import os

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data import make_clip_sampler

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset, labeled_video_dataset
from typing import Tuple

# %%

def WalkDataset(
    data_path_1: str,
    data_path_2: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
) -> Tuple[LabeledVideoDataset, LabeledVideoDataset]: 
    '''
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
    '''
    dataset_1 = labeled_video_dataset(
        data_path_1,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder
    )
    
    dataset_2 = labeled_video_dataset(
        data_path_2,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder
    )
    
    return dataset_1, dataset_2


# %%

class WalkDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        # use this for dataloader
        self._TRAIN_PATH_1 = opt.train_path_1
        self._TRAIN_PATH_2 = opt.train_path_2

        self._PRE_PROCESS_FLAG = opt.pre_process_flag

        self._BATCH_SIZE = opt.batch_size
        self._NUM_WORKERS = opt.num_workers
        self._IMG_SIZE = opt.img_size

        # frame rate
        self._CLIP_DURATION = opt.clip_duration
        self.uniform_temporal_subsample_num = opt.uniform_temporal_subsample_num

        self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # uniform clip T frames from the given n sec video.
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                            
                            # dived the pixel from [0, 255] tp [0, 1], to save computing resources.
                            Div255(),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),

                            # RandomShortSideScale(min_size=256, max_size=320),
                            # RandomCrop(self._IMG_SIZE),

                            # ShortSideScale(self._IMG_SIZE),

                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

        self.raw_train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # uniform clip T frames from the given n sec video.
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                            
                            # dived the pixel from [0, 255] to [0, 1], to save computing resources.
                            Div255(),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),

                            RandomShortSideScale(min_size=256, max_size=320),

                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    )
                )
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        '''
        transform = self.train_transform if self._PRE_PROCESS_FLAG else self.raw_train_transform

        # if stage == "f it" or stage == None:
        if stage in ("fit", None):
            self.train_dataset_1, self.train_dataset_2 = WalkDataset(
                data_path=os.path.join(self._TRAIN_PATH_1, "train"),
                second_data_path=os.path.join(self._TRAIN_PATH_2, "train"),
                clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
                transform=transform,
            )

        if stage in ("fit", "validate", None):
            self.val_dataset_1, self.val_dataset_2 = WalkDataset(
                data_path=os.path.join(self._TRAIN_PATH_1, "val"),
                second_data_path=os.path.join(self._TRAIN_PATH_2, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=transform,
            )

        if stage in ("predict", "test", None):
            self.test_dataset_1, self.test_dataset_2 = WalkDataset(
                data_path=os.path.join(self._TRAIN_PATH_1, "val"),
                second_data_path=os.path.join(self._TRAIN_PATH_2, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=transform
            )

    def train_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.train_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            DataLoader(
                self.train_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            )
        ]

    def val_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.val_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            DataLoader(
                self.val_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            )
        ]

    def test_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.test_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            DataLoader(
                self.test_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            )
        ]

    def predict_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.test_dataset_1,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            ),
            DataLoader(
                self.test_dataset_2,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
            )
        ]