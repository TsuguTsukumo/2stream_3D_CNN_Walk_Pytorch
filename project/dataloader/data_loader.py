"""
a pytorch lightning data module based dataloader, for train/val/test dataset prepare.

"""

from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    Lambda
)

# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     Normalize,
#     RandomShortSideScale,
#     # UniformTemporalSubsample,
#     # Div255,
# )

from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2 import UniformTemporalSubsample

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule
import os

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data import make_clip_sampler
from pytorch_lightning.trainer.supporters import CombinedLoader

from pytorchvideo.data.labeled_video_dataset import (
    LabeledVideoDataset,
    labeled_video_dataset,
)

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
        inpt = inpt.permute(1,0,2,3)
        return self._call_kernel(F.uniform_temporal_subsample, inpt, self.num_samples)


def WalkDataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
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
        LabeledVideoDataset: _description_
    """
    return labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )

class WalkDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        # use this for dataloader
        self._TRAIN_PATH_A = opt.train_path_a
        self._TRAIN_PATH_B = opt.train_path_b

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
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                            # Lambda(x: x/255),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
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
        
        data_path_a = self._TRAIN_PATH_A
        data_path_b = self._TRAIN_PATH_B
        transform = self.train_transform

        # if stage == "f it" or stage == None:
        if stage in ("fit", None):
            self.train_dataset_a = WalkDataset(
                data_path=os.path.join(data_path_a, "train"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=transform,
                video_sampler=torch.utils.data.SequentialSampler,
            )

            self.train_dataset_b = WalkDataset(
                data_path=os.path.join(data_path_b, "train"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=transform,
                video_sampler=torch.utils.data.SequentialSampler,
            )

        if stage in ("fit", "validate", "predict", "test", None):
            self.val_dataset_a = WalkDataset(
                data_path=os.path.join(data_path_a, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=transform,
                video_sampler=torch.utils.data.SequentialSampler,
            )

            self.val_dataset_b = WalkDataset(
                data_path=os.path.join(data_path_b, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=transform,
                video_sampler=torch.utils.data.SequentialSampler,
            )

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        loader_a = DataLoader(
            self.train_dataset_a,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )

        loader_b = DataLoader(
            self.train_dataset_b,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )
        loaders = {"a": loader_a, "b": loader_b}
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        loader_a = DataLoader(
            self.train_dataset_a,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )

        loader_b = DataLoader(
            self.train_dataset_b,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )
        loaders = {"a": loader_a, "b": loader_b}
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        loader_a = DataLoader(
            self.train_dataset_a,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )

        loader_b = DataLoader(
            self.train_dataset_b,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )
        loaders = {"a": loader_a, "b": loader_b}
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loader

    def predict_dataloader(self) -> DataLoader:
        """
        create the Walk pred partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        loader_a = DataLoader(
            self.train_dataset_a,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )

        loader_b = DataLoader(
            self.train_dataset_b,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )
        loaders = {"a": loader_a, "b": loader_b}
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loader
