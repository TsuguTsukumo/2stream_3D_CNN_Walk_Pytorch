from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision

try:
    from project.utils.helper import save_CM, save_inference, save_metrics
except ModuleNotFoundError:
    from utils.helper import save_CM, save_inference, save_metrics


class PairedVideoTrainer(LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()

        self.model_type = hparams.model
        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr
        self.num_classes = hparams.model.model_class_num
        self.uniform_temporal_subsample_num = hparams.data.uniform_temporal_subsample_num
        self.log_path = hparams.train.log_path

        self.model = self.build_model(hparams)
        self.save_hyperparameters()

        self._accuracy = BinaryAccuracy()
        self._precision = BinaryPrecision()
        self._f1_score = BinaryF1Score()

        self.test_pred_list: list[float] = []
        self.test_label_list: list[int] = []

    @abstractmethod
    def build_model(self, hparams):
        raise NotImplementedError

    def forward(self, video_ap: torch.Tensor, video_lat: torch.Tensor) -> torch.Tensor:
        return self.model(video_ap, video_lat)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val", use_no_grad=True)

    def on_test_start(self) -> None:
        self.test_pred_list.clear()
        self.test_label_list.clear()
        logging.info("test start")

    def on_test_end(self) -> None:
        logging.info("test end")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        results = self._shared_step(
            batch,
            stage="test",
            use_no_grad=True,
            track_predictions=True,
        )
        return results["preds_sigmoid"]

    def on_test_epoch_end(self) -> None:
        if not self.test_pred_list:
            logging.warning("Skip saving test artifacts because no predictions were collected.")
            return

        save_inference(
            self.test_pred_list,
            self.test_label_list,
            fold=self.logger.name,
            save_path=self.log_path,
        )
        save_metrics(
            self.test_pred_list,
            self.test_label_list,
            fold=self.logger.name,
            save_path=self.log_path,
            num_class=self.num_classes,
        )
        save_CM(
            self.test_pred_list,
            self.test_label_list,
            save_path=self.log_path,
            num_class=self.num_classes,
            fold=self.logger.name,
        )

        logging.info("test epoch end")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val/loss",
            },
        }

    def _shared_step(
        self,
        batch: dict,
        stage: str,
        use_no_grad: bool = False,
        track_predictions: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._check_info(batch)

        label = batch["ap"]["label"]
        video_ap = batch["ap"]["video"]
        video_lat = batch["lat"]["video"]

        if use_no_grad:
            with torch.no_grad():
                preds = self(video_ap, video_lat)
        else:
            preds = self(video_ap, video_lat)

        preds = preds.squeeze(dim=-1)
        preds_sigmoid = torch.sigmoid(preds)
        loss = F.binary_cross_entropy_with_logits(preds, label.float())

        accuracy = self._accuracy(preds_sigmoid, label)
        precision = self._precision(preds_sigmoid, label)
        f1_score = self._f1_score(preds_sigmoid, label)

        self.log_dict(
            {
                f"{stage}/f1": f1_score,
                f"{stage}/loss": loss,
                f"{stage}/acc": accuracy,
                f"{stage}/precision": precision,
            },
            on_step=True,
            on_epoch=True,
            batch_size=label.size(0),
        )

        if track_predictions:
            self.test_pred_list.extend(preds_sigmoid.detach().cpu().tolist())
            self.test_label_list.extend(label.detach().cpu().tolist())

        return {
            "loss": loss,
            "preds_sigmoid": preds_sigmoid,
        }

    def _check_info(self, batch: dict) -> None:
        video_info_ap = batch["ap"]
        video_info_lat = batch["lat"]

        video_ap = video_info_ap["video"]
        video_lat = video_info_lat["video"]
        assert video_ap.shape == video_lat.shape

        label_ap = video_info_ap["label"]
        label_lat = video_info_lat["label"]
        assert label_ap.shape == label_lat.shape

        for index in range(len(label_ap)):
            assert label_ap[index] == label_lat[index]

        video_name_ap = video_info_ap["video_name"]
        video_name_lat = video_info_lat["video_name"]
        assert len(video_name_ap) == len(video_name_lat)

        for index in range(len(video_name_ap)):
            assert video_name_ap[index] == video_name_lat[index]
