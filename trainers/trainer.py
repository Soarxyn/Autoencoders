import random

from abc import ABC, abstractmethod
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid


class BaseModule(pl.LightningModule, ABC):
    def __init__(self, autoencoder: nn.Module, metrics: tm.MetricCollection) -> None:
        super(BaseModule, self).__init__()

        self.autoencoder = autoencoder
        self.m = nn.Sigmoid()

        self.training_metrics = metrics.clone()
        self.validation_metrics = metrics.clone()

        self.training_loss = tm.MeanMetric()
        self.validation_loss = tm.MeanMetric()

        self.validation_samples: List[torch.Tensor] = []
        self.validation_inputs: List[torch.Tensor] = []
        self.validation_preds: List[torch.Tensor] = []

    @property
    @abstractmethod
    def criterion(self) -> nn.Module:
        pass

    @abstractmethod
    def preprocess_batch(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x):
        return self.autoencoder(x)

    def on_train_start(self) -> None:
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                self.tb_experiment = logger.experiment
                break

    def training_step(self, batch, batch_idx):
        x, _ = batch

        x_in = self.preprocess_batch(x)
        y = x.clone().int()

        prediction = self.forward(x_in)
        prediction_p = self.m(prediction)

        loss = self.criterion().forward(prediction, prediction_p, x)

        self.training_metrics(prediction, y)
        self.training_loss(loss.item())

        self.tb_experiment.add_scalars(
            "Loss/Step", {"Train": loss.item()}, self.global_step
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        x_in = self.preprocess_batch(x)
        y = x.clone().int()

        prediction = self.forward(x_in)
        prediction_p = self.m(prediction)

        loss = self.criterion().forward(prediction, prediction_p, x)

        self.validation_metrics(prediction, y)
        self.validation_loss(loss.item())

        self.validation_samples.append(x[0].detach().cpu())
        self.validation_inputs.append(x_in[0].detach().cpu())
        self.validation_preds.append(prediction[0].detach().cpu())

        return loss

    def on_train_epoch_end(self) -> None:
        for name, metric in self.training_metrics.items():
            if isinstance(metric, tm.Metric):
                self.tb_experiment.add_scalars(
                    name, {"Train": metric.compute()}, self.current_epoch
                )
        self.tb_experiment.add_scalars(
            "Loss/Epoch", {"Train": self.training_loss.compute()}, self.current_epoch
        )

        self.training_metrics.reset()
        self.training_loss.reset()

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            for name, metric in self.validation_metrics.items():
                if isinstance(metric, tm.Metric):
                    self.tb_experiment.add_scalars(
                        name, {"Val": metric.compute()}, self.current_epoch
                    )
        val_loss = self.validation_loss.compute()

        if not self.trainer.sanity_checking:
            self.tb_experiment.add_scalars(
                "Loss/Epoch", {"Val": val_loss}, self.current_epoch
            )
        self.log("val_loss", val_loss)

        self.validation_metrics.reset()
        self.validation_loss.reset()


class SampleLogger(pl.Callback):
    def __init__(self, n_samples: int = 16, n_rows: int = 8, freq: int = 5) -> None:
        super(SampleLogger, self).__init__()

        self.n_samples: int = n_samples
        self.n_rows: int = n_rows
        self.freq: int = freq

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: BaseModule
    ) -> None:
        if trainer.current_epoch % self.freq == 0 and not trainer.sanity_checking:
            idxs = random.sample(
                range(len(pl_module.validation_samples)), k=self.n_samples
            )

            samples = [pl_module.validation_samples[idx] for idx in idxs]
            inputs = [pl_module.validation_inputs[idx] for idx in idxs]
            preds = [pl_module.validation_preds[idx] for idx in idxs]

            log_image = make_grid(samples + inputs + preds, nrow=self.n_rows)

            if isinstance(trainer.logger, TensorBoardLogger):
                trainer.logger.experiment.add_image(
                    "Validation/Visualization",
                    log_image,
                    trainer.current_epoch,
                )
