import random
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchmetrics as tm
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid


class DenoiseAutoencoder(pl.LightningModule):
    def __init__(self, autoencoder: nn.Module, learning_rate: float) -> None:
        super(DenoiseAutoencoder, self).__init__()

        self.autoencoder = autoencoder
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        metrics = tm.MetricCollection(
            {"Accuracy": tm.Accuracy("binary"), "Peak SNR": tm.PeakSignalNoiseRatio()}
        )

        self.training_metrics = metrics.clone()
        self.validation_metrics = metrics.clone()
        # TODO: self.training_metrics = metrics.clone()

        self.training_loss = tm.MeanMetric()
        self.validation_loss = tm.MeanMetric()

        self.validation_samples: List[torch.Tensor] = []
        self.validation_inputs: List[torch.Tensor] = []
        self.validation_preds: List[torch.Tensor] = []

    def forward(self, x):
        return self.autoencoder(x)

    def on_train_start(self) -> None:
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                self.tb_experiment = logger.experiment
                break

    def training_step(self, batch, batch_idx):
        x, _ = batch
        noisy = x + torch.randn(x.size(), device="cuda")
        y = x.clone().int()

        prediction = self.forward(noisy)
        loss = self.criterion(prediction, x)

        self.training_metrics(prediction, y)
        self.training_loss(loss.item())

        self.tb_experiment.add_scalars(
            "Loss/Step", {"Train": loss.item()}, self.global_step
        )

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

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        noisy = x + torch.randn(x.size(), device="cuda")
        y = x.clone().int()

        prediction = self.forward(noisy)
        loss = self.criterion(prediction, x)

        self.validation_metrics(prediction, y)
        self.validation_loss(loss.item())

        self.validation_samples.append(x[0].detach().cpu())
        self.validation_inputs.append(noisy[0].detach().cpu())
        self.validation_preds.append(prediction[0].detach().cpu())

        return loss

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=self.learning_rate)
        scheduler = sched.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class SampleLogger(pl.Callback):
    def __init__(self, n_samples: int = 16, n_rows: int = 8, freq: int = 5) -> None:
        super(SampleLogger, self).__init__()

        self.n_samples: int = n_samples
        self.n_rows: int = n_rows
        self.freq: int = freq

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: DenoiseAutoencoder
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
