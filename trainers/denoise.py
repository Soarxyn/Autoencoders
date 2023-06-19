import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchmetrics as tm

from .trainer import BaseModule


class DenoiseLoss(nn.Module):
    def __init__(self) -> None:
        super(DenoiseLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, pred, y):
        return self.mse(pred, y)


class DenoiseAutoencoder(BaseModule):
    def __init__(self, autoencoder: nn.Module, learning_rate: float) -> None:
        metrics = tm.MetricCollection(
            {"Accuracy": tm.Accuracy("binary"), "Peak SNR": tm.PeakSignalNoiseRatio()}
        )

        super(DenoiseAutoencoder, self).__init__(autoencoder, metrics)
        self.learning_rate = learning_rate

    def criterion(self) -> nn.Module:
        return DenoiseLoss()

    def preprocess_batch(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn(x.size(), device="cuda")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=self.learning_rate)
        scheduler = sched.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
