import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchmetrics as tm

from .trainer import BaseModule


class InpaintingLoss(nn.Module):
    def __init__(self) -> None:
        super(InpaintingLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, pred, y):
        return self.mse(pred, y) + self.bce(logits, y)


class InpaintingAutoencoder(BaseModule):
    def __init__(self, autoencoder: nn.Module, learning_rate: float) -> None:
        metrics = tm.MetricCollection({"Accuracy": tm.Accuracy("binary")})

        super(InpaintingAutoencoder, self).__init__(autoencoder, metrics)
        self.learning_rate = learning_rate

    def criterion(self) -> nn.Module:
        return InpaintingLoss()

    def preprocess_batch(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()

        hidden = x.clone()
        hidden[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0

        return hidden

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=self.learning_rate)
        scheduler = sched.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
