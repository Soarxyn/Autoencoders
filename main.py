from typing import List

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import pytorch_lightning.loggers as loggers
import torch
from core.resunet import ResUNet
from datasets.mnist import MNISTDatamodule
from trainers.inpainting import InpaintingAutoencoder
from trainers.trainer import SampleLogger
from trainers.denoise import DenoiseAutoencoder


def init_trainer(
    callbacks: List[pl.Callback],
    logger: loggers.Logger,
    max_epochs: int = 100,
    acc_grad_batches: int = 16,
) -> pl.Trainer:
    return pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=logger,
        precision="16-mixed",
        callbacks=callbacks,
        max_epochs=max_epochs,
        accumulate_grad_batches=acc_grad_batches,
    )


def main() -> None:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    tensorboard = loggers.TensorBoardLogger(save_dir="", default_hp_metric=False)
    trainer = init_trainer(
        callbacks=[
            SampleLogger(freq=1),
            cb.LearningRateMonitor(logging_interval="epoch"),
            cb.ModelCheckpoint(
                monitor="val_loss",
                dirpath="checkpoints/",
                filename="resunet",
                mode="min",
                verbose=True,
            ),
            cb.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode="min",
            ),
            cb.StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        logger=tensorboard,
    )

    mnist = MNISTDatamodule()
    network = ResUNet(input_dim=1, output_dim=1, filters=[32, 64, 128, 10], logits=True)
    autoencoder = DenoiseAutoencoder(autoencoder=network, learning_rate=1e-3)

    trainer.fit(autoencoder, datamodule=mnist)


if __name__ == "__main__":
    main()
