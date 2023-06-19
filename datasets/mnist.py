import os
from typing import Literal

import albumentations as A

import idx2numpy

import numpy as np

import pytorch_lightning as pl

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
import cv2


class MNISTDataset(Dataset):
    def __init__(
        self,
        data_dir="data",
        transform=None,
        stage: Literal["fit", "test"] = "fit",
    ) -> None:
        super(MNISTDataset, self).__init__()

        prefix = "train" if stage == "fit" else "t10k"

        samples_zip: str = os.path.join(
            data_dir, "MNIST", "raw", f"{prefix}-images-idx3-ubyte"
        )

        labels_zip: str = os.path.join(
            data_dir, "MNIST", "raw", f"{prefix}-labels-idx1-ubyte"
        )

        self.samples = idx2numpy.convert_from_file(samples_zip)
        self.labels = idx2numpy.convert_from_file(labels_zip)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = self.samples[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = image.astype(np.float32).reshape(1, 32, 32) / 255.0
        image = torch.from_numpy(image)

        return image, label


class MNISTDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
    ) -> None:
        super(MNISTDatamodule, self).__init__()

        self.data_dir: str = data_dir
        self.batch_size: int = batch_size

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            mnist: MNISTDataset = MNISTDataset(
                transform=A.Compose(
                    [
                        A.Resize(32, 32),
                        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT),
                    ]
                )
            )

            self.training_dataset, self.validation_dataset = random_split(
                mnist, [55000, 5000]
            )
        else:
            self.test_dataset: MNISTDataset = MNISTDataset(self.data_dir, stage="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_dataset, batch_size=1, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)
