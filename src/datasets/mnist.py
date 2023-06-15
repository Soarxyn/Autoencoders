import gzip
import logging
import os
import struct
from typing import Dict, Literal, Tuple

import albumentations as A

import numpy as np

import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        transform: A.Compose | None = None,
        stage: Literal["fit", "test"] = "fit",
    ) -> None:
        super(MNISTDataset, self).__init__()

        prefix = "train" if stage == "fit" else "t10k"

        samples_zip: str = os.path.join(
            data_dir, "MNIST", "raw", f"{prefix}-images-idx3-ubyte.gz"
        )

        labels_zip: str = os.path.join(
            data_dir, "MNIST", "raw", f"{prefix}-labels-idx1-ubyte.gz"
        )

        with gzip.open(samples_zip, "r") as samples_file:
            _, image_count, height, width = struct.unpack(
                ">IIII", samples_file.read(16)
            )
            image_data: bytes = samples_file.read()

            self.samples: NDArray[np.uint8] = np.frombuffer(
                image_data, dtype=np.uint8
            ).reshape((image_count, 1, height, width))

            self.samples = np.where(self.samples > 127, 1, 0)
        with gzip.open(labels_zip, "r") as labels_file:
            _, label_count = struct.unpack(">II", labels_file.read(8))
            label_data: bytes = labels_file.read()

            self.labels: NDArray[np.uint8] = np.frombuffer(
                label_data, dtype=np.uint8
            ).reshape((image_count,))
        if transform == None:
            self.transform = A.Compose([ToTensorV2()])
        else:
            self.transform = transform
        logging.info(f"MNIST - {len(self.samples)} samples in {stage} stage.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[NDArray[np.uint8], np.uint8]:
        image: NDArray[np.uint8] = self.samples[index]
        label: np.uint8 = self.labels[index]

        transformed: Dict[str, NDArray[np.uint8]] = self.transform(image=image)

        return transformed["image"], label


class MNISTDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        transforms: A.Compose | None = None,
    ) -> None:
        super(MNISTDatamodule, self).__init__()

        self.data_dir: str = data_dir
        self.batch_size: int = batch_size

        self.transforms: A.Compose | None = transforms

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            mnist: MNISTDataset = MNISTDataset(self.data_dir, self.transforms, "fit")

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
