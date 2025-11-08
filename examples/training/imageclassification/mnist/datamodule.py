"""PyTorch Lightning datamodule for MNIST image classification."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from datasets import DatasetDict, load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class MNISTClassificationDataModule(pl.LightningDataModule):
    """Lightning datamodule for MNIST classification."""

    dataset_name: str = "mnist"

    def __init__(
        self,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_augmentation: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_augmentation = use_augmentation
        self.seed = seed

        self._datasets: Dict[str, Any] = {}

        # Training transforms with optional augmentation
        augmentations = []
        if use_augmentation:
            augmentations.extend(
                [
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(0, translate=(0.1, 0.1)),
                ]
            )
        augmentations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_transform = transforms.Compose(augmentations)

        # Validation/test transforms (no augmentation)
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """Download the dataset locally if it is not already present."""

        load_dataset(self.dataset_name)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        """Load and transform the dataset splits for the requested stage."""

        if self._datasets:
            return

        dataset_dict: DatasetDict = load_dataset(self.dataset_name)

        # MNIST has train and test splits
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]

        train_dataset.set_transform(self._transform_train_batch)
        test_dataset.set_transform(self._transform_val_batch)

        self._datasets["train"] = train_dataset
        self._datasets["val"] = test_dataset
        self._datasets["test"] = test_dataset

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""

        dataset = self._datasets.get("train")
        if dataset is None:
            raise RuntimeError("Call `setup()` before requesting the train dataloader.")

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""

        dataset = self._datasets.get("val")
        if dataset is None:
            raise RuntimeError("Call `setup()` before requesting the val dataloader.")

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""

        dataset = self._datasets.get("test")
        if dataset is None:
            raise RuntimeError("Call `setup()` before requesting the test dataloader.")

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _transform_train_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a batch of training examples."""

        images = [self.train_transform(image.convert("L")) for image in examples["image"]]
        labels = examples["label"]
        return {"images": images, "labels": labels}

    def _transform_val_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a batch of validation/test examples."""

        images = [self.val_transform(image.convert("L")) for image in examples["image"]]
        labels = examples["label"]
        return {"images": images, "labels": labels}

    def _collate_fn(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of examples into images and labels tensors."""

        if isinstance(batch[0], dict):
            images = torch.stack([item["images"] for item in batch], dim=0)
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
            return images, labels
        raise TypeError("Unsupported batch format for MNISTClassificationDataModule")


if __name__ == "__main__":
    # Simple sanity check
    datamodule = MNISTClassificationDataModule(
        train_batch_size=32,
        val_batch_size=32,
        num_workers=0,
        use_augmentation=True,
    )
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    print("Train batch shape:", images.shape, labels.shape)

    batch = next(iter(val_loader))
    images, labels = batch
    print("Val batch shape:", images.shape, labels.shape)

