"""PyTorch Lightning datamodule for CIFAR10 image classification."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from datasets import DatasetDict, load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class CIFAR10ClassificationDataModule(pl.LightningDataModule):
    """Lightning datamodule for CIFAR10 classification with data augmentation."""

    dataset_name: str = "cifar10"

    def __init__(
        self,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        use_augmentation: bool = True,
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
        
        # Training transforms with data augmentation
        if use_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ])
        
        # Validation/test transforms (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

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

        # CIFAR-10 has train and test splits
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]

        # Set transforms
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
        images = [self.train_transform(image) for image in examples["img"]]
        labels = examples["label"]
        return {"images": images, "labels": labels}

    def _transform_val_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a batch of validation/test examples."""
        images = [self.val_transform(image) for image in examples["img"]]
        labels = examples["label"]
        return {"images": images, "labels": labels}

    def _collate_fn(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of examples into images and labels tensors."""
        if isinstance(batch[0], dict):
            images = torch.stack([item["images"] for item in batch], dim=0)
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
            return images, labels
        raise TypeError("Unsupported batch format for CIFAR10ClassificationDataModule")


if __name__ == "__main__":
    # Test the datamodule
    datamodule = CIFAR10ClassificationDataModule(
        train_batch_size=32,
        val_batch_size=32,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    print("Train dataloader:")
    train_loader = datamodule.train_dataloader()
    print(f"  Number of batches: {len(train_loader)}")
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: images={images.shape}, labels={labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\nVal dataloader:")
    val_loader = datamodule.val_dataloader()
    print(f"  Number of batches: {len(val_loader)}")
    images, labels = next(iter(val_loader))
    print(f"  Batch shape: images={images.shape}, labels={labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")


