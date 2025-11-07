"""PyTorch Lightning datamodule for the CIFAR10 dataset."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from datasets import DatasetDict, load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class CIFAR10DataModule(pl.LightningDataModule):
    """Lightning datamodule for the Hugging Face `cifar10` dataset."""

    dataset_name: str = "cifar10"

    def __init__(
        self,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = False,
        val_split: float = 0.02,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.seed = seed

        self._datasets: Dict[str, Any] = {}
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

        val_key = self._choose_val_split(dataset_dict)

        if val_key:
            train_dataset = dataset_dict["train"]
            val_dataset = dataset_dict[val_key]
        else:
            split = dataset_dict["train"].train_test_split(
                test_size=self.val_split,
                seed=self.seed,
            )
            train_dataset = split["train"]
            val_dataset = split["test"]

        train_dataset.set_transform(self._transform_batch)
        val_dataset.set_transform(self._transform_batch)

        self._datasets["train"] = train_dataset
        self._datasets["val"] = val_dataset

    def train_dataloader(self) -> DataLoader:
        dataset = self._datasets.get("train")
        if dataset is None:
            raise RuntimeError("Call `setup()` before requesting the train dataloader.")

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._datasets.get("val")
        if dataset is None:
            raise RuntimeError("Call `setup()` before requesting the val dataloader.")

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _choose_val_split(self, dataset_dict: DatasetDict) -> Optional[str]:
        """Pick an existing validation-like split if available."""

        for candidate in ("validation", "val", "test"):
            if candidate in dataset_dict:
                return candidate
        return None

    def _transform_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        images = [self.image_transform(image) for image in examples["img"]]
        examples["pixel_values"] = images
        return {"pixel_values": examples["pixel_values"]}

    def _collate_fn(self, batch: Any) -> torch.Tensor:
        if isinstance(batch[0], dict) and "pixel_values" in batch[0]:
            return torch.stack([item["pixel_values"] for item in batch], dim=0)
        if torch.is_tensor(batch[0]):
            return torch.stack(batch, dim=0)
        raise TypeError("Unsupported batch format for CIFAR10DataModule")


if __name__ == "__main__":
    datamodule = CIFAR10DataModule()
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule.train_dataloader())
    print(datamodule.val_dataloader())


