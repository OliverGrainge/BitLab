import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from src.utils import get_data_dir


# ============================================================================
# Base Classes
# ============================================================================

class BaseDatasetSFT(Dataset, ABC):
    """
    Abstract base class for supervised fine-tuning datasets.
    
    Subclasses must implement methods to provide prompt-response pairs
    for training language models in chat format.
    """
    
    @abstractmethod
    def __getitem__(self, index: int) -> List[Dict[str, str]]:
        """
        Get a single example from the dataset.
        
        Args:
            index: Index of the example to retrieve.
            
        Returns:
            A list of message dictionaries, each with 'role' and 'content' keys.
            Format: [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            The size of the dataset.
        """
        pass


class BaseDatasetPT(Dataset, ABC):
    """
    Abstract base class for pretraining datasets.
    
    Subclasses must implement methods to provide raw text for
    causal language model pretraining.
    """
    
    @abstractmethod
    def __getitem__(self, index: int) -> str:
        """
        Get a single example from the dataset.
        
        Args:
            index: Index of the example to retrieve.
            
        Returns:
            Raw text string for pretraining.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            The size of the dataset.
        """
        pass


# ============================================================================
# SFT Dataset Implementations
# ============================================================================

class AlpacaSFTDataset(BaseDatasetSFT):
    """
    Alpaca dataset formatted for supervised fine-tuning.
    
    Converts instruction-input-output format to chat messages.
    """
    
    def __init__(self, split: str = "train"):
        super().__init__()
        if split != "train":
            raise ValueError(f"Split {split} not supported for Alpaca dataset")

        self.dataset = load_dataset("tatsu-lab/alpaca")[split]

    def __getitem__(self, index: int) -> List[Dict[str, str]]:
        row = self.dataset[index]
        instruction = row["instruction"]
        input_text = row["input"]
        
        # Combine instruction and input with a space if input is not empty
        if input_text:
            prompt = f"{instruction} {input_text}"
        else:
            prompt = instruction
        
        response = row["output"]

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return message

    def __len__(self) -> int:
        return len(self.dataset)


class MNLISFTDataset(BaseDatasetSFT):
    """
    MultiNLI dataset formatted for supervised fine-tuning
    using LM-style prompting (not a classifier head).
    """

    LABEL_MAP = {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }

    def __init__(self, split="train"):
        if split not in ["train", "validation_matched", "validation_mismatched"]:
            raise ValueError(f"Split {split} not supported for MultiNLI dataset")

        super().__init__()
        self.dataset = load_dataset("nyu-mll/multi_nli")[split]

    def __getitem__(self, index: int):
        row = self.dataset[index]

        premise = row["premise"].strip()
        hypothesis = row["hypothesis"].strip()
        label_id = row["label"]

        # Skip unlabeled examples if any (MNLI has some)
        if label_id == -1:
            raise IndexError("Unlabeled MNLI example")

        label_text = self.LABEL_MAP[label_id]

        messages = [
            {
                "role": "user",
                "content": (
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}\n\n"
                    "Does the premise entail, contradict, or is neutral "
                    "with respect to the hypothesis?\n\n"
                    "Answer with one word: entailment, neutral, or contradiction."
                )
            },
            {
                "role": "assistant",
                "content": f"The answer is: {label_text}"
            },
            {
                "role": "assistant",
                "content": label_text  # Just "entailment", "neutral", or "contradiction"
            }
        ]

        return messages

    def __len__(self):
        return len(self.dataset)

# ============================================================================
# Pretraining Dataset Implementations
# ============================================================================

class FineWebEduDataset(BaseDatasetPT):
    """
    FineWeb-Edu dataset for pretraining.

    Returns raw text strings from educational web content.
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            data_path: Path relative to BITLAB_DATA_DIR (default: fineweb-edu).
                      Use download_fineweb_edu() to create.
        """
        super().__init__()
        if split != "train": 
            raise ValueError(f"Split {split} not supported for FineWeb-Edu dataset")

        rel = "fineweb-edu"
        self.dataset = load_from_disk(os.path.join(get_data_dir(), rel))
        print(f"Loaded FineWeb-Edu dataset with {len(self.dataset)} documents")

    def __getitem__(self, index: int) -> str:
        """
        Returns:
            Raw text string for pretraining.
        """
        row = self.dataset[index]
        return row["text"]

    def __len__(self) -> int:
        return len(self.dataset)


class FalconRefinedWebDataset(BaseDatasetPT):
    """
    Falcon-RefinedWeb dataset for pretraining.

    Returns raw text strings from high-quality web content filtered
    and processed for the Falcon LLM.
    """

    def __init__(self, split: str = "train"):
        """
        Args:
            data_path: Path relative to BITLAB_DATA_DIR (default: falcon-refinedweb).
                      Use download_falcon_refinedweb() to create.
        """
        if split != "train": 
            raise ValueError(f"Split {split} not supported for FineWeb-Edu dataset")
        super().__init__()
        rel = "falcon-refinedweb" 
        self.dataset = load_from_disk(os.path.join(get_data_dir(), rel))
        print(f"Loaded Falcon-RefinedWeb dataset with {len(self.dataset)} documents")

    def __getitem__(self, index: int) -> str:
        """
        Returns:
            Raw text string for pretraining.
        """
        row = self.dataset[index]
        return row["content"]

    def __len__(self) -> int:
        return len(self.dataset)


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS_REGISTRY = {
    # SFT Datasets
    "alpaca": AlpacaSFTDataset,
    
    # Pretraining Datasets
    "fineweb-edu": FineWebEduDataset,
    "falcon-refinedweb": FalconRefinedWebDataset,
}

def load_bitlab_dataset(dataset_name: str, split: str = "train"): 
    if dataset_name not in DATASETS_REGISTRY: 
        raise ValueError(f"Dataset {dataset_name} not found")
    return DATASETS_REGISTRY[dataset_name](split=split)