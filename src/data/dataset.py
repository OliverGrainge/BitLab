from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset


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
    
    def __init__(self):
        super().__init__()
        self.dataset = load_dataset("tatsu-lab/alpaca")["train"]

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


# ============================================================================
# Pretraining Dataset Implementations
# ============================================================================

class FineWebEduDataset(BaseDatasetPT):
    """
    FineWeb-Edu dataset for pretraining.
    
    Returns raw text strings from educational web content.
    """
    
    def __init__(self, data_path: str = "data/fineweb-edu"):
        """
        Args:
            data_path: Path to the saved FineWeb-Edu dataset on disk.
                      Should be created using download_fineweb_edu().
        """
        super().__init__()
        self.dataset = load_from_disk(data_path)
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


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS_REGISTRY = {
    # SFT Datasets
    "alpaca": AlpacaSFTDataset,
    
    # Pretraining Datasets
    "fineweb-edu": FineWebEduDataset,
}