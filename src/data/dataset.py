from abc import ABC, abstractmethod
from typing import Any, Dict
from datasets import load_dataset 

from torch.utils.data import Dataset


class BaseDatasetSFT(Dataset, ABC):
    """
    Abstract base class for supervised fine-tuning datasets.
    
    Subclasses must implement methods to provide prompt-response pairs
    for training language models.
    """
    
    @abstractmethod
    def __getitem__(self, index: int) -> list:
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
    


class AlpacaSFTDataset(BaseDatasetSFT): 
    def __init__(self): 
        super().__init__()
        self.dataset = load_dataset("tatsu-lab/alpaca")["train"]

    def __getitem__(self, index: int) -> Dict[str, Any]:
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


DATASETS_REGISTRY = {
    "alpaca": AlpacaSFTDataset,
}

