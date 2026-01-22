from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
import sys
from pathlib import Path


from src.models.tokenizers import TOKENIZERS_REGISTRY
from src.data.dataset import DATASETS_REGISTRY

class SFTDataModule(LightningDataModule): 
    def __init__(self, model_name: str, dataset_name: str, batch_size: int = 16, num_workers: int = 4):
        super().__init__()
        self.model_name = str(model_name) 
        self.dataset_name = str(dataset_name)
        self.dataset_name = str(dataset_name)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def setup(self, stage: str):
        self.tokenizer = TOKENIZERS_REGISTRY[self.model_name]()
        self.dataset = DATASETS_REGISTRY[self.dataset_name]()

    def train_dataloader(self): 
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch): 
        prompts = [b["prompt"] for b in batch]  # Fixed typo
        responses = [b["response"] for b in batch]
        
        # Tokenize prompts and responses separately to track lengths
        prompt_encodings = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=512
        )
        
        response_encodings = self.tokenizer(
            responses,
            add_special_tokens=False,  # Don't duplicate special tokens
            padding=False,
            truncation=True,
            max_length=512
        )
        
        # Concatenate prompt + response for each example
        input_ids = []
        labels = []
        
        for prompt_ids, response_ids in zip(prompt_encodings["input_ids"], response_encodings["input_ids"]):
            # Combine prompt and response
            combined_ids = prompt_ids + response_ids
            input_ids.append(combined_ids)
            
            # Create labels: -100 for prompt (ignored), actual tokens for response
            label_ids = [-100] * len(prompt_ids) + response_ids
            labels.append(label_ids)
        
        # Pad to same length
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, lab in zip(input_ids, labels):
            padding_length = max_len - len(ids)
            
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_labels.append(lab + [-100] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }



class AlpacaSFTDataModule(SFTDataModule):
    def __init__(self, model_name: str, batch_size: int = 16, num_workers: int = 4):
        super().__init__(model_name, "alpaca", batch_size, num_workers)

DATALOADERS_REGISTRY = {
    "alpaca-sft": AlpacaSFTDataModule,
}


if __name__ == "__main__":
    datamodule = AlpacaSFTDataModule(model_name="qwen2_5_05_instruct", batch_size=16, num_workers=4)
    datamodule.setup("fit")
    dl = datamodule.train_dataloader()
    for batch in dl:
        print(batch)
        break