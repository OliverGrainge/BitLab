from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.models.tokenizers import load_bitlab_tokenizer
from src.data.dataset import load_bitlab_dataset

class SFTDataModule(LightningDataModule): 
    def __init__(self, tokenizer_name: str, dataset_name: str, batch_size: int = 16, num_workers: int = 4, max_length: int = None):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer_name = str(tokenizer_name) 
        self.dataset_name = str(dataset_name)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_length = int(max_length) if max_length is not None else None

    def setup(self, stage: str):
        self.tokenizer = load_bitlab_tokenizer(self.tokenizer_name)
        self.dataset = load_bitlab_dataset(self.dataset_name, split="train")

    def train_dataloader(self): 
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        batch: List[List[Dict[str,str]]]
        each item like:
            [
            {"role":"system","content":...},
            {"role":"user","content":...},
            {"role":"assistant","content":...},
            ]
        """

        input_ids_list = []
        labels_list = []

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Many causal LMs don't define pad; EOS is a common fallback
            pad_id = self.tokenizer.eos_token_id

        for messages in batch:
            # Filter to get only system and user messages (everything before assistant)
            prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]
            
            # Full conversation including assistant
            full_messages = messages

            # Tokenize prompt with generation prompt
            # This gives us everything up to and including "<|im_start|>assistant\n"
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]

            # Tokenize full conversation without generation prompt
            # This includes the actual assistant response
            full_ids = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]

            # Build labels: 
            # - Mask everything up to where assistant response begins (the prompt)
            # - Keep the actual assistant response tokens for training
            prompt_length = prompt_ids.shape[0]
            labels = full_ids.clone()
            labels[:prompt_length] = -100  # Mask system, user, and assistant header

            # Truncate to max_length if specified (prevents OOM from very long sequences)
            if self.max_length is not None and full_ids.shape[0] > self.max_length:
                full_ids = full_ids[:self.max_length]
                labels = labels[:self.max_length]

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        # Pad to max length in batch
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



class AlpacaSFTDataModule(SFTDataModule):
    def __init__(self, tokenizer_name: str, batch_size: int = 16, num_workers: int = 4, max_length: int = None):
        super().__init__(tokenizer_name, "alpaca", batch_size, num_workers, max_length)


class MNLISFTDataModule(SFTDataModule): 
    def __init__(self, tokenizer_name: str, batch_size: int = 16, num_workers: int = 4, max_length: int = None):
        super().__init__(tokenizer_name, "mnli", batch_size, num_workers, max_length)

class PretrainingDataModule(LightningDataModule):
    """
    DataModule for causal language model pretraining.
    
    Unlike SFT which uses chat templates, pretraining:
    - Tokenizes raw text directly
    - Uses all tokens for training (no masking)
    - Creates labels by shifting input_ids by 1
    """
    
    def __init__(
        self, 
        tokenizer_name: str, 
        dataset_name: str, 
        batch_size: int = 16, 
        num_workers: int = 4, 
        max_length: int = 512,
        stride: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer_name = str(tokenizer_name)
        self.dataset_name = str(dataset_name)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_length = int(max_length)
        # Stride for chunking long documents (if None, defaults to max_length)
        self.stride = int(stride) if stride is not None else self.max_length

    def setup(self, stage: str):
        self.tokenizer = load_bitlab_tokenizer(self.tokenizer_name)
        self.dataset = load_bitlab_dataset(self.dataset_name, split="train")

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # Shuffle for pretraining
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        """
        Batch collation for pretraining.
        
        Args:
            batch: List[str] - raw text strings
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # Tokenize all texts in batch
        # We use truncation and padding to ensure consistent length
        encoding = self.tokenizer(
            batch,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Create labels by shifting input_ids
        # For causal LM: predict token i using tokens 0...i-1
        # So labels[i] = input_ids[i], but we mask padding tokens
        labels = input_ids.clone()
        
        # Mask padding tokens in labels (they shouldn't contribute to loss)
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class FineWebEduPTDataModule(PretrainingDataModule):
    """Convenience class for FineWeb-Edu pretraining."""
    
    def __init__(
        self, 
        tokenizer_name: str, 
        batch_size: int = 16, 
        num_workers: int = 4, 
        max_length: int = 512,
        stride: int = None,
    ):
        super().__init__(
            tokenizer_name, 
            "fineweb-edu", 
            batch_size, 
            num_workers, 
            max_length,
            stride,
        )


class FalconRefinedWebPTDataModule(PretrainingDataModule):
    """Convenience class for Falcon-RefinedWeb pretraining."""
    
    def __init__(
        self, 
        tokenizer_name: str, 
        batch_size: int = 16, 
        num_workers: int = 4, 
        max_length: int = 512,
        stride: int = None,
    ):
        super().__init__(
            tokenizer_name, 
            "falcon-refinedweb", 
            batch_size, 
            num_workers, 
            max_length,
            stride,
        )


# Add to registry
DATALOADERS_REGISTRY = {
    "alpaca-sft": AlpacaSFTDataModule,
    "mnli-sft": MNLISFTDataModule,

    # Pretraining DataModules 
    "fineweb-edu-pt": FineWebEduPTDataModule,
    "falcon-refinedweb-pt": FalconRefinedWebPTDataModule,
}


def load_bitlab_datamodule(datamodule_name: str, **kwargs):
    if datamodule_name not in DATALOADERS_REGISTRY:
        raise ValueError(f"DataModule {datamodule_name} not found")
    return DATALOADERS_REGISTRY[datamodule_name](**kwargs)


if __name__ == "__main__":
    # Test the pretraining data module
    datamodule = FineWebEduPTDataModule(
        tokenizer_name="qwen2_5_05_instruct",
        batch_size=4,
        num_workers=0,
        max_length=128,
    )
    datamodule.setup("fit")
    tokenizer = datamodule.tokenizer
    dl = datamodule.train_dataloader()
    
    print("=== PRETRAINING DATA MODULE TEST ===\n")
    
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= 2:  # Only show 2 batches
            break
            
        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}")
        print(f"{'='*60}")
        
        # Show first example in batch
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        attention_mask = batch["attention_mask"][0]
        
        print(f"\nSequence length: {len(input_ids)}")
        print(f"Non-padding tokens: {attention_mask.sum().item()}")
        print(f"Training tokens (labels != -100): {(labels != -100).sum().item()}")
        
        # Decode full sequence
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print("\n=== FULL DECODED TEXT ===")
        print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
        
        # Show token-by-token for first 20 tokens
        print("\n=== TOKEN-BY-TOKEN BREAKDOWN (first 20 tokens) ===")
        for i in range(min(20, len(input_ids))):
            token_id = input_ids[i].item()
            label_id = labels[i].item()
            is_padding = attention_mask[i].item() == 0
            
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            status = "PAD" if is_padding else "TRAIN"
            
            print(f"{i:3d} | {status:5s} | Token ID: {token_id:5d} | Label: {label_id:5d} | {repr(token_text)}")
        
        print("\n" + "="*60)