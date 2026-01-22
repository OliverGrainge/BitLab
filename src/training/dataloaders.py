from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.models.tokenizers import TOKENIZERS_REGISTRY
from src.data.dataset import DATASETS_REGISTRY

class SFTDataModule(LightningDataModule): 
    def __init__(self, model_name: str, dataset_name: str, batch_size: int = 16, num_workers: int = 4, max_length: int = None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = str(model_name) 
        self.dataset_name = str(dataset_name)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_length = int(max_length) if max_length is not None else None

    def setup(self, stage: str):
        self.tokenizer = TOKENIZERS_REGISTRY[self.model_name]()
        self.dataset = DATASETS_REGISTRY[self.dataset_name]()

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
            # prompt = everything up to (but not including) assistant answer
            prompt_messages = messages[:-1]         # system + user
            full_messages = messages               # system + user + assistant

            # Tokenize prompt *with* generation prompt so the template ends at assistant-start
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]

            # Tokenize full conversation *without* generation prompt (assistant text is already present)
            full_ids = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]

            # Build labels: ignore prompt tokens, learn only assistant tokens
            cut = prompt_ids.shape[0]
            labels = full_ids.clone()
            labels[:cut] = -100

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
    def __init__(self, model_name: str, batch_size: int = 16, num_workers: int = 4, max_length: int = None):
        super().__init__(model_name, "alpaca", batch_size, num_workers, max_length)

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