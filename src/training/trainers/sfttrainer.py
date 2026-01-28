import copy
from collections import defaultdict
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from src.models.models import load_bitlab_model


class SFTTrainer(pl.LightningModule):
    def __init__(self, model_name, learning_rate=5e-5, weight_decay=0.00):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = str(model_name)
        self.model = load_bitlab_model(model_name)
        
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        
        # Loss function
        self.ce_loss_fn = CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits
    
    def compute_loss(self, logits, labels):
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = self.loss_fn(shift_logits, shift_labels)
        return loss
    
    def training_step(self, batch, batch_idx):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        loss = self.compute_loss(logits, batch["labels"])
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        loss = self.compute_loss(logits, batch["labels"])
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
