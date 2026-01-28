import copy
from typing import List, Dict, Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from bitcore import BitLinear
from tqdm.auto import tqdm

from src.models.models import load_bitlab_model


class RMSNormNoParam(nn.Module):
    """RMS Normalization without learnable parameters."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def __repr__(self) -> str:
        return f"RMSNormNoParam(dim={self.dim}, eps={self.eps})"


class BitDistillPreTrainer(pl.LightningModule):
    """
    Stage 1: Continual Pretraining with BitLinear quantization
    Uses only cross-entropy loss on the quantized model (no teacher-student setup).
    """
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        target_quant_modules: List[str] = None,
        target_subln_modules: List[str] = None,
        quant_type: str = "bitnet",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Hyperparameter Validation
        assert target_quant_modules is not None and len(target_quant_modules) > 0, \
            "Must specify at least one layer pattern to quantize in target_quant_modules"
        
        # Store hyperparameters
        self.model_name = str(model_name)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.target_quant_modules = target_quant_modules
        self.target_subln_modules = target_subln_modules or []
        self.quant_type = quant_type
        
        # Token tracking
        self.total_tokens_seen = 0
        
        # Initialize model
        self.model = load_bitlab_model(model_name)
        
        # Ensure model is in train mode
        self.model.train()
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def get_target_linear_modules(self) -> List[Tuple[str, nn.Module, bool]]:
        """
        Returns a list of (module_name, module, needs_subln) tuples for quantization.
        """
        results = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(pattern in name for pattern in self.target_quant_modules):
                continue
            needs_subln = any(pattern in name for pattern in self.target_subln_modules)
            results.append((name, module, needs_subln))
        return results

    def _set_module_by_name(self, name: str, module: nn.Module) -> None:
        """Replace a module in the model by its dotted name."""
        parts = name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    def prepare_qat(self) -> None:
        """Replace Linear layers with BitLinear for Quantization-Aware Training."""
        modules_to_replace = self.get_target_linear_modules()
        iterator = modules_to_replace
        # Only show tqdm bar on global rank zero to avoid duplicates
        if getattr(self, "trainer", None) is None or self.trainer.is_global_zero:
            iterator = tqdm(modules_to_replace, desc="[QAT] Quantizing BitLinear layers")

        for name, module, needs_subln in iterator:
            bitlinear = BitLinear.from_linear(module, quant_type=self.quant_type)

            if needs_subln:
                new_module = nn.Sequential(
                    RMSNormNoParam(bitlinear.in_features),
                    bitlinear
                )
            else:
                new_module = bitlinear

            self._set_module_by_name(name, new_module)


    def _get_trainer_compute_dtype(self) -> torch.dtype:
        """Determine the compute dtype from trainer precision settings."""
        p = getattr(self.trainer, "precision", None)

        if p is None:
            p = getattr(getattr(self.trainer, "precision_plugin", None), "precision", None)
        if p is None:
            plugin = getattr(getattr(self.trainer, "strategy", None), "precision_plugin", None)
            p = getattr(plugin, "precision", None)

        # Map precision to dtype
        if p in (None, 32, "32", "32-true", "32_true"):
            return torch.float32
        if p in (64, "64", "64-true", "64_true"):
            return torch.float64
        if p in ("bf16", "bf16-true", "bf16_true", "bf16-mixed", "bf16_mixed"):
            return torch.bfloat16
        if p in (16, "16", "16-true", "16_true", "16-mixed", "16_mixed"):
            return torch.float16

        return torch.float32

    def on_fit_start(self) -> None:
        """Initialize quantization before training starts."""
        self.prepare_qat()
        dtype = self._get_trainer_compute_dtype()
        self.model.to(device=self.device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for next-token prediction."""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        batch_size, seq_len, vocab_size = shift_logits.shape
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        return self.ce_loss(shift_logits, shift_labels)

    def _count_tokens(self, labels: torch.Tensor) -> int:
        """Count the number of valid (non-padding) tokens in the batch."""
        return (labels != -100).sum().item()

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        prefix: str,
    ) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self._compute_ce_loss(logits, labels)
        perplexity = torch.exp(loss)

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single training step."""
        # Count tokens in this batch
        num_tokens = self._count_tokens(batch["labels"])
        self.total_tokens_seen += num_tokens

        loss = self._shared_step(batch, batch_idx, "train")

        # Log total tokens seen (matches AWQ pretrainer).
        self.log("tokens_seen", float(self.total_tokens_seen), on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single validation step."""
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Any:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer