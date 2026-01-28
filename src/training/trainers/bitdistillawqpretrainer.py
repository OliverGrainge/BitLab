import copy
from typing import List, Dict, Optional, Tuple, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from bitcore import BitLinear
from src.models.models import load_bitlab_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class RMSNormNoParam(nn.Module):
    """RMS Normalization without learnable parameters."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def __repr__(self) -> str: 
        return f"RMSNormNoParam(dim={self.dim}, eps={self.eps})"


def awq_prequantize_weight(
    W: torch.Tensor,
    X: torch.Tensor,
    *,
    n_bits: int = 4,
    group_size: int = 128,
    alpha_grid: int = 20,
    clip_ratio: Optional[float] = None,
    eps: float = 1e-8,
    max_eval_tokens: int = 4096,
) -> torch.Tensor:
    """
    AWQ (Activation-aware Weight Quantization) with grid-search over alpha.
    
    Quantizes W*diag(s) with groupwise symmetric quantization, then returns
    the effective dequantized weight W_eff = dequant(W*diag(s)) * diag(s)^-1.
    
    Args:
        W: Weight matrix [out_features, in_features]
        X: Activation samples [..., in_features]
        n_bits: Number of bits for quantization (2-8)
        group_size: Size of groups for groupwise quantization
        alpha_grid: Number of alpha values to search over
        clip_ratio: Optional clipping ratio for quantization range
        eps: Small epsilon for numerical stability
        max_eval_tokens: Maximum tokens to use for evaluation
    
    Returns:
        W_eff: Effective dequantized weight [out_features, in_features]
    """
    assert W.dim() == 2, "W must be [out_features, in_features]"
    assert X.shape[-1] == W.shape[1], "X last dim must match W in_features"
    assert 2 <= n_bits <= 8, "n_bits should be in [2, 8]"
    assert group_size > 0, "group_size must be positive"

    out_features, in_features = W.shape
    device = W.device
    dtype = W.dtype
    qmax = (1 << (n_bits - 1)) - 1

    # Prepare activation data
    # X may be on CPU, so we'll move only the needed subset to device
    X2 = X.reshape(-1, in_features)
    if X2.shape[0] > max_eval_tokens:
        # Sample indices on CPU (works regardless of X device)
        idx = torch.randint(0, X2.shape[0], (max_eval_tokens,), device=X2.device)
        X2 = X2[idx]
    
    # Move the sampled activations to the weight's device for computation
    X2 = X2.to(device)

    act_scale = torch.clamp(X2.abs().mean(dim=0), min=eps)
    X_for_obj = X2.transpose(0, 1).contiguous()
    Y_fp = (W @ X_for_obj).to(torch.float32)

    def quantize_groupwise_symm(Wf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Quantize weight matrix using groupwise symmetric quantization."""
        in_ = Wf.shape[1]
        n_groups = (in_ + group_size - 1) // group_size
        pad = n_groups * group_size - in_
        
        Wp = torch.nn.functional.pad(Wf, (0, pad)) if pad else Wf
        Wg = Wp.view(out_features, n_groups, group_size)

        max_abs = Wg.abs().amax(dim=2)
        if clip_ratio is not None:
            max_abs = max_abs * float(clip_ratio)

        scale = torch.clamp(max_abs / qmax, min=eps)
        q = torch.round(Wg / scale.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8)
        q = q.view(out_features, n_groups * group_size)
        
        if pad:
            q = q[:, :in_]
        return q, scale, n_groups, pad

    def dequantize_groupwise_symm(
        q: torch.Tensor, 
        scale: torch.Tensor, 
        n_groups: int, 
        pad: int
    ) -> torch.Tensor:
        """Dequantize weight matrix."""
        in_ = q.shape[1]
        qp = torch.nn.functional.pad(q, (0, pad)) if pad else q
        qg = qp.view(out_features, n_groups, group_size).to(torch.float32)
        Wg = qg * scale.unsqueeze(-1).to(torch.float32)
        Wp = Wg.view(out_features, n_groups * group_size)
        
        if pad:
            Wp = Wp[:, :in_]
        return Wp.to(dtype)

    # Grid search over alpha
    alphas = torch.linspace(0.0, 1.0, steps=alpha_grid, device=device)
    best_loss = float('inf')
    best_params = None

    for alpha in alphas:
        s = torch.clamp(torch.pow(act_scale, alpha), min=eps)
        W_scaled = W * s.unsqueeze(0)
        X_scaled = X_for_obj / s.unsqueeze(1)

        q, scale, ng, pad = quantize_groupwise_symm(W_scaled)
        W_hat_scaled = dequantize_groupwise_symm(q, scale, ng, pad)
        Y_q = (W_hat_scaled @ X_scaled).to(torch.float32)
        loss = torch.mean((Y_q - Y_fp) ** 2).item()

        if loss < best_loss:
            best_loss = loss
            best_params = (s, q, scale, ng, pad)

    # Unpack best parameters and compute effective weight
    best_s, best_q, best_scale, best_ng, best_pad = best_params
    W_hat_scaled = dequantize_groupwise_symm(best_q, best_scale, best_ng, best_pad)
    W_eff = W_hat_scaled / best_s.unsqueeze(0).to(dtype)

    return W_eff


class BitDistillAWQPreTrainer(pl.LightningModule):
    """
    Stage 1: Continual Pretraining with BitLinear quantization.
    
    Supports both QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization)
    with AWQ activation-aware weight quantization.
    """
    
    def __init__(
        self, 
        model_name: str, 
        learning_rate: float = 5e-5, 
        weight_decay: float = 0.0, 
        target_quant_modules: List[str] = None, 
        target_subln_modules: Optional[List[str]] = None,
        calibration_samples: int = 128,
        quant_type: str = "bitnet",
        n_bit_ptq: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        assert target_quant_modules is not None and len(target_quant_modules) > 0, \
            "Must specify at least one layer pattern to quantize in target_quant_modules"
        
        self.model_name = str(model_name)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.calibration_samples = int(calibration_samples)
        self.target_quant_modules = target_quant_modules
        self.target_subln_modules = target_subln_modules or []
        self.quant_type = quant_type
        self.total_tokens_seen = 0
        self.n_bit_ptq = int(n_bit_ptq)
        self.model = load_bitlab_model(model_name)
        self.model.train()
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

    def collect_activations(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Collect activation samples for AWQ calibration in eval mode.
        Preserves and restores the model's original training state.
        
        Returns:
            Dictionary mapping module names to their activation tensors.
        """
        # Save original training state
        was_training = self.model.training
        
        # Switch to eval mode for calibration
        self.model.eval()
        
        activations = {}
        modules_to_collect = self.get_target_linear_modules()

        def make_hook(name: str):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = input[0].detach().cpu()
            return hook

        # Register hooks
        hooks = []
        for name, module, _ in modules_to_collect:
            hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            sample_count = self.calibration_samples

            # Progress bar is based on number of samples, not batches
            use_tqdm = getattr(self, "trainer", None) is None or self.trainer.is_global_zero
            pbar = tqdm(
                total=sample_count,
                desc="[PTQ] Collecting activations",
            ) if use_tqdm else None

            with torch.no_grad():
                for batch in dataloader:
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()
                    }
                    
                    _ = self.model(**batch)
                    
                    batch_size = batch["input_ids"].shape[0]
                    sample_count -= batch_size

                    if pbar is not None:
                        # Advance by the true number of samples consumed
                        pbar.update(batch_size)
                    
                    if sample_count <= 0:
                        break
        finally:
            # Always restore original state and remove hooks
            for h in hooks:
                h.remove()
            
            # Restore original training mode
            if was_training:
                self.model.train()
            else:
                self.model.eval()

        # Keep activations on CPU to avoid OOM
        # They will be moved to device only when needed in awq_prequantize_weight
        
        return activations

    def prepare_ptq(self) -> None:
        """Apply AWQ Post-Training Quantization to weight matrices."""
        if not self.trainer.is_global_zero:
            return

        activations = self.collect_activations(self.trainer.datamodule.train_dataloader())
        
        modules = self.get_target_linear_modules()
        iterator = modules
        # Only show tqdm bar on global rank zero to avoid duplicates
        if self.trainer.is_global_zero:
            iterator = tqdm(modules, desc="[PTQ] AWQ quantization")

        for name, module, needs_subln in iterator:
            if name not in activations:
                continue
                
            with torch.no_grad():
                quantized_weight = awq_prequantize_weight(
                    module.weight.data, 
                    activations[name],
                    n_bits=self.n_bit_ptq
                )
                module.weight.data = quantized_weight

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
        self.prepare_ptq()
        self.prepare_qat()
        
        dtype = self._get_trainer_compute_dtype()
        self.model.to(device=self.device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for next-token prediction."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
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
        prefix: str
    ) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self._compute_ce_loss(logits, labels)
        perplexity = torch.exp(loss)
        
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_perplexity", perplexity, on_step=False, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single training step."""
        num_tokens = self._count_tokens(batch["labels"])
        self.total_tokens_seen += num_tokens
        
        loss = self._shared_step(batch, batch_idx, "train")
        
        self.log("tokens_seen", float(self.total_tokens_seen), 
                 on_step=True, on_epoch=False, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single validation step."""
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        return optimizer