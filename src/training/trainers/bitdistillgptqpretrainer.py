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


@torch.no_grad()
def gptq_prequantize_weight(
    W: torch.Tensor,
    X: torch.Tensor,
    *,
    n_bits: int = 4,
    group_size: int = 128,
    blocksize: int = 128,
    damp_percent: float = 0.01,
    eps: float = 1e-8,
    max_eval_tokens: int = 4096,
) -> torch.Tensor:
    """
    GPTQ (Frantar et al., ICLR 2023): approximate second-order, one-shot weight quantization.

    This implements the core layer-wise GPTQ procedure (Algorithm 1) with:
      - Hessian approximation from calibration activations X
      - Dampening (lambda on diagonal)
      - Cholesky reformulation for numerical stability
      - Blockwise ("lazy") updates for speed
      - Per-row, per-group asymmetric uniform quantization (min-max grid), which is
        the paper's default quantizer choice; group parameters are recomputed using
        the *current* updated weights as GPTQ progresses.

    Returns:
        W_q: dequantized effective weights (same shape/dtype/device as W).

    Notes:
      - W is [out_features, in_features]
      - X is activation samples [..., in_features]
      - This returns dequantized weights; if you also want packed ints/scales/zps, you
        can extend it, but this keeps the same "returns weights" contract as your AWQ helper.

    Paper reference: GPTQ Algorithm 1 + dampening + Cholesky variant.  [oai_citation:1‡gptq.pdf](sediment://file_00000000537071f483273703ff502479)
    """
    assert W.dim() == 2, "W must be [out_features, in_features]"
    assert X.shape[-1] == W.shape[1], "X last dim must match W in_features"
    assert 2 <= n_bits <= 8, "n_bits should be in [2, 8]"
    assert group_size > 0, "group_size must be positive"
    assert blocksize > 0, "blocksize must be positive"
    assert 0.0 <= damp_percent <= 1.0, "damp_percent should be in [0, 1]"

    orig_device = W.device
    dtype = W.dtype
    device = torch.device("cuda") if torch.cuda.is_available() else orig_device
    # Move to compute device; preserve W's dtype throughout (Q returned in same precision as W).
    W = W.to(device=device, dtype=dtype)
    X = X.to(device)

    out_features, in_features = W.shape
    qlevels = 1 << n_bits  # 2^b

    # ---------
    # 1) Prepare activation matrix and Hessian approx H = X^T X (+ damp)
    # ---------
    X2 = X.reshape(-1, in_features)
    if X2.shape[0] > max_eval_tokens:
        idx = torch.randint(0, X2.shape[0], (max_eval_tokens,), device=X2.device)
        X2 = X2[idx]
    X2 = X2.to(device=device, dtype=torch.float32)

    # Hessian approx for objective ||WX - WqX||_F^2 per layer:
    # H ~ X^T X (scaling constant doesn't change the minimizer; damping uses diag stats)
    # Shape: [in_features, in_features]
    H = X2.t().matmul(X2)

    # Dampening: add lambda to diagonal, typically 1% of avg diagonal (paper).  [oai_citation:2‡gptq.pdf](sediment://file_00000000537071f483273703ff502479)
    diag_mean = torch.mean(torch.diag(H))
    lam = float(damp_percent) * float(diag_mean)
    if lam > 0:
        H = H + lam * torch.eye(in_features, device=device, dtype=torch.float32)

    # Compute H^{-1} robustly via Cholesky, then take Cholesky of H^{-1} and transpose
    # to match the paper's reformulation.  [oai_citation:3‡gptq.pdf](sediment://file_00000000537071f483273703ff502479)
    try:
        L = torch.linalg.cholesky(H)  # H = L L^T
        Hinv = torch.cholesky_inverse(L)  # H^{-1}
    except RuntimeError:
        # Fallback: explicit inverse (less stable)
        Hinv = torch.linalg.pinv(H)

    # U is upper-triangular such that Hinv = U^T U, i.e. U = chol(Hinv)^T
    try:
        U = torch.linalg.cholesky(Hinv).transpose(0, 1).contiguous()
    except RuntimeError:
        # If Hinv isn't numerically SPD due to edge cases, add a tiny jitter and retry
        jitter = (eps + 1e-6) * torch.eye(in_features, device=device, dtype=torch.float32)
        U = torch.linalg.cholesky(Hinv + jitter).transpose(0, 1).contiguous()

    # Working copy of weights in float32 for stable updates
    W_work = W.to(torch.float32).clone()

    # Output dequantized weights
    Q = torch.empty_like(W_work)

    # ---------
    # 2) Quantizer: per-row, per-group asymmetric uniform (min-max) grid
    #    Params recomputed from *current* W_work (important when grouping).  [oai_citation:4‡gptq.pdf](sediment://file_00000000537071f483273703ff502479)
    # ---------
    def quant_dequant_column_asymm_grouped(
        Wcur: torch.Tensor,  # [out, in]
        col_j: int,
    ) -> torch.Tensor:
        """Quantize+dequantize column j using per-row params of its group (min-max)."""
        g0 = (col_j // group_size) * group_size
        g1 = min(g0 + group_size, in_features)

        # group slice per row
        G = Wcur[:, g0:g1]  # [out, g]

        wmin = G.amin(dim=1)
        wmax = G.amax(dim=1)

        # avoid zero range
        scale = (wmax - wmin) / float(qlevels - 1)
        scale = torch.clamp(scale, min=eps)

        # zero-point for asymmetric quant: q = round(w/scale) + zp, with zp = round(-min/scale)
        zp = torch.round(-wmin / scale).clamp(0, qlevels - 1)

        w = Wcur[:, col_j]
        q = torch.round(w / scale + zp).clamp(0, qlevels - 1)
        w_hat = (q - zp) * scale
        return w_hat

    # ---------
    # 3) GPTQ main loop (Algorithm 1): blockwise, column-by-column, with error compensation  [oai_citation:5‡gptq.pdf](sediment://file_00000000537071f483273703ff502479)
    # ---------
    B = int(blocksize)
    for i in range(0, in_features, B):
        i_end = min(i + B, in_features)
        bsz = i_end - i

        # E will store quantization errors for columns in this block: [out, bsz]
        E = torch.zeros((out_features, bsz), device=device, dtype=torch.float32)

        # process columns within the block
        for jj, j in enumerate(range(i, i_end)):
            # quantize this column (on current updated W_work)
            qcol = quant_dequant_column_asymm_grouped(W_work, j)
            Q[:, j] = qcol

            # quantization error scaled by diagonal term (paper uses [H^{-1}]jj in chol form)  [oai_citation:6‡gptq.pdf](sediment://file_00000000537071f483273703ff502479)
            denom = torch.clamp(U[j, j], min=eps)
            e = (W_work[:, j] - qcol) / denom
            E[:, jj] = e

            # update weights within current block (including this col onward)
            # W[:, j:i_end] -= e[:,None] * U[j, j:i_end]
            U_row_block = U[j, j:i_end].unsqueeze(0)  # [1, bsz - (j-i)]
            W_work[:, j:i_end] -= e.unsqueeze(1) * U_row_block

        # after finishing block, update remaining columns to the right
        if i_end < in_features:
            # W[:, i_end:] -= E @ U[i:i_end, i_end:]
            W_work[:, i_end:] -= E.matmul(U[i:i_end, i_end:])

    # Same shape, dtype, and device as W (precision format preserved).
    return Q.to(device=orig_device, dtype=dtype)


class BitDistillGPTQPreTrainer(pl.LightningModule):
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
            sample_count = 24

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
            iterator = tqdm(modules, desc="[PTQ] GPTQ quantization")

        for name, module, needs_subln in iterator:
            if name not in activations:
                continue
                
            with torch.no_grad():
                quantized_weight = gptq_prequantize_weight(
                    W=module.weight.data, 
                    X=activations[name],
                    n_bits=self.n_bit_ptq
                )
                module.weight.data = quantized_weight.to(device=self.device, dtype=self.dtype)

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