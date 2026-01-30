"""
BitDistill trainer: continual pretraining with BitLinear quantization.

Single trainer with configurable PTQ initialization methods:
- ptq_method=None: QAT only (no PTQ initialization)
- ptq_method="absmax": Per-row absmax PTQ then QAT
- ptq_method="awq": Activation-aware weight quantization PTQ then QAT
- ptq_method="gptq": Second-order GPTQ PTQ then QAT
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from bitcore import BitLinear
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.models import load_bitlab_model
from src.models.tokenizers import load_bitlab_tokenizer


# -----------------------------------------------------------------------------
# Shared components
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# PTQ weight functions
# -----------------------------------------------------------------------------


@torch.no_grad()
def ptq_prequantize_weight(
    W: torch.Tensor,
    n_bits: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fake-quantize + dequantize an nn.Linear weight with symmetric absmax
    quantization per output channel (per row).
    """
    assert W.dim() == 2, "W must be [out_features, in_features]"
    assert 2 <= n_bits <= 8, "n_bits must be in [2, 8]"
    qmax = (1 << (n_bits - 1)) - 1
    row_absmax = W.abs().amax(dim=1)
    scale = (row_absmax / qmax).clamp_min(eps).unsqueeze(1)
    q = torch.round(W / scale).clamp(-qmax, qmax)
    return (q * scale).to(W.dtype)


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
    Returns effective dequantized weight W_eff.
    """
    assert W.dim() == 2, "W must be [out_features, in_features]"
    assert X.shape[-1] == W.shape[1], "X last dim must match W in_features"
    assert 2 <= n_bits <= 8, "n_bits should be in [2, 8]"
    assert group_size > 0, "group_size must be positive"

    out_features, in_features = W.shape
    device = W.device
    dtype = W.dtype
    qmax = (1 << (n_bits - 1)) - 1

    X2 = X.reshape(-1, in_features)
    if X2.shape[0] > max_eval_tokens:
        idx = torch.randint(0, X2.shape[0], (max_eval_tokens,), device=X2.device)
        X2 = X2[idx]
    X2 = X2.to(device)
    act_scale = torch.clamp(X2.abs().mean(dim=0), min=eps)
    X_for_obj = X2.transpose(0, 1).contiguous()
    Y_fp = (W @ X_for_obj).to(torch.float32)

    def quantize_groupwise_symm(Wf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
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
        q: torch.Tensor, scale: torch.Tensor, n_groups: int, pad: int
    ) -> torch.Tensor:
        in_ = q.shape[1]
        qp = torch.nn.functional.pad(q, (0, pad)) if pad else q
        qg = qp.view(out_features, n_groups, group_size).to(torch.float32)
        Wg = qg * scale.unsqueeze(-1).to(torch.float32)
        Wp = Wg.view(out_features, n_groups * group_size)
        if pad:
            Wp = Wp[:, :in_]
        return Wp.to(dtype)

    alphas = torch.linspace(0.0, 1.0, steps=alpha_grid, device=device)
    best_loss = float("inf")
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
    best_s, best_q, best_scale, best_ng, best_pad = best_params
    W_hat_scaled = dequantize_groupwise_symm(best_q, best_scale, best_ng, best_pad)
    W_eff = W_hat_scaled / best_s.unsqueeze(0).to(dtype)
    return W_eff


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
    GPTQ: approximate second-order, one-shot weight quantization.
    Returns dequantized effective weights (same shape/dtype/device as W).
    """
    assert W.dim() == 2, "W must be [out_features, in_features]"
    assert X.shape[-1] == W.shape[1], "X last dim must match W in_features"
    assert 2 <= n_bits <= 8, "n_bits should be in [2, 8]"
    assert group_size > 0 and blocksize > 0 and 0.0 <= damp_percent <= 1.0

    orig_device = W.device
    dtype = W.dtype
    device = torch.device("cuda") if torch.cuda.is_available() else orig_device
    W = W.to(device=device, dtype=dtype)
    X = X.to(device)

    out_features, in_features = W.shape
    qlevels = 1 << n_bits

    X2 = X.reshape(-1, in_features)
    if X2.shape[0] > max_eval_tokens:
        idx = torch.randint(0, X2.shape[0], (max_eval_tokens,), device=X2.device)
        X2 = X2[idx]
    X2 = X2.to(device=device, dtype=torch.float32)
    H = X2.t().matmul(X2)
    diag_mean = torch.mean(torch.diag(H))
    lam = float(damp_percent) * float(diag_mean)
    if lam > 0:
        H = H + lam * torch.eye(in_features, device=device, dtype=torch.float32)
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except RuntimeError:
        Hinv = torch.linalg.pinv(H)
    try:
        U = torch.linalg.cholesky(Hinv).transpose(0, 1).contiguous()
    except RuntimeError:
        jitter = (eps + 1e-6) * torch.eye(in_features, device=device, dtype=torch.float32)
        U = torch.linalg.cholesky(Hinv + jitter).transpose(0, 1).contiguous()

    W_work = W.to(torch.float32).clone()
    Q = torch.empty_like(W_work)

    def quant_dequant_column_asymm_grouped(Wcur: torch.Tensor, col_j: int) -> torch.Tensor:
        g0 = (col_j // group_size) * group_size
        g1 = min(g0 + group_size, in_features)
        G = Wcur[:, g0:g1]
        wmin = G.amin(dim=1)
        wmax = G.amax(dim=1)
        scale = (wmax - wmin) / float(qlevels - 1)
        scale = torch.clamp(scale, min=eps)
        zp = torch.round(-wmin / scale).clamp(0, qlevels - 1)
        w = Wcur[:, col_j]
        q = torch.round(w / scale + zp).clamp(0, qlevels - 1)
        return (q - zp) * scale

    B = int(blocksize)
    for i in range(0, in_features, B):
        i_end = min(i + B, in_features)
        bsz = i_end - i
        E = torch.zeros((out_features, bsz), device=device, dtype=torch.float32)
        for jj, j in enumerate(range(i, i_end)):
            qcol = quant_dequant_column_asymm_grouped(W_work, j)
            Q[:, j] = qcol
            denom = torch.clamp(U[j, j], min=eps)
            e = (W_work[:, j] - qcol) / denom
            E[:, jj] = e
            U_row_block = U[j, j:i_end].unsqueeze(0)
            W_work[:, j:i_end] -= e.unsqueeze(1) * U_row_block
        if i_end < in_features:
            W_work[:, i_end:] -= E.matmul(U[i:i_end, i_end:])

    return Q.to(device=orig_device, dtype=dtype)


# -----------------------------------------------------------------------------
# Unified trainer with configurable PTQ method
# -----------------------------------------------------------------------------


class BitDistillPreTrainer(pl.LightningModule):
    """
    Stage 1: Continual Pretraining with BitLinear quantization.
    Supports optional PTQ → QAT pipeline with configurable PTQ methods.
    
    Args:
        ptq_method: PTQ initialization method. Options:
            - None: QAT only, no PTQ initialization
            - "absmax": Per-row absmax quantization
            - "awq": Activation-aware weight quantization
            - "gptq": Second-order GPTQ quantization
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        target_quant_modules: Optional[List[str]] = None,
        target_subln_modules: Optional[List[str]] = None,
        quant_type: str = "bitnet",
        ptq_method: Optional[str] = None,
        calibration_samples: Optional[int] = None,
        n_bit_ptq: Optional[int] = None,
        log_grad_norm: bool = True,
        log_weight_stats: bool = True,
        log_gpu_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = str(model_name)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.target_quant_modules = target_quant_modules or []
        self.target_subln_modules = target_subln_modules or []
        self.quant_type = quant_type
        self.ptq_method = ptq_method
        self.log_grad_norm = log_grad_norm
        self.log_weight_stats = log_weight_stats
        self.log_gpu_memory = log_gpu_memory

        # Only convert to int if not None, otherwise keep as None.
        self.calibration_samples = (
            int(calibration_samples) if calibration_samples is not None else None
        )
        self.n_bit_ptq = int(n_bit_ptq) if n_bit_ptq is not None else None

        self.total_tokens_seen = 0
        self.training_start_time = None
        self.last_log_time = None
        self.tokens_since_last_log = 0
        self._logged_model_stats = False
        
        # Track whether we're loading from checkpoint
        self._loaded_from_checkpoint = False
        
        # Validate PTQ method
        if ptq_method is not None:
            if not self.target_quant_modules:
                raise ValueError("PTQ requires at least one layer pattern in target_quant_modules")
            valid_methods = ["absmax", "awq", "gptq"]
            if ptq_method not in valid_methods:
                raise ValueError(f"ptq_method must be one of {valid_methods} or None, got '{ptq_method}'")
        
        self.model = load_bitlab_model(model_name)
        self.model.train()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def get_target_linear_modules(self) -> List[Tuple[str, nn.Module, bool]]:
        """List of (module_name, module, needs_subln) for quantization."""
        if not self.target_quant_modules:
            return []
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
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    def collect_activations(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """Collect activations from calibration samples for activation-aware PTQ methods."""
        was_training = self.model.training
        self.model.eval()
        activations: Dict[str, torch.Tensor] = {}
        modules_to_collect = self.get_target_linear_modules()

        def make_hook(name: str):
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                if name not in activations:
                    activations[name] = input[0].detach().cpu()
            return hook

        hooks = [m.register_forward_hook(make_hook(n)) for n, m, _ in modules_to_collect]
        try:
            sample_count = self.calibration_samples
            use_tqdm = getattr(self, "trainer", None) is None or self.trainer.is_global_zero
            pbar = tqdm(total=sample_count, desc="[PTQ] Collecting activations") if use_tqdm else None
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
                        pbar.update(batch_size)
                    if sample_count <= 0:
                        break
            if pbar is not None:
                pbar.close()
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                self.model.train()
            else:
                self.model.eval()
        return activations

    def prepare_ptq(self) -> None:
        """Apply PTQ initialization based on configured method."""
        if self.ptq_method is None:
            return
        
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        
        modules = self.get_target_linear_modules()
        
        if self.ptq_method == "absmax":
            # Simple per-row absmax quantization
            iterator = tqdm(modules, desc="[PTQ] AbsMax quantization") if self.trainer.is_global_zero else modules
            for name, module, _ in iterator:
                with torch.no_grad():
                    w = ptq_prequantize_weight(module.weight.data, n_bits=self.n_bit_ptq)
                    module.weight.data = w
        
        elif self.ptq_method in ["awq", "gptq"]:
            # Activation-aware methods require calibration data
            activations = self.collect_activations(self.trainer.datamodule.train_dataloader())
            
            if self.ptq_method == "awq":
                iterator = tqdm(modules, desc="[PTQ] AWQ quantization") if self.trainer.is_global_zero else modules
                for name, module, _ in iterator:
                    if name not in activations:
                        continue
                    with torch.no_grad():
                        w = awq_prequantize_weight(
                            module.weight.data, activations[name], n_bits=self.n_bit_ptq
                        )
                        module.weight.data = w
            
            elif self.ptq_method == "gptq":
                iterator = tqdm(modules, desc="[PTQ] GPTQ quantization") if self.trainer.is_global_zero else modules
                for name, module, _ in iterator:
                    if name not in activations:
                        continue
                    with torch.no_grad():
                        w = gptq_prequantize_weight(
                            W=module.weight.data, X=activations[name], n_bits=self.n_bit_ptq
                        )
                        module.weight.data = w.to(
                            device=module.weight.data.device, dtype=module.weight.data.dtype
                        )

    def prepare_qat(self) -> None:
        """Replace target Linear layers with BitLinear for QAT."""
        modules_to_replace = self.get_target_linear_modules()
        if not modules_to_replace:
            return

        iterator = tqdm(modules_to_replace, desc="[QAT] Quantizing BitLinear layers")
        for name, module, needs_subln in iterator:
            bitlinear = BitLinear.from_linear(module, quant_type=self.quant_type)
            if needs_subln:
                new_module = nn.Sequential(RMSNormNoParam(bitlinear.in_features), bitlinear)
            else:
                new_module = bitlinear
            self._set_module_by_name(name, new_module)

    def _get_trainer_compute_dtype(self) -> torch.dtype:
        p = getattr(self.trainer, "precision", None)
        if p is None:
            p = getattr(getattr(self.trainer, "precision_plugin", None), "precision", None)
        if p is None:
            plugin = getattr(getattr(self.trainer, "strategy", None), "precision_plugin", None)
            p = getattr(plugin, "precision", None)
        if p in (None, 32, "32", "32-true", "32_true"):
            return torch.float32
        if p in (64, "64", "64-true", "64_true"):
            return torch.float64
        if p in ("bf16", "bf16-true", "bf16_true", "bf16-mixed", "bf16_mixed"):
            return torch.bfloat16
        if p in (16, "16", "16-true", "16_true", "16-mixed", "16_mixed"):
            return torch.float16
        return torch.float32

    def _log_quantization_stats(self) -> None:
        """Log statistics about quantized layers."""
        total_layers = 0
        quantized_layers = 0
        total_params = 0
        quantized_params = 0
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                total_layers += 1
                total_params += module.weight.numel()
            if isinstance(module, BitLinear):
                quantized_layers += 1
                quantized_params += module.weight.numel()
        
        if total_layers > 0:
            self.log("model/total_layers", float(total_layers), on_step=True, on_epoch=False)
            self.log("model/quantized_layers", float(quantized_layers), on_step=True, on_epoch=False)
            self.log("model/quantization_ratio", float(quantized_layers) / float(total_layers), 
                    on_step=True, on_epoch=False)
        
        if total_params > 0:
            self.log("model/total_params_M", float(total_params) / 1e6, on_step=True, on_epoch=False)
            self.log("model/quantized_params_M", float(quantized_params) / 1e6, on_step=True, on_epoch=False)
            self.log("model/param_quantization_ratio", float(quantized_params) / float(total_params),
                    on_step=True, on_epoch=False)

    def _log_weight_statistics(self) -> None:
        """Log weight distribution statistics for BitLinear layers."""
        if not self.log_weight_stats:
            return
        
        weight_means = []
        weight_stds = []
        weight_max_abs = []
        
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                w = module.weight.data.detach().float()
                weight_means.append(w.mean().item())
                weight_stds.append(w.std().item())
                weight_max_abs.append(w.abs().max().item())
        
        if weight_means:
            self.log("weights/mean_avg", sum(weight_means) / len(weight_means), 
                    on_step=False, on_epoch=True)
            self.log("weights/std_avg", sum(weight_stds) / len(weight_stds),
                    on_step=False, on_epoch=True)
            self.log("weights/max_abs_avg", sum(weight_max_abs) / len(weight_max_abs),
                    on_step=False, on_epoch=True)

    def _log_gradient_norms(self) -> None:
        """Log gradient norms for monitoring training stability."""
        if not self.log_grad_norm:
            return
        
        total_norm = 0.0
        num_params = 0
        max_grad = 0.0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1
                max_grad = max(max_grad, p.grad.data.abs().max().item())
        
        if num_params > 0:
            total_norm = total_norm ** 0.5
            self.log("gradients/global_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False)
            self.log("gradients/max_value", max_grad, on_step=True, on_epoch=False, prog_bar=False)

    def _log_gpu_memory_stats(self) -> None:
        """Log GPU memory usage."""
        if not self.log_gpu_memory or not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
        
        self.log("system/gpu_memory_allocated_GB", allocated, on_step=True, on_epoch=False)
        self.log("system/gpu_memory_reserved_GB", reserved, on_step=True, on_epoch=False)

    def _log_throughput_metrics(self, num_tokens: int) -> None:
        """Log throughput metrics (tokens/sec)."""
        current_time = time.time()
        
        if self.training_start_time is None:
            self.training_start_time = current_time
            self.last_log_time = current_time
        
        # Log instantaneous throughput (since last log)
        if self.last_log_time is not None:
            time_delta = current_time - self.last_log_time
            if time_delta > 0:
                self.tokens_since_last_log += num_tokens
                tokens_per_sec = self.tokens_since_last_log / time_delta
                self.log("performance/tokens_per_sec", tokens_per_sec, 
                        on_step=True, on_epoch=False, prog_bar=True)
                # Reset for next measurement
                self.last_log_time = current_time
                self.tokens_since_last_log = 0
        
        # Log average throughput (since training start)
        total_time = current_time - self.training_start_time
        if total_time > 0:
            avg_tokens_per_sec = self.total_tokens_seen / total_time
            self.log("performance/avg_tokens_per_sec", avg_tokens_per_sec,
                    on_step=False, on_epoch=True, prog_bar=False)

    def on_fit_start(self) -> None:
        """
        Template method: PTQ → QAT pipeline.
        Skips PTQ if loading from checkpoint (weights already trained).
        """
        # If loading from checkpoint, QAT structure is already prepared
        # in on_load_checkpoint, and we don't need PTQ
        if not self._loaded_from_checkpoint:
            self.prepare_ptq()
            self.prepare_qat()
        
        # Always ensure correct dtype and device
        dtype = self._get_trainer_compute_dtype()
        self.model.to(device=self.device, dtype=dtype)
        
        # Reset timing metrics
        self.training_start_time = None
        self.last_log_time = None
        self.tokens_since_last_log = 0
        
        # Reset flag for future training runs
        self._loaded_from_checkpoint = False
        
        # Flag to log model stats on first training step
        self._logged_model_stats = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        return self.ce_loss(shift_logits, shift_labels)

    def _count_tokens(self, labels: torch.Tensor) -> int:
        return (labels != -100).sum().item()

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, prefix: str
    ) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Log batch statistics
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        self.log(f"{prefix}/batch_size", float(batch_size), on_step=True, on_epoch=False)
        self.log(f"{prefix}/seq_length", float(seq_length), on_step=True, on_epoch=False)
        
        logits = self(input_ids, attention_mask)
        loss = self._compute_ce_loss(logits, labels)
        perplexity = torch.exp(loss)
        
        # Enhanced logging with better organization
        self.log(
            f"{prefix}/loss", loss, 
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            f"{prefix}/perplexity", perplexity,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        
        # Log token accuracy (optional, useful for monitoring)
        with torch.no_grad():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            predictions = shift_logits.argmax(dim=-1)
            
            # Only compute accuracy on non-padding tokens
            mask = shift_labels != -100
            if mask.sum() > 0:
                correct = (predictions == shift_labels) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                self.log(f"{prefix}/token_accuracy", accuracy, 
                        on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Log model stats and initial learning rate on first training step
        if not self._logged_model_stats:
            self._log_quantization_stats()
            self.log("optimization/initial_lr", self.learning_rate, on_step=True, on_epoch=False)
            self._logged_model_stats = True
        
        num_tokens = self._count_tokens(batch["labels"])
        self.total_tokens_seen += num_tokens
        
        loss = self._shared_step(batch, batch_idx, "train")
        
        # Log cumulative tokens
        self.log("train/tokens_seen", float(self.total_tokens_seen), 
                on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/tokens_seen_M", float(self.total_tokens_seen) / 1e6,
                on_step=True, on_epoch=False, prog_bar=False)
        
        # Log throughput metrics
        self._log_throughput_metrics(num_tokens)
        
        # Log GPU memory
        self._log_gpu_memory_stats()
        
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Called before optimizer.step(). Log gradient norms here."""
        self._log_gradient_norms()

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log weight statistics
        self._log_weight_statistics()
        
        # Log current learning rate
        try:
            if self.trainer and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log("optimization/learning_rate", current_lr, on_step=False, on_epoch=True)
        except (AttributeError, IndexError):
            pass  # Skip if optimizer not available yet

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        return optimizer

    def chat(
        self,
        prompt: str,
        generation_params: dict = {"max_new_tokens": 100},
        show_tokens: bool = True,
        use_chat_template: bool = False,
    ) -> Tuple[str, str]:
        """Generate text with self.model. Returns (generated_text_clean, generated_text_with_special)."""
        tokenizer = load_bitlab_tokenizer(self.model_name)
        has_chat_template = (
            use_chat_template and getattr(tokenizer, "chat_template", None) is not None
        )
        if has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            formatted_prompt = text
        else:
            if use_chat_template and not getattr(tokenizer, "chat_template", None):
                print("Note: use_chat_template is true but this tokenizer has no chat_template; using plain prompt.")
            inputs = tokenizer(prompt, return_tensors="pt")
            formatted_prompt = prompt

        if show_tokens:
            print("\n" + "=" * 80)
            print("INPUT ANALYSIS")
            print("=" * 80)
            print(f"\nOriginal prompt:\n{repr(prompt)}\n")
            if use_chat_template and formatted_prompt != prompt:
                print(f"Formatted with chat template:\n{repr(formatted_prompt)}\n")
            input_ids = inputs["input_ids"][0].tolist()
            print(f"Token IDs ({len(input_ids)} tokens):\n{input_ids}")
            decoded_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
            print(f"\nDecoded with special tokens:\n{repr(decoded_with_special)}")
            print(f"\nIndividual tokens:")
            for i, token_id in enumerate(input_ids):
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                token_name = tokenizer.convert_ids_to_tokens(token_id)
                print(f"  [{i:3d}] ID={token_id:6d} | Token={repr(token_name):20s} | Decoded={repr(token_str)}")
            print("=" * 80)

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_params)
        self.model.train()

        output_ids = outputs[0].tolist()
        input_ids_list = inputs["input_ids"][0].tolist()
        if show_tokens:
            print("\n" + "=" * 80)
            print("OUTPUT ANALYSIS")
            print("=" * 80)
            print(f"\nOutput Token IDs ({len(output_ids)} tokens total):\n{output_ids}")
            generated_ids = output_ids[len(input_ids_list):]
            print(f"\nGenerated Token IDs ({len(generated_ids)} new tokens):\n{generated_ids}")
            full_decoded = tokenizer.decode(output_ids, skip_special_tokens=False)
            print(f"\nFull decoded output (with special tokens):\n{repr(full_decoded)}")
            if generated_ids:
                print("\nIndividual generated tokens:")
                for i, token_id in enumerate(generated_ids):
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    token_name = tokenizer.convert_ids_to_tokens(token_id)
                    print(f"  [{i:3d}] ID={token_id:6d} | Token={repr(token_name):20s} | Decoded={repr(token_str)}")
            print("=" * 80)

        generated_text_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_text_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "=" * 80)
        print("FINAL OUTPUT")
        print("=" * 80)
        print(generated_text_with_special)
        print("\n" + "=" * 80)
        return generated_text_clean, generated_text_with_special


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called before loading state dict from checkpoint.
        Prepares model structure (BitLinear/RMSNorm) to match checkpoint.
        """
        # Mark that we're loading from checkpoint
        self._loaded_from_checkpoint = True
        
        # Prepare QAT structure before loading weights
        # This ensures the model has the right layers (BitLinear, RMSNorm)
        # to receive the checkpoint's state dict
        self.prepare_qat()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called before saving state dict to checkpoint.
        Prepares model structure (BitLinear/RMSNorm) to match checkpoint.
        """
        
        has_qat = any(isinstance(m, BitLinear) for m in self.model.modules())
        if not has_qat: 
            self.prepare_qat()
        super().on_save_checkpoint(checkpoint)