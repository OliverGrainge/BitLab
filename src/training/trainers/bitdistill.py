"""
BitDistill trainer: continual pretraining with BitLinear quantization.

Single trainer with configurable PTQ initialization methods:
- ptq_method=None: QAT only (no PTQ initialization)
- ptq_method="absmax": Per-row absmax PTQ then QAT
- ptq_method="awq": Activation-aware weight quantization PTQ then QAT
- ptq_method="gptq": Second-order GPTQ PTQ then QAT

Loss types:
- loss_type="ce": Standard cross-entropy (next-token prediction)
- loss_type="kl": KL-divergence distillation from a frozen teacher (deep copy
  of the original pretrained model, captured before any PTQ/QAT is applied).
"""

from typing import Any, Dict, List, Optional, Tuple
import copy
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    When ``loss.loss_type="kl"`` a frozen deep copy of the original pretrained
    model is captured *before* any PTQ or QAT modifications and used as the
    teacher for KL-divergence distillation.  The teacher is never checkpointed —
    on resume it is re-loaded from the base pretrained weights.

    Args:
        initialization: Optional dict for PTQ initialization. Keys:
            - ptq_method: None (QAT only), "absmax", "awq", or "gptq"
            - calibration_samples: int (required for awq/gptq)
            - n_bit_ptq: int (required when ptq_method is set)
        loss: Optional dict for loss config. Keys:
            - loss_type: "ce" (cross-entropy) or "kl" (distillation)
            - distill_temperature: float (default 2.0)
            - distill_beta: float (default 0.5) for JSD mixture weight
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        target_quant_modules: Optional[List[str]] = None,
        target_subln_modules: Optional[List[str]] = None,
        quant_type: str = "bitnet",
        initialization: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        initialization = initialization or {}
        loss = loss or {}

        self.model_name = str(model_name)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.target_quant_modules = target_quant_modules or []
        self.target_subln_modules = target_subln_modules or []
        self.quant_type = str(quant_type)

        # ---- Initialization config (PTQ) ----
        self.ptq_method = initialization.get("ptq_method", None)
        cal = initialization.get("calibration_samples", None)
        self.calibration_samples = int(cal) if cal is not None else None
        nbit = initialization.get("n_bit_ptq", None)
        self.n_bit_ptq = int(nbit) if nbit is not None else None

        # ---- Loss config ----
        self.loss_type = loss.get("loss_type", "ce")
        self.distill_temperature = float(loss.get("distill_temperature", 2.0))
        self.distill_beta = float(loss.get("distill_beta", 0.5))

        self.total_tokens_seen = 0
        self.training_start_time = None
        self.last_log_time = None
        self.tokens_since_last_log = 0
        self._loaded_from_checkpoint = False

        self._validate_configs()

        # Student model
        self.model = load_bitlab_model(self.model_name)
        self.model.train()

        # Teacher is created lazily in on_fit_start; kept as None until then
        self.teacher: Optional[nn.Module] = None

    def _validate_configs(self) -> None:
        """Validate PTQ and loss configuration."""
        if self.ptq_method is not None:
            if not self.target_quant_modules:
                raise ValueError("PTQ requires at least one layer pattern in target_quant_modules")
            valid_methods = ["absmax", "awq", "gptq"]
            if self.ptq_method not in valid_methods:
                raise ValueError(
                    f"ptq_method must be one of {valid_methods} or None, got '{self.ptq_method}'"
                )
            if self.n_bit_ptq is None:
                raise ValueError("n_bit_ptq is required when ptq_method is set")
            if self.ptq_method in ("awq", "gptq") and self.calibration_samples is None:
                raise ValueError(
                    "calibration_samples is required when ptq_method is 'awq' or 'gptq'"
                )
        if self.loss_type not in ("ce", "kl"):
            raise ValueError(f"loss_type must be 'ce' or 'kl', got '{self.loss_type}'")

    # -------------------------------------------------------------------------
    # Teacher management
    # -------------------------------------------------------------------------

    def _init_teacher(self) -> None:
        """
        Deep-copy the current (unmodified) student model into a frozen teacher.
        Must be called *before* prepare_ptq / prepare_qat.
        """
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _reload_teacher(self) -> None:
        """
        Load a fresh copy of the pretrained model as teacher.
        Used when resuming from checkpoint (student is already quantized, so a
        deep-copy would capture the wrong weights).
        """
        self.teacher = load_bitlab_model(self.model_name)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    # -------------------------------------------------------------------------
    # Module helpers
    # -------------------------------------------------------------------------

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

    def get_subln_only_linear_modules(self) -> List[Tuple[str, nn.Module]]:
        """Linear layers that need SubLN but are NOT being converted to BitLinear.

        These are layers whose name matches a pattern in ``target_subln_modules``
        but does *not* match any pattern in ``target_quant_modules``.  They stay
        as plain ``nn.Linear`` but get wrapped in
        ``Sequential(RMSNormNoParam, Linear)``.
        """
        if not self.target_subln_modules:
            return []
        results = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(pattern in name for pattern in self.target_subln_modules):
                continue
            # Skip layers already handled by the BitLinear path in prepare_qat
            if any(pattern in name for pattern in self.target_quant_modules):
                continue
            results.append((name, module))
        return results

    def _set_module_by_name(self, name: str, module: nn.Module) -> None:
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # PTQ & QAT
    # -------------------------------------------------------------------------

    def prepare_ptq(self) -> None:
        """Apply PTQ initialization based on configured method."""
        if self.ptq_method is None:
            return

        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        modules = self.get_target_linear_modules()

        if self.ptq_method == "absmax":
            iterator = tqdm(modules, desc="[PTQ] AbsMax quantization") if self.trainer.is_global_zero else modules
            for name, module, _ in iterator:
                with torch.no_grad():
                    w = ptq_prequantize_weight(module.weight.data, n_bits=self.n_bit_ptq)
                    module.weight.data = w

        elif self.ptq_method in ["awq", "gptq"]:
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
        """Replace target Linear layers with BitLinear for QAT, and wrap any
        remaining Linear layers that match ``target_subln_modules`` with SubLN."""
        # --- Pass 1: BitLinear replacement (+ SubLN where both flags match) ---
        modules_to_replace = self.get_target_linear_modules()
        if modules_to_replace:
            iterator = tqdm(modules_to_replace, desc="[QAT] Quantizing BitLinear layers")
            for name, module, needs_subln in iterator:
                bitlinear = BitLinear.from_linear(module, quant_type=self.quant_type)
                if needs_subln:
                    new_module = nn.Sequential(RMSNormNoParam(bitlinear.in_features), bitlinear)
                else:
                    new_module = bitlinear
                self._set_module_by_name(name, new_module)

        # --- Pass 2: SubLN-only wrapping for plain Linear layers ---
        subln_only = self.get_subln_only_linear_modules()
        if subln_only:
            iterator = tqdm(subln_only, desc="[QAT] Adding SubLN to Linear layers")
            for name, module in iterator:
                new_module = nn.Sequential(RMSNormNoParam(module.in_features), module)
                self._set_module_by_name(name, new_module)

    # -------------------------------------------------------------------------
    # Lifecycle hooks
    # -------------------------------------------------------------------------

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

    def on_fit_start(self) -> None:
        """
        PTQ → QAT pipeline.  When using KL distillation the teacher is
        snapshotted here so it always reflects the original pretrained weights.
        """
        if not self._loaded_from_checkpoint:
            # 1) Snapshot teacher BEFORE any PTQ/QAT touches the student
            if self.loss_type == "kl":
                self._init_teacher()

            # 2) Optionally quantize weights in-place, then wrap with BitLinear
            self.prepare_ptq()
            self.prepare_qat()
        else:
            # Resuming from checkpoint: student structure was already rebuilt in
            # on_load_checkpoint and weights were loaded.  Re-create teacher from
            # the original pretrained checkpoint (not the quantized student).
            if self.loss_type == "kl":
                self._reload_teacher()

        # Move everything to the correct device & dtype
        dtype = self._get_trainer_compute_dtype()
        self.model.to(device=self.device, dtype=dtype)
        if self.teacher is not None:
            self.teacher.to(device=self.device, dtype=dtype)

        # Reset timing metrics
        self.training_start_time = None
        self.last_log_time = None
        self.tokens_since_last_log = 0

        self._loaded_from_checkpoint = False

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called before loading state dict from checkpoint.
        Rebuilds the QAT layer structure so it can receive the checkpoint's
        state dict.  Teacher keys are never in the checkpoint (see
        on_save_checkpoint), so nothing extra is needed here.
        """
        self._loaded_from_checkpoint = True
        self.prepare_qat()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Strip teacher weights from the checkpoint — the teacher is always
        reconstructed from the base pretrained model on resume.
        """
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher.")
        }

        has_qat = any(isinstance(m, BitLinear) for m in self.model.modules())
        if not has_qat:
            self.prepare_qat()
        super().on_save_checkpoint(checkpoint)

    # -------------------------------------------------------------------------
    # Forward & loss
    # -------------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Next-token cross-entropy loss, ignoring padding."""
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    @torch.no_grad()
    def _get_teacher_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the frozen teacher and return its logits."""
        return self.teacher(input_ids=input_ids, attention_mask=attention_mask).logits

    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generalized Jensen–Shannon divergence (paper's distillation objective).

        DJSD^(β)(Pt || Ps) = β KL(Pt || M) + (1-β) KL(Ps || M),
        where M = β Pt + (1-β) Ps.

        Only positions where labels != -100 contribute (next-token shift).
        Uses distill_temperature for softening; scales by T^2 (optional but
        consistent with common distillation practice).
        """
        T = self.distill_temperature
        beta = self.distill_beta
        beta = max(0.0, min(1.0, beta))  # clamp to [0, 1] defensively

        # Align to next-token positions
        shift_student = student_logits[..., :-1, :].contiguous()  # [B, L-1, V]
        shift_teacher = teacher_logits[..., :-1, :].contiguous()  # [B, L-1, V]
        shift_labels  = labels[..., 1:].contiguous()              # [B, L-1]

        # Mask valid positions
        mask = (shift_labels != -100)  # [B, L-1]
        n_valid = mask.sum().clamp(min=1)

        # Temperature-scaled distributions
        log_pt = F.log_softmax(shift_teacher / T, dim=-1)  # log P_t
        log_ps = F.log_softmax(shift_student / T, dim=-1)  # log P_s
        pt = log_pt.exp()
        ps = log_ps.exp()

        # Mixture distribution M = β Pt + (1-β) Ps
        # add eps for numerical stability before log
        eps = 1e-8
        m = beta * pt + (1.0 - beta) * ps
        log_m = torch.log(m.clamp_min(eps))

        # KL(P || M) = sum_v P(v) * (log P(v) - log M(v))
        kl_t = (pt * (log_pt - log_m)).sum(dim=-1)  # [B, L-1]
        kl_s = (ps * (log_ps - log_m)).sum(dim=-1)  # [B, L-1]

        jsd_per_token = beta * kl_t + (1.0 - beta) * kl_s  # [B, L-1]

        loss = (jsd_per_token * mask.float()).sum() / n_valid
        return loss * (T * T)

    # -------------------------------------------------------------------------
    # Training / validation step
    # -------------------------------------------------------------------------

    def _count_tokens(self, labels: torch.Tensor) -> int:
        return (labels != -100).sum().item()

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, prefix: str
    ) -> torch.Tensor:
        input_ids     = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels        = batch["labels"]

        logits = self(input_ids, attention_mask)

        ce_loss = self._compute_ce_loss(logits, labels)

        if self.loss_type == "ce":
            loss = ce_loss
            self.log(f"{prefix}_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:  # "kl"
            teacher_logits = self._get_teacher_logits(input_ids, attention_mask)
            loss = self._compute_kl_loss(logits, teacher_logits, labels)
            self.log(f"{prefix}_kl_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Perplexity is always based on CE so it's directly comparable across runs
        self.log(f"{prefix}_perplexity", torch.exp(ce_loss), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Token-level accuracy
        with torch.no_grad():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != -100
            if mask.sum() > 0:
                correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                self.log(f"{prefix}_accuracy", accuracy,
                         on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        num_tokens = self._count_tokens(batch["labels"])
        self.total_tokens_seen += num_tokens

        loss = self._shared_step(batch, batch_idx, "train")

        # Tokens processed
        self.log("train_tokens_M", float(self.total_tokens_seen) / 1e6,
                 on_step=True, on_epoch=False, prog_bar=True)

        # Throughput
        current_time = time.time()
        if self.training_start_time is None:
            self.training_start_time = current_time
            self.last_log_time       = current_time

        if self.last_log_time is not None:
            time_delta = current_time - self.last_log_time
            if time_delta > 0:
                self.tokens_since_last_log += num_tokens
                self.log("train_tokens_per_sec",
                         self.tokens_since_last_log / time_delta,
                         on_step=True, on_epoch=False, prog_bar=True)
                self.last_log_time        = current_time
                self.tokens_since_last_log = 0

        return loss

    def on_train_epoch_end(self) -> None:
        """Log learning rate at epoch end."""
        try:
            if self.trainer and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log("train_learning_rate", current_lr, on_step=False, on_epoch=True)
        except (AttributeError, IndexError):
            pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

    # -------------------------------------------------------------------------
    # Inference helper
    # -------------------------------------------------------------------------

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