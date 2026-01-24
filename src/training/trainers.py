import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from typing import List

import copy 
from bitcore import BitLinear 
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict

from src.models.models import MODELS_REGISTRY


class SFTTrainer(pl.LightningModule):
    def __init__(self, model_name, learning_rate=5e-5, weight_decay=0.00):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = str(model_name)
        self.model = MODELS_REGISTRY[model_name]()  # Call the loader function
        
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




class RMSNormNoParam(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # x shape: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def __repr__(self): 
        return f"RMSNormNoParam(dim={self.dim}, eps={self.eps})"


class BitDistillTrainer(pl.LightningModule): 
    def __init__(
        self, 
        model_name: str, 
        learning_rate: float = 5e-5, 
        weight_decay: float = 0.01, 
        quant_patterns: List[str] = None, 
        attention_distill_patterns: Dict[str, str] = {"query": "q_proj", "key": "k_proj", "value": "v_proj"},
        quant_type: str = "bitnet",
        lambda_kl: float = 1.0,  # Weight for KL loss (λ in paper Eq. 13)
        gamma_attention: float = 1e3,  # Weight for attention distillation loss (γ in paper Eq. 13)
        temperature: float = 1.0,  # Temperature for distillation (1.0 = no temperature)
        split_heads: int = 1,  # Number of heads to split into for attention distillation
        stage_switch_tokens: int = None,  # Token count at which to switch from stage 1 (CE only) to stage 2 (distillation)
        distill_layer: int = None  # Specific layer index to distill (None = all layers, paper recommends single layer)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ===== CHANGE 5: Hyperparameter Validation =====
        assert quant_patterns is not None and len(quant_patterns) > 0, \
            "Must specify at least one layer pattern to quantize in quant_patterns"
        assert float(lambda_kl) >= 0, "lambda_kl must be non-negative"
        assert float(gamma_attention) >= 0, "gamma_attention must be non-negative"
        if float(gamma_attention) > 0:
            assert attention_distill_patterns is not None, \
                "attention_distill_patterns must be provided when gamma_attention > 0"
        if int(distill_layer) is not None:
            assert isinstance(distill_layer, int), "distill_layer must be an integer layer index"
        # ===============================================
        
        # ===== CHANGE 4: Renamed alpha/beta to match paper Eq. 13 =====
        # Store hyperparameters
        self.model_name = str(model_name)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.quant_patterns = quant_patterns
        self.quant_type = quant_type
        self.lambda_kl = float(lambda_kl)  # Renamed from alpha to match paper (λ)
        self.gamma_attention = float(gamma_attention)  # Renamed from beta to match paper (γ)
        self.temperature = float(temperature)
        self.split_heads = split_heads
        self.attention_distill_patterns = attention_distill_patterns
        self.stage_switch_tokens = stage_switch_tokens  # None means always use distillation
        # ===== CHANGE 3: Add distill_layer parameter =====
        self.distill_layer = distill_layer  # None means distill all layers, int means specific layer
        # =================================================
        # ==================================================================
        
        # Token tracking
        self.total_tokens_seen = 0
        
        # Training stage tracking
        self._current_stage = 1 if stage_switch_tokens is not None else 2
        
        # Initialize models
        self.teacher = MODELS_REGISTRY[model_name]()
        self.student = copy.deepcopy(self.teacher)
        
        # Prepare models
        self.prepare_student()
        self.prepare_teacher()
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    @property
    def current_stage(self):
        """
        Returns the current training stage:
        - Stage 1: Continual pretraining with CE loss only
        - Stage 2: Distillation with CE + KL + attention losses
        """
        if self.stage_switch_tokens is None:
            return 2  # Always use distillation if no switch token specified
        
        if self.total_tokens_seen < self.stage_switch_tokens:
            return 1  # Stage 1: CE only
        else:
            return 2  # Stage 2: Full distillation

    def prepare_student(self): 
        """Convert specified Linear layers in student to BitLinear"""
        # Collect modules to replace first to avoid modifying during iteration
        modules_to_replace = []
        for name, module in self.student.named_modules(): 
            if isinstance(module, torch.nn.Linear):
                if any(pattern in name for pattern in self.quant_patterns):
                    modules_to_replace.append((name, module))
        
        print(f"Replacing {len(modules_to_replace)} Linear layers with BitLinear")
        
        # Replace the modules
        for name, module in modules_to_replace:
            bitlinear = BitLinear.from_linear(module, quant_type=self.quant_type)
            module = nn.Sequential(RMSNormNoParam(bitlinear.in_features), bitlinear)
            # Navigate to parent and replace the module
            parts = name.split('.')
            parent = self.student
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module)

    def prepare_teacher(self): 
        """Freeze teacher and set to eval mode"""
        for param in self.teacher.parameters(): 
            param.requires_grad = False
        self.teacher.eval()
    
    def on_train_start(self):
        """Ensure teacher stays in eval mode at the start of training"""
        super().on_train_start()
        self.teacher.eval()
    
    def on_train_batch_start(self, batch, batch_idx):
        """Ensure teacher stays in eval mode during training"""
        super().on_train_batch_start(batch, batch_idx)
        self.teacher.eval()
        
        # Log stage transitions
        new_stage = self.current_stage
        if new_stage != self._current_stage:
            print(f"\n{'='*50}")
            print(f"STAGE TRANSITION: Stage {self._current_stage} -> Stage {new_stage}")
            print(f"Total tokens seen: {self.total_tokens_seen:,}")
            print(f"{'='*50}\n")
            self._current_stage = new_stage

    def forward(self, input_ids, attention_mask, collect_attention_states=False):
        """Forward pass through both teacher and student"""
        # Initialize attention states storage
        attention_states = defaultdict(list)
        hooks = []
        
        if collect_attention_states:
            def get_activation_hook(key):
                def hook(module, input, output):
                    attention_states[key].append(output.detach())
                return hook
            
            # Register hooks for teacher
            for name, module in self.teacher.named_modules():
                if isinstance(module, torch.nn.Linear):
                    query_pattern = self.attention_distill_patterns["query"]
                    key_pattern = self.attention_distill_patterns["key"]
                    value_pattern = self.attention_distill_patterns["value"]
                    
                    if query_pattern in name:
                        hooks.append(module.register_forward_hook(get_activation_hook("teacher_query_act")))
                    elif key_pattern in name:
                        hooks.append(module.register_forward_hook(get_activation_hook("teacher_key_act")))
                    elif value_pattern in name:
                        hooks.append(module.register_forward_hook(get_activation_hook("teacher_value_act")))
            
            # Register hooks for student
            for name, module in self.student.named_modules():
                # Check if it's a BitLinear wrapped in Sequential
                if isinstance(module, torch.nn.Sequential) and len(module) == 2:
                    linear_module = module[1]  # BitLinear is second in Sequential
                elif isinstance(module, torch.nn.Linear):
                    linear_module = module
                else:
                    continue
                
                query_pattern = self.attention_distill_patterns["query"]
                key_pattern = self.attention_distill_patterns["key"]
                value_pattern = self.attention_distill_patterns["value"]
                
                if query_pattern in name:
                    hooks.append(linear_module.register_forward_hook(get_activation_hook("student_query_act")))
                elif key_pattern in name:
                    hooks.append(linear_module.register_forward_hook(get_activation_hook("student_key_act")))
                elif value_pattern in name:
                    hooks.append(linear_module.register_forward_hook(get_activation_hook("student_value_act")))
        
        # Forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Prepare output dictionary
        output_dict = {
            "student_logits": student_outputs.logits,
            "teacher_logits": teacher_outputs.logits
        }
        
        if collect_attention_states:
            output_dict.update(dict(attention_states))
        
        return output_dict

    def _ce_loss(self, student_logits, labels):
        """Compute cross-entropy loss for next-token prediction"""
        # Shift logits and labels for next-token prediction
        student_shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        batch_size, seq_len, vocab_size = student_shift_logits.shape
        student_shift_logits = student_shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Cross-Entropy Loss (standard language modeling)
        ce_loss = self.ce_loss(student_shift_logits, shift_labels)
        
        return ce_loss, shift_labels

    def _kl_loss(self, student_logits, teacher_logits, shift_labels):
        """Compute KL divergence loss for distillation"""
        # Shift logits for next-token prediction
        student_shift_logits = student_logits[..., :-1, :].contiguous()
        teacher_shift_logits = teacher_logits[..., :-1, :].contiguous()
        
        # Flatten the tokens
        batch_size, seq_len, vocab_size = student_shift_logits.shape
        student_shift_logits = student_shift_logits.view(-1, vocab_size)
        teacher_shift_logits = teacher_shift_logits.view(-1, vocab_size)
        
        # Only compute on non-masked positions
        mask = shift_labels != -100
        
        if mask.sum() > 0:  # Avoid division by zero
            # Apply temperature for distillation
            student_log_probs = F.log_softmax(
                student_shift_logits[mask] / self.temperature, 
                dim=-1
            )
            teacher_probs = F.softmax(
                teacher_shift_logits[mask] / self.temperature, 
                dim=-1
            )
            
            # Compute KL divergence: KL(teacher || student)
            kl_loss = F.kl_div(
                student_log_probs, 
                teacher_probs, 
                reduction="batchmean",
                log_target=False
            )
            
            # Scale by temperature^2 (standard in distillation literature)
            kl_loss = kl_loss * (self.temperature ** 2)
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)
        
        return kl_loss

    def _attention_distill_loss(self, student_states, teacher_states, split_heads=1):
        """
        Compute attention distillation loss using relation matrices.
        
        ===== CHANGE 3: Single-layer distillation support =====
        Paper recommends distilling only a single layer for better optimization flexibility.
        When self.distill_layer is set, only that layer is used for distillation.
        =======================================================
        
        Args:
            student_states: List of tensors [query, key, value] each with shape [num_layers, B, num_heads, L, head_dim]
            teacher_states: List of tensors [query, key, value] each with shape [num_layers, B, num_heads, L, head_dim]
            split_heads: Number of groups to split heads into (for computational efficiency)
        
        Returns:
            distill_loss: Scalar tensor representing the attention distillation loss
        """
        distill_loss = 0.0
        num_comparisons = 0
        
        # Stack the states: query, key, value
        # Each should be a list of activations from different layers
        for qkv_idx in range(3):  # 0=query, 1=key, 2=value
            s_values = student_states[qkv_idx]  # List of layer activations
            t_values = teacher_states[qkv_idx]  # List of layer activations
            
            # Determine which layers to distill
            num_layers = min(len(s_values), len(t_values))
            
            # ===== CHANGE 3: Single-layer distillation =====
            if self.distill_layer is not None:
                # Only distill the specified layer
                # Handle negative indexing (e.g., -1 for last layer)
                layer_idx = self.distill_layer if self.distill_layer >= 0 else num_layers + self.distill_layer
                
                # Validate layer index
                if layer_idx < 0 or layer_idx >= num_layers:
                    print(f"Warning: distill_layer {self.distill_layer} is out of range [0, {num_layers-1}]. Skipping.")
                    continue
                
                layers_to_process = [layer_idx]
            else:
                # Distill all layers
                layers_to_process = range(num_layers)
            # ===============================================
            
            # Process selected layer(s)
            for layer_idx in layers_to_process:
                s_layer = s_values[layer_idx]  # [B, num_heads, L, head_dim] or [B, L, hidden_dim]
                t_layer = t_values[layer_idx]  # [B, num_heads, L, head_dim] or [B, L, hidden_dim]
                
                # Handle different input shapes
                if s_layer.dim() == 3:  # [B, L, hidden_dim]
                    B, L, hidden_dim = s_layer.shape
                    # Infer number of heads and head dimension
                    # Assume hidden_dim = num_heads * head_dim
                    # We'll reshape to [B, L, num_heads, head_dim] then transpose
                    num_heads = hidden_dim // (hidden_dim // split_heads) if split_heads > 1 else 1
                    head_dim = hidden_dim // num_heads
                    
                    s_layer = s_layer.reshape(B, L, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
                    t_layer = t_layer.reshape(B, L, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
                
                B, num_heads, L, head_dim = s_layer.shape
                
                # Compute D: dimension after splitting heads
                D = num_heads * head_dim // split_heads
                
                # Normalize and reshape for split heads
                # [B, num_heads, L, head_dim] -> [B, L, split_heads, D]
                s_reshaped = s_layer.transpose(1, 2).reshape(B, L, split_heads, D)
                t_reshaped = t_layer.transpose(1, 2).reshape(B, L, split_heads, D)
                
                # Transpose to [B, split_heads, L, D]
                s_reshaped = s_reshaped.transpose(1, 2)
                t_reshaped = t_reshaped.transpose(1, 2)
                
                # Normalize
                s_norm = F.normalize(s_reshaped, dim=-1)
                t_norm = F.normalize(t_reshaped, dim=-1)
                
                # Compute relation matrices: [B, split_heads, L, L]
                s_relation = torch.matmul(s_norm, s_norm.transpose(-2, -1))
                t_relation = torch.matmul(t_norm, t_norm.transpose(-2, -1))
                
                # Apply temperature and reshape: [B, split_heads, L, L] -> [B*split_heads*L, L]
                s_relation = (s_relation / self.temperature).reshape(-1, L)
                t_relation = (t_relation / self.temperature).reshape(-1, L)
                
                # Compute probabilities
                s_prob = F.softmax(s_relation, dim=-1).clamp(min=1e-8)
                t_prob = F.softmax(t_relation, dim=-1).clamp(min=1e-8)
                
                # Compute KL divergence
                layer_loss = F.kl_div(
                    torch.log(s_prob), 
                    t_prob, 
                    reduction="batchmean", 
                    log_target=False
                )
                
                distill_loss += layer_loss
                num_comparisons += 1
        
        # Average over all comparisons
        if num_comparisons > 0:
            distill_loss = distill_loss / num_comparisons
        
        return distill_loss

    def compute_loss(self, outputs, labels, stage):
        """
        Compute loss based on training stage.
        
        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels
            stage: Current training stage (1 or 2)
        
        Returns:
            dict: Dictionary containing loss components
        """
        student_logits = outputs["student_logits"]
        teacher_logits = outputs["teacher_logits"]
        
        # Compute CE loss (used in both stages)
        ce_loss, shift_labels = self._ce_loss(student_logits, labels)
        
        loss_dict = {"ce_loss": ce_loss}
        
        # Stage 1: CE loss only (continual pretraining)
        if stage == 1:
            return loss_dict
        
        # Stage 2: Full distillation (CE + KL + attention)
        # Compute KL loss
        kl_loss = self._kl_loss(student_logits, teacher_logits, shift_labels)
        loss_dict["kl_loss"] = kl_loss
        
        # Compute attention distillation loss if states are available
        if "student_query_act" in outputs and "teacher_query_act" in outputs:
            student_states = [
                outputs["student_query_act"],
                outputs["student_key_act"],
                outputs["student_value_act"]
            ]
            teacher_states = [
                outputs["teacher_query_act"],
                outputs["teacher_key_act"],
                outputs["teacher_value_act"]
            ]
            
            attention_loss = self._attention_distill_loss(
                student_states, 
                teacher_states, 
                split_heads=self.split_heads
            )
            loss_dict["attention_loss"] = attention_loss
        
        return loss_dict

    def _count_tokens(self, labels):
        """Count the number of valid (non-padding) tokens in the batch"""
        # Count tokens that are not -100 (the ignore_index)
        return (labels != -100).sum().item()

    def training_step(self, batch, batch_idx): 
        """Single training step"""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Count tokens in this batch
        num_tokens = self._count_tokens(labels)
        self.total_tokens_seen += num_tokens
        
        # Determine current stage
        stage = self.current_stage
        
        # Forward with attention state collection only in stage 2 if gamma_attention > 0
        collect_attention = (stage == 2) and (self.gamma_attention > 0)
        outputs = self(input_ids, attention_mask, collect_attention_states=collect_attention)
        
        # Compute losses based on stage
        loss_dict = self.compute_loss(outputs, labels, stage)
        
        # ===== CHANGE 4: Loss formulation matching paper Eq. 13 =====
        # Paper: L = LCE + λ*LLD + γ*LAD
        # Compute combined loss based on stage
        if stage == 1:
            # Stage 1: CE loss only
            loss = loss_dict["ce_loss"]
        else:
            # Stage 2: LCE + λ*LKL + γ*LAD (matching paper Eq. 13)
            loss = loss_dict["ce_loss"] + self.lambda_kl * loss_dict["kl_loss"]
            
            if "attention_loss" in loss_dict:
                loss += self.gamma_attention * loss_dict["attention_loss"]
        # ================================================================
        
        # Log current stage
        self.log("train/stage", float(stage), on_step=True, on_epoch=False, prog_bar=True)
        
        # Standard logging (step-based)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", loss_dict["ce_loss"], on_step=True, on_epoch=True)
        
        # Token-based logging
        self.log("train_tokens/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_tokens/ce_loss", loss_dict["ce_loss"], on_step=True, on_epoch=False, prog_bar=False)
        
        # Stage 2 specific logging
        if stage == 2:
            if "kl_loss" in loss_dict:
                self.log("train/kl_loss", loss_dict["kl_loss"], on_step=True, on_epoch=True)
                self.log("train_tokens/kl_loss", loss_dict["kl_loss"], on_step=True, on_epoch=False, prog_bar=False)
            
            if "attention_loss" in loss_dict:
                self.log("train/attention_loss", loss_dict["attention_loss"], on_step=True, on_epoch=True)
                self.log("train_tokens/attention_loss", loss_dict["attention_loss"], 
                        on_step=True, on_epoch=False, prog_bar=False)
        
        # Log total tokens seen (this will be the x-axis value)
        self.log("tokens_seen", float(self.total_tokens_seen), on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step (optional but recommended)"""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Determine current stage
        stage = self.current_stage
        
        # Forward with attention state collection only in stage 2 if gamma_attention > 0
        collect_attention = (stage == 2) and (self.gamma_attention > 0)
        outputs = self(input_ids, attention_mask, collect_attention_states=collect_attention)
        
        # Compute losses based on stage
        loss_dict = self.compute_loss(outputs, labels, stage)
        
        # ===== CHANGE 4: Loss formulation matching paper Eq. 13 =====
        # Compute combined loss based on stage
        if stage == 1:
            # Stage 1: CE loss only
            loss = loss_dict["ce_loss"]
        else:
            # Stage 2: LCE + λ*LKL + γ*LAD (matching paper Eq. 13)
            loss = loss_dict["ce_loss"] + self.lambda_kl * loss_dict["kl_loss"]
            
            if "attention_loss" in loss_dict:
                loss += self.gamma_attention * loss_dict["attention_loss"]
        # ================================================================
        
        # Log current stage
        self.log("val/stage", float(stage), on_step=False, on_epoch=True)
        
        # Standard logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ce_loss", loss_dict["ce_loss"], on_step=False, on_epoch=True)
        
        # Token-based logging
        self.log("val_tokens/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_tokens/ce_loss", loss_dict["ce_loss"], on_step=False, on_epoch=True, prog_bar=False)
        
        # Stage 2 specific logging
        if stage == 2:
            if "kl_loss" in loss_dict:
                self.log("val/kl_loss", loss_dict["kl_loss"], on_step=False, on_epoch=True)
                self.log("val_tokens/kl_loss", loss_dict["kl_loss"], on_step=False, on_epoch=True, prog_bar=False)
            
            if "attention_loss" in loss_dict:
                self.log("val/attention_loss", loss_dict["attention_loss"], on_step=False, on_epoch=True)
                self.log("val_tokens/attention_loss", loss_dict["attention_loss"], 
                        on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    def configure_optimizers(self): 
        """Configure optimizer"""
        # Only optimize student parameters
        optimizer = torch.optim.AdamW(
            self.student.parameters(),  # Only student, not teacher!
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Optional: Add learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=self.trainer.max_epochs
        # )
        # return [optimizer], [scheduler]
        
        return optimizer

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to return only student model weights.
        The student model is a HuggingFace model, so its state_dict already has keys
        with "model." prefix (e.g., "model.embed_tokens.weight").
        
        We return the student's state_dict directly (without adding another prefix),
        so the checkpoint has keys in "model.xxx" format, which matches what
        HuggingFace models expect. This allows direct loading without key manipulation.
        
        When Lightning saves a checkpoint, this ensures only student weights
        are saved in the correct format for direct model loading.
        """
        # Get student model's state dict directly
        # HuggingFace models already have "model." prefix in their state_dict keys
        # Note: destination parameter is for internal PyTorch use, we ignore it
        student_state = self.student.state_dict(prefix='', keep_vars=keep_vars)
        return student_state
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to load weights into student model.
        The state_dict from our custom state_dict() method has keys with "model." prefix
        (e.g., "model.embed_tokens.weight"), which matches the student model's format.
        
        This allows loading checkpoints saved by our custom state_dict() method
        directly into the student model without key manipulation.
        """
        # The state_dict already has "model." prefix matching the student model's format
        # Load directly into student model
        return self.student.load_state_dict(state_dict, strict=strict)


TRAINERS_REGISTRY = {
    "sfttrainer": SFTTrainer,
    "bitdistilltrainer": BitDistillTrainer,
}



if __name__ == "__main__": 
    trainer = BitDistillTrainer(
        model_name="qwen2_5_05_pt",
        learning_rate=5e-5,
        weight_decay=0.01,
        quant_patterns=["mlp"],
        quant_type="bitnet"
    )

    print(trainer)

