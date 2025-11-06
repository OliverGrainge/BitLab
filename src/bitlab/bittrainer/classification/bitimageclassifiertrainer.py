"""
PyTorch Lightning Trainer for Image Classification

This trainer implements supervised learning for image classification with support
for various loss types, learning rate schedules, and optimization strategies.
"""

import math
from typing import Optional, Literal, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix


class BitImageClassifierTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for image classification.
    
    Implements supervised learning with support for:
    - Multiple loss functions (cross-entropy, focal loss, label smoothing)
    - Various learning rate schedules (constant, linear, cosine)
    - Comprehensive metrics tracking (accuracy, F1, precision, recall)
    - Automatic mixed precision support
    - Multi-class and binary classification
    
    Args:
        model: The classification model (should output logits)
        num_classes: Number of output classes
        # Loss configuration
        loss_type: Loss function - "cross_entropy", "focal", "bce" (default: "cross_entropy")
        label_smoothing: Label smoothing factor (default: 0.0)
        focal_alpha: Focal loss alpha parameter (default: 0.25)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        # Training parameters
        learning_rate: Learning rate (default: 1e-3)
        lr_warmup_steps: Number of warmup steps (default: 500)
        lr_scheduler: Learning rate schedule after warmup - "constant", "linear", or "cosine" (default: "cosine")
        max_lr_steps: Maximum steps for lr decay (required for linear/cosine, ignored for constant)
        # Optimizer parameters
        optimizer: Optimizer type - "adam", "adamw", or "sgd" (default: "adamw")
        weight_decay: Weight decay (default: 0.01)
        adam_beta1: Adam beta1 (default: 0.9)
        adam_beta2: Adam beta2 (default: 0.999)
        adam_epsilon: Adam epsilon (default: 1e-8)
        sgd_momentum: SGD momentum (default: 0.9)
        sgd_nesterov: Use Nesterov momentum for SGD (default: True)
        # Metrics configuration
        top_k: Compute top-k accuracy for k values (default: [1, 5])
        compute_per_class_metrics: Compute per-class precision/recall/F1 (default: False)
        # Visualization
        log_samples_every_n_epochs: Log sample predictions every N epochs (default: 5)
        num_samples_to_log: Number of samples to log (default: 16)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        # Loss configuration
        loss_type: Literal["cross_entropy", "focal", "bce"] = "cross_entropy",
        label_smoothing: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Training parameters
        learning_rate: float = 1e-3,
        lr_warmup_steps: int = 500,
        lr_scheduler: Literal["constant", "linear", "cosine"] = "cosine",
        max_lr_steps: Optional[int] = None,
        # Optimizer parameters
        optimizer: Literal["adam", "adamw", "sgd"] = "adamw",
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        sgd_momentum: float = 0.9,
        sgd_nesterov: bool = True,
        # Metrics configuration
        top_k: list[int] = None,
        compute_per_class_metrics: bool = False,
        # Visualization
        log_samples_every_n_epochs: int = 5,
        num_samples_to_log: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        # Store model
        self.model = model
        
        # Store parameters
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_scheduler_type = lr_scheduler
        self.max_lr_steps = max_lr_steps
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.sgd_momentum = sgd_momentum
        self.sgd_nesterov = sgd_nesterov
        self.top_k = top_k if top_k is not None else [1, 5]
        self.compute_per_class_metrics = compute_per_class_metrics
        self.log_samples_every_n_epochs = log_samples_every_n_epochs
        self.num_samples_to_log = num_samples_to_log
        
        # Validate binary classification setup
        if loss_type == "bce" and num_classes != 2:
            raise ValueError("BCE loss requires num_classes=2")
        
        # Initialize metrics
        task = "binary" if num_classes == 2 else "multiclass"
        
        # Training metrics
        self.train_acc = Accuracy(task=task, num_classes=num_classes if task == "multiclass" else None)
        self.train_f1 = F1Score(task=task, num_classes=num_classes if task == "multiclass" else None)
        
        # Validation metrics
        self.val_acc = Accuracy(task=task, num_classes=num_classes if task == "multiclass" else None)
        self.val_f1 = F1Score(task=task, num_classes=num_classes if task == "multiclass" else None)
        
        # Test metrics
        self.test_acc = Accuracy(task=task, num_classes=num_classes if task == "multiclass" else None)
        self.test_f1 = F1Score(task=task, num_classes=num_classes if task == "multiclass" else None)
        
        # Top-k accuracy metrics (only for multiclass)
        if task == "multiclass":
            self.val_top_k_metrics = nn.ModuleDict({
                f"top{k}": Accuracy(task="multiclass", num_classes=num_classes, top_k=k)
                for k in self.top_k
            })
            self.test_top_k_metrics = nn.ModuleDict({
                f"top{k}": Accuracy(task="multiclass", num_classes=num_classes, top_k=k)
                for k in self.top_k
            })
        
        # Per-class metrics (optional)
        if compute_per_class_metrics and task == "multiclass":
            self.val_precision = Precision(task=task, num_classes=num_classes, average=None)
            self.val_recall = Recall(task=task, num_classes=num_classes, average=None)
            self.val_f1_per_class = F1Score(task=task, num_classes=num_classes, average=None)
            
            self.test_precision = Precision(task=task, num_classes=num_classes, average=None)
            self.test_recall = Recall(task=task, num_classes=num_classes, average=None)
            self.test_f1_per_class = F1Score(task=task, num_classes=num_classes, average=None)
        
        # Confusion matrix for validation
        self.val_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes if task == "multiclass" else None)
        
        # Storage for sample logging
        self.val_sample_images = []
        self.val_sample_labels = []
        self.val_sample_preds = []
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.
        
        Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
        
        Args:
            logits: Model predictions [B, num_classes]
            targets: Ground truth labels [B]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def get_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on loss type."""
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        
        elif self.loss_type == "focal":
            return self.focal_loss(logits, targets)
        
        elif self.loss_type == "bce":
            # For binary classification with BCEWithLogitsLoss
            # Expects logits of shape [B, 1] or [B] and targets of shape [B]
            if logits.dim() > 1 and logits.shape[1] == 2:
                # If model outputs [B, 2], take second column
                logits = logits[:, 1]
            elif logits.dim() > 1 and logits.shape[1] == 1:
                # If model outputs [B, 1], squeeze
                logits = logits.squeeze(1)
            
            targets = targets.float()
            return F.binary_cross_entropy_with_logits(logits, targets)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.get_loss(logits, targets)
        
        # Get predictions for metrics
        if self.loss_type == "bce":
            preds = torch.sigmoid(logits if logits.dim() == 1 else logits[:, 1])
            preds = (preds > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        images, targets = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.get_loss(logits, targets)
        
        # Get predictions for metrics
        if self.loss_type == "bce":
            preds = torch.sigmoid(logits if logits.dim() == 1 else logits[:, 1])
            preds = (preds > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        
        # Update top-k metrics
        if hasattr(self, 'val_top_k_metrics'):
            for metric in self.val_top_k_metrics.values():
                metric(logits, targets)
        
        # Update per-class metrics
        if self.compute_per_class_metrics and hasattr(self, 'val_precision'):
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
            self.val_f1_per_class(preds, targets)
        
        # Update confusion matrix
        self.val_confusion_matrix(preds, targets)
        
        # Store samples for visualization (only first batch, limited number)
        if batch_idx == 0 and len(self.val_sample_images) < self.num_samples_to_log:
            num_to_store = min(self.num_samples_to_log - len(self.val_sample_images), images.shape[0])
            self.val_sample_images.append(images[:num_to_store].cpu())
            self.val_sample_labels.append(targets[:num_to_store].cpu())
            self.val_sample_preds.append(preds[:num_to_store].cpu())
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val/f1", self.val_f1, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log additional metrics at the end of validation."""
        # Log top-k accuracies
        if hasattr(self, 'val_top_k_metrics'):
            for name, metric in self.val_top_k_metrics.items():
                self.log(f"val/{name}_acc", metric.compute(), sync_dist=True)
        
        # Log per-class metrics
        if self.compute_per_class_metrics and hasattr(self, 'val_precision'):
            precision = self.val_precision.compute()
            recall = self.val_recall.compute()
            f1 = self.val_f1_per_class.compute()
            
            for i in range(self.num_classes):
                self.log(f"val/class_{i}_precision", precision[i], sync_dist=True)
                self.log(f"val/class_{i}_recall", recall[i], sync_dist=True)
                self.log(f"val/class_{i}_f1", f1[i], sync_dist=True)
        
        # Log confusion matrix
        if self.logger is not None and self.current_epoch % self.log_samples_every_n_epochs == 0:
            cm = self.val_confusion_matrix.compute()
            
            try:
                if hasattr(self.logger.experiment, 'add_figure'):
                    # TensorBoard
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 10))
                    im = ax.imshow(cm.cpu().numpy(), cmap='Blues')
                    ax.figure.colorbar(im, ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title('Confusion Matrix')
                    self.logger.experiment.add_figure(
                        "val/confusion_matrix", fig, global_step=self.global_step
                    )
                    plt.close(fig)
                else:
                    # WandB - log as heatmap instead of using confusion_matrix plot
                    import wandb
                    self.logger.experiment.log({
                        "val/confusion_matrix": wandb.Image(cm.cpu().numpy()),
                        "global_step": self.global_step
                    })
            except (AttributeError, ImportError) as e:
                print(f"Warning: Could not log confusion matrix - {type(e).__name__}: {e}")
        
        # Log sample predictions
        if (self.val_sample_images and 
            self.logger is not None and 
            self.current_epoch % self.log_samples_every_n_epochs == 0):
            
            images = torch.cat(self.val_sample_images, dim=0)[:self.num_samples_to_log]
            labels = torch.cat(self.val_sample_labels, dim=0)[:self.num_samples_to_log]
            preds = torch.cat(self.val_sample_preds, dim=0)[:self.num_samples_to_log]
            
            # Normalize images to [0, 1] if needed
            if images.min() < 0:
                images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            # Create grid with labels
            grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)
            
            try:
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoard
                    self.logger.experiment.add_image(
                        "val/samples", grid, global_step=self.global_step
                    )
                    # Log predictions as text
                    pred_text = "\n".join([
                        f"Sample {i}: True={labels[i].item()}, Pred={preds[i].item()}"
                        for i in range(min(8, len(labels)))
                    ])
                    self.logger.experiment.add_text(
                        "val/predictions", pred_text, global_step=self.global_step
                    )
                else:
                    # WandB
                    import wandb
                    self.logger.experiment.log({
                        "val/samples": wandb.Image(
                            grid,
                            caption=f"Epoch {self.current_epoch}"
                        ),
                        "global_step": self.global_step
                    })
            except (AttributeError, ImportError) as e:
                print(f"Warning: Could not log sample images - {type(e).__name__}: {e}")
        
        # Clear sample storage
        self.val_sample_images = []
        self.val_sample_labels = []
        self.val_sample_preds = []
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step."""
        images, targets = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.get_loss(logits, targets)
        
        # Get predictions for metrics
        if self.loss_type == "bce":
            preds = torch.sigmoid(logits if logits.dim() == 1 else logits[:, 1])
            preds = (preds > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        
        # Update top-k metrics
        if hasattr(self, 'test_top_k_metrics'):
            for metric in self.test_top_k_metrics.values():
                metric(logits, targets)
        
        # Update per-class metrics
        if self.compute_per_class_metrics and hasattr(self, 'test_precision'):
            self.test_precision(preds, targets)
            self.test_recall(preds, targets)
            self.test_f1_per_class(preds, targets)
        
        # Log metrics
        self.log("test/loss", loss, sync_dist=True)
        self.log("test/acc", self.test_acc, sync_dist=True)
        self.log("test/f1", self.test_f1, sync_dist=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Log additional metrics at the end of testing."""
        # Log top-k accuracies
        if hasattr(self, 'test_top_k_metrics'):
            for name, metric in self.test_top_k_metrics.items():
                self.log(f"test/{name}_acc", metric.compute(), sync_dist=True)
        
        # Log per-class metrics
        if self.compute_per_class_metrics and hasattr(self, 'test_precision'):
            precision = self.test_precision.compute()
            recall = self.test_recall.compute()
            f1 = self.test_f1_per_class.compute()
            
            for i in range(self.num_classes):
                self.log(f"test/class_{i}_precision", precision[i], sync_dist=True)
                self.log(f"test/class_{i}_recall", recall[i], sync_dist=True)
                self.log(f"test/class_{i}_f1", f1[i], sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
                nesterov=self.sgd_nesterov
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Create learning rate scheduler with warmup and different decay schedules
        if self.lr_scheduler_type == "constant":
            # Constant LR after warmup
            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return step / self.lr_warmup_steps
                return 1.0
                
        elif self.lr_scheduler_type == "linear":
            # Linear decay after warmup
            if self.max_lr_steps is None:
                raise ValueError("max_lr_steps must be specified for linear scheduler")
            
            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return step / self.lr_warmup_steps
                elif step >= self.max_lr_steps:
                    return 0.0
                else:
                    # Linear decay from 1.0 to 0.0
                    progress = (step - self.lr_warmup_steps) / (self.max_lr_steps - self.lr_warmup_steps)
                    return 1.0 - progress
                    
        elif self.lr_scheduler_type == "cosine":
            # Cosine annealing after warmup
            if self.max_lr_steps is None:
                raise ValueError("max_lr_steps must be specified for cosine scheduler")
            
            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return step / self.lr_warmup_steps
                elif step >= self.max_lr_steps:
                    return 0.0
                else:
                    # Cosine decay from 1.0 to 0.0
                    progress = (step - self.lr_warmup_steps) / (self.max_lr_steps - self.lr_warmup_steps)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }