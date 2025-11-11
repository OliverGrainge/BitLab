"""
PyTorch Lightning Trainer for Multiclass Image Classification

This trainer implements supervised learning for multiclass image classification with support
for various loss types, learning rate schedules, and optimization strategies.
"""

import math
from typing import Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from bitlab.bittrainer.callbacks import (ClassificationVisualizationLogger,
                                         GradientNormLogger,
                                         WeightHistogramLogger)


class BitImageClassificationTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for multiclass image classification.

    Implements supervised learning with support for:
    - Multiple loss functions (cross-entropy, focal loss with label smoothing)
    - Various learning rate schedules (constant, linear, cosine)
    - Comprehensive metrics tracking (accuracy, F1, precision, recall)
    - Automatic mixed precision support
    - Top-k accuracy metrics

    Args:
        model: The classification model (should output logits)
        num_classes: Number of output classes (must be >= 2)

        # Loss configuration
        loss_type: Loss function - "cross_entropy" or "focal" (default: "cross_entropy")
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
        loss_type: Literal["cross_entropy", "focal"] = "cross_entropy",
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

        # Validate inputs
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        # Store model and parameters
        self.model = model
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
        self.top_k = top_k if top_k is not None else [1, 5] if num_classes > 5 else [1]
        self.compute_per_class_metrics = compute_per_class_metrics
        self.log_samples_every_n_epochs = log_samples_every_n_epochs
        self.num_samples_to_log = num_samples_to_log

        # Initialize metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize all metrics for training, validation, and testing."""
        # Determine task type
        task = "binary" if self.num_classes == 2 else "multiclass"

        # Training metrics
        self.train_acc = Accuracy(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )
        self.train_f1 = F1Score(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )

        # Validation metrics
        self.val_acc = Accuracy(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )
        self.val_f1 = F1Score(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )
        self.val_confusion_matrix = ConfusionMatrix(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )

        # Test metrics
        self.test_acc = Accuracy(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )
        self.test_f1 = F1Score(
            task=task, num_classes=self.num_classes if task == "multiclass" else None
        )

        # Top-k accuracy metrics (only for multiclass with > 5 classes)
        if task == "multiclass" and self.num_classes > 5:
            self.val_top_k_metrics = nn.ModuleDict(
                {
                    f"top{k}": Accuracy(
                        task="multiclass", num_classes=self.num_classes, top_k=k
                    )
                    for k in self.top_k
                    if k < self.num_classes
                }
            )
            self.test_top_k_metrics = nn.ModuleDict(
                {
                    f"top{k}": Accuracy(
                        task="multiclass", num_classes=self.num_classes, top_k=k
                    )
                    for k in self.top_k
                    if k < self.num_classes
                }
            )

        # Per-class metrics (optional)
        if self.compute_per_class_metrics and task == "multiclass":
            self.val_precision = Precision(
                task=task, num_classes=self.num_classes, average=None
            )
            self.val_recall = Recall(
                task=task, num_classes=self.num_classes, average=None
            )
            self.val_f1_per_class = F1Score(
                task=task, num_classes=self.num_classes, average=None
            )

            self.test_precision = Precision(
                task=task, num_classes=self.num_classes, average=None
            )
            self.test_recall = Recall(
                task=task, num_classes=self.num_classes, average=None
            )
            self.test_f1_per_class = F1Score(
                task=task, num_classes=self.num_classes, average=None
            )

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
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on configured loss type.

        Args:
            logits: Model predictions [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            Loss value
        """
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(
                logits, targets, label_smoothing=self.label_smoothing
            )
        elif self.loss_type == "focal":
            if self.label_smoothing > 0:
                # Apply label smoothing with focal loss
                # First compute smoothed targets
                n_classes = logits.size(-1)
                smoothed = torch.full_like(
                    logits, self.label_smoothing / (n_classes - 1)
                )
                smoothed.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing)

                # Compute focal loss with smoothed targets
                log_probs = F.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)

                # Focal weight
                focal_weight = (1 - probs) ** self.focal_gamma

                # Weighted cross entropy with smoothed labels
                loss = -(smoothed * log_probs * focal_weight).sum(dim=-1)
                return loss.mean() * self.focal_alpha
            else:
                return self.focal_loss(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        images, targets = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.compute_loss(logits, targets)

        # Get predictions for metrics
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/acc", self.train_acc, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)

        # Log learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        images, targets = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.compute_loss(logits, targets)

        # Get predictions for metrics
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_confusion_matrix(preds, targets)

        # Update top-k metrics
        if hasattr(self, "val_top_k_metrics"):
            for metric in self.val_top_k_metrics.values():
                metric(logits, targets)

        # Update per-class metrics
        if self.compute_per_class_metrics and hasattr(self, "val_precision"):
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
            self.val_f1_per_class(preds, targets)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val/f1", self.val_f1, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        """Log additional metrics at the end of validation epoch."""
        # Log top-k accuracies
        if hasattr(self, "val_top_k_metrics"):
            for name, metric in self.val_top_k_metrics.items():
                self.log(f"val/{name}_acc", metric.compute(), sync_dist=True)

        # Log per-class metrics
        if self.compute_per_class_metrics and hasattr(self, "val_precision"):
            precision = self.val_precision.compute()
            recall = self.val_recall.compute()
            f1 = self.val_f1_per_class.compute()

            # Log individual class metrics
            for i in range(self.num_classes):
                self.log(f"val/class_{i}_precision", precision[i], sync_dist=True)
                self.log(f"val/class_{i}_recall", recall[i], sync_dist=True)
                self.log(f"val/class_{i}_f1", f1[i], sync_dist=True)

            # Log mean metrics
            self.log("val/mean_precision", precision.mean(), sync_dist=True)
            self.log("val/mean_recall", recall.mean(), sync_dist=True)
            self.log("val/mean_f1_per_class", f1.mean(), sync_dist=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step."""
        images, targets = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = self.compute_loss(logits, targets)

        # Get predictions for metrics
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

        # Update top-k metrics
        if hasattr(self, "test_top_k_metrics"):
            for metric in self.test_top_k_metrics.values():
                metric(logits, targets)

        # Update per-class metrics
        if self.compute_per_class_metrics and hasattr(self, "test_precision"):
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
        if hasattr(self, "test_top_k_metrics"):
            for name, metric in self.test_top_k_metrics.items():
                self.log(f"test/{name}_acc", metric.compute(), sync_dist=True)

        # Log per-class metrics
        if self.compute_per_class_metrics and hasattr(self, "test_precision"):
            precision = self.test_precision.compute()
            recall = self.test_recall.compute()
            f1 = self.test_f1_per_class.compute()

            # Log individual class metrics
            for i in range(self.num_classes):
                self.log(f"test/class_{i}_precision", precision[i], sync_dist=True)
                self.log(f"test/class_{i}_recall", recall[i], sync_dist=True)
                self.log(f"test/class_{i}_f1", f1[i], sync_dist=True)

            # Log mean metrics
            self.log("test/mean_precision", precision.mean(), sync_dist=True)
            self.log("test/mean_recall", recall.mean(), sync_dist=True)
            self.log("test/mean_f1_per_class", f1.mean(), sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer = self._create_optimizer()

        # Create learning rate scheduler
        scheduler = self._create_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
                nesterov=self.sgd_nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup."""
        if self.lr_scheduler_type == "constant":
            # Constant LR after warmup
            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return float(step) / float(max(1, self.lr_warmup_steps))
                return 1.0

        elif self.lr_scheduler_type == "linear":
            # Linear decay after warmup
            if self.max_lr_steps is None:
                raise ValueError("max_lr_steps must be specified for linear scheduler")

            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return float(step) / float(max(1, self.lr_warmup_steps))
                elif step >= self.max_lr_steps:
                    return 0.0
                else:
                    # Linear decay from 1.0 to 0.0
                    progress = (step - self.lr_warmup_steps) / (
                        self.max_lr_steps - self.lr_warmup_steps
                    )
                    return max(0.0, 1.0 - progress)

        elif self.lr_scheduler_type == "cosine":
            # Cosine annealing after warmup
            if self.max_lr_steps is None:
                raise ValueError("max_lr_steps must be specified for cosine scheduler")

            def lr_lambda(step):
                if step < self.lr_warmup_steps:
                    return float(step) / float(max(1, self.lr_warmup_steps))
                elif step >= self.max_lr_steps:
                    return 0.0
                else:
                    # Cosine decay from 1.0 to 0.0
                    progress = (step - self.lr_warmup_steps) / (
                        self.max_lr_steps - self.lr_warmup_steps
                    )
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def configure_callbacks(self):
        """Configure callbacks for training."""
        callbacks = super().configure_callbacks() or []

        # Add custom callbacks
        callbacks.extend(
            [
                GradientNormLogger(every_n_steps=100),
                WeightHistogramLogger(),
                ClassificationVisualizationLogger(
                    num_samples_to_log=self.num_samples_to_log,
                    log_samples_every_n_epochs=self.log_samples_every_n_epochs,
                ),
            ]
        )

        return callbacks
