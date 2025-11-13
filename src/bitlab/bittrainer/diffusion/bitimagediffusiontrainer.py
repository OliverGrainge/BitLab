"""
PyTorch Lightning Trainer for Unconditional Diffusion Models

This trainer now delegates all image generation to the model's ImageGenerationMixin,
resulting in a much simpler and cleaner implementation.
"""

import math
from typing import Any, Dict, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bittrainer.callbacks import (
    ImageSampleCallback,
    GradientNormLogger,
    WeightHistogramLogger,
    CleanFIDCallback
)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA model.
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                shadow = self.shadow[name].to(param.device, dtype=param.dtype)
                new_average = (1.0 - self.decay) * param.data + self.decay * shadow
                self.shadow[name] = new_average.detach().clone()

    def apply_shadow(self) -> None:
        """Apply shadow parameters to model (backup current parameters)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.detach().clone()
                shadow = self.shadow[name].to(param.device, dtype=param.dtype)
                param.data.copy_(shadow)

    def restore(self) -> None:
        """Restore original parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        """Return EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {
                name: param.detach().cpu() for name, param in self.shadow.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state.get("decay", self.decay)
        shadow = state.get("shadow", {})

        # Get device and dtype from model
        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = model_param.dtype

        self.shadow = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name in shadow:
                self.shadow[name] = shadow[name].to(device=device, dtype=dtype)
            else:
                # Fallback to current parameter value if missing
                self.shadow[name] = param.data.detach().clone()

        self.backup = {}


class BitImageDiffusionTrainer(pl.LightningModule):
    """
    Simplified PyTorch Lightning trainer for unconditional diffusion models.

    This trainer delegates all image generation and sampling logic to the model's
    ImageGenerationMixin, resulting in cleaner separation of concerns.

    The model must inherit from ImageGenerationMixin to work with this trainer.

    Args:
        model: The denoising model (must inherit from ImageGenerationMixin)

        # Loss configuration
        loss_type: Loss function - "l1", "l2", or "huber" (default: "l2")

        # Training parameters
        learning_rate: Learning rate (default: 1e-4)
        lr_warmup_steps: Number of warmup steps (default: 1000)
        lr_scheduler: LR schedule - "constant", "linear", or "cosine" (default: "constant")
        max_lr_steps: Maximum steps for lr decay (required for linear/cosine)

        # EMA parameters
        use_ema: Use exponential moving average (default: True)
        ema_decay: EMA decay rate (default: 0.9999)

        # Sampling parameters (for logging/validation)
        num_sample_steps: Number of DDIM sampling steps (default: 50)
        sample_method: Sampling method - "ddim" or "ddpm" (default: "ddim")
        sample_eta: DDIM stochasticity (default: 0.0)
        num_samples: Number of samples to generate for logging (default: 16)

        # Optimizer parameters
        optimizer: Optimizer type - "adam" or "adamw" (default: "adamw")
        weight_decay: Weight decay (default: 0.0)
        adam_beta1: Adam beta1 (default: 0.9)
        adam_beta2: Adam beta2 (default: 0.999)
        adam_epsilon: Adam epsilon (default: 1e-8)
    """

    def __init__(
        self,
        model: nn.Module,
        # Loss configuration
        loss_type: Literal["l1", "l2", "huber"] = "l2",
        # Training parameters
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 1000,
        lr_scheduler: Literal["constant", "linear", "cosine"] = "constant",
        max_lr_steps: Optional[int] = None,
        # EMA parameters
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        # Sampling parameters
        num_sample_steps: int = 50,
        sample_method: Literal["ddim", "ddpm"] = "ddim",
        sample_eta: float = 0.0,
        num_samples: int = 16,
        # Optimizer parameters
        optimizer: Literal["adam", "adamw"] = "adamw",
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        # Callback configuration
        log_every_n_steps: int = 10_000,
        image_sample_num_images: int = 16,
        image_sample_nrow: Optional[int] = None,
        image_sample_log_key: str = "train/image_grid",
        image_sample_use_ema: Optional[bool] = None,
        fid_dataset_name: str = "cifar10",
        fid_dataset_res: int = 32,
        fid_dataset_split: str = "train",
        fid_mode: str = "clean",
        fid_num_gen: int = 1_000,
        fid_batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])



        # Validate that model has generation capabilities
        if not hasattr(model, "generate"):
            raise ValueError(
                "Model must inherit from ImageGenerationMixin to use this trainer. "
                "The model should have a generate() method for image generation."
            )

        # Store model and parameters
        self.model = model
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_scheduler_type = lr_scheduler
        self.max_lr_steps = max_lr_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.num_sample_steps = num_sample_steps
        self.sample_method = sample_method
        self.sample_eta = sample_eta
        self.num_samples = num_samples
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive.")
        self.log_every_n_steps = log_every_n_steps
        self.image_sample_num_images = image_sample_num_images
        self.image_sample_nrow = image_sample_nrow
        self.image_sample_log_key = image_sample_log_key
        self.image_sample_use_ema = (
            image_sample_use_ema if image_sample_use_ema is not None else self.use_ema
        )
        self.fid_dataset_name = fid_dataset_name
        self.fid_dataset_res = fid_dataset_res
        self.fid_dataset_split = fid_dataset_split
        self.fid_mode = fid_mode
        self.fid_num_gen = fid_num_gen
        self.fid_batch_size = fid_batch_size

        # Setup EMA
        self.ema_model = EMAModel(self.model, decay=ema_decay) if use_ema else None

        # For consistent validation sampling
        self.validation_noise = None

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on configured loss type.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Loss value
        """
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "l2":
            return F.mse_loss(pred, target)
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, t)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step - now simplified by using model's generation capabilities.
        
        The model handles all diffusion schedule management through its mixin.
        """
        x_start = batch
        batch_size = x_start.shape[0]

        # Sample random timesteps uniformly
        t = torch.randint(
            0, self.model.num_timesteps, (batch_size,), device=self.device
        )

        # Sample Gaussian noise
        noise = torch.randn_like(x_start)

        # Forward diffusion process (handled by model's mixin)
        x_t = self.model.q_sample(x_start, t, noise)

        # Predict with model
        model_output = self(x_t, t)

        # Compute target based on model's prediction type (handled by mixin)
        target = self.model.get_training_target(x_start, noise, t)

        # Compute loss
        loss = self.compute_loss(model_output, target)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False)

        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update EMA model after each training batch."""
        if self.ema_model is not None:
            self.ema_model.update()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step - compute validation loss."""
        x_start = batch
        batch_size = x_start.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0, self.model.num_timesteps, (batch_size,), device=self.device
        )

        # Sample noise
        noise = torch.randn_like(x_start)

        # Forward diffusion (handled by model)
        x_t = self.model.q_sample(x_start, t, noise)

        # Predict with model
        model_output = self(x_t, t)

        # Compute target (handled by model)
        target = self.model.get_training_target(x_start, noise, t)

        # Compute loss
        loss = self.compute_loss(model_output, target)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def generate_samples(
        self,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        method: Optional[str] = None,
        eta: Optional[float] = None,
        use_ema: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using the model's generation capabilities.
        
        This is a convenience wrapper around model.generate() that handles EMA.
        All generation logic is delegated to the model's ImageGenerationMixin.
        
        Args:
            batch_size: Number of samples (default: self.num_samples)
            num_steps: Number of sampling steps (default: self.num_sample_steps)
            method: Sampling method (default: self.sample_method)
            eta: DDIM stochasticity (default: self.sample_eta)
            use_ema: Use EMA model (default: self.use_ema)
            noise: Initial noise (default: random)
            **kwargs: Additional arguments passed to model.generate()
        
        Returns:
            Generated images [B, C, H, W]
        """
        # Set defaults
        if batch_size is None:
            batch_size = self.num_samples
        if num_steps is None:
            num_steps = self.num_sample_steps
        if method is None:
            method = self.sample_method
        if eta is None:
            eta = self.sample_eta
        if use_ema is None:
            use_ema = self.use_ema

        # Apply EMA if requested
        if use_ema and self.ema_model is not None:
            self.ema_model.apply_shadow()

        try:
            # Delegate to model's generate method
            samples = self.model.generate(
                batch_size=batch_size,
                num_steps=num_steps,
                method=method,
                eta=eta,
                noise=noise,
                device=self.device,
                **kwargs,
            )
        finally:
            # Restore original parameters if EMA was used
            if use_ema and self.ema_model is not None:
                self.ema_model.restore()

        return samples

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save EMA state with checkpoint."""
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None and self.ema_model is not None:
            self.ema_model.load_state_dict(ema_state)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer = self._create_optimizer()

        # Create scheduler
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
                    # Linear decay
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
                    # Cosine decay
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

        callbacks.extend(
            [
                GradientNormLogger(every_n_steps=self.log_every_n_steps),
                WeightHistogramLogger(log_every_n_steps=self.log_every_n_steps),
                ImageSampleCallback(
                    num_images=self.image_sample_num_images,
                    every_n_steps=self.log_every_n_steps,
                    nrow=self.image_sample_nrow,
                    log_key=self.image_sample_log_key,
                    use_ema=self.image_sample_use_ema,
                ),
                CleanFIDCallback(
                    dataset_name=self.fid_dataset_name,
                    dataset_res=self.fid_dataset_res,
                    dataset_split=self.fid_dataset_split,
                    mode=self.fid_mode,
                    num_gen=self.fid_num_gen,
                    batch_size=self.fid_batch_size,
                    every_n_steps=self.log_every_n_steps,
                ),
            ]
        )

        return callbacks