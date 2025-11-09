"""
PyTorch Lightning Trainer for Unconditional Diffusion Models

This trainer implements DDPM (Denoising Diffusion Probabilistic Models) training
with support for various noise schedules, loss types, and sampling strategies.
"""

import math
from typing import Optional, Literal, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from bitlab.bittrainer.callbacks import (
    DiffusionSampleLogger,
    GradientNormLogger,
    WeightHistogramLogger,
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
            "shadow": {name: param.detach().cpu() for name, param in self.shadow.items()},
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


def get_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> torch.Tensor:
    """
    Get beta schedule for diffusion process.
    
    Args:
        schedule: Type of schedule ("linear", "cosine", "quadratic")
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta values for each timestep
    
    Raises:
        ValueError: If schedule type is unknown
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    
    elif schedule == "cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        steps = num_timesteps + 1
        s = 0.008  # Small offset to prevent beta from being too small near t=0
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    elif schedule == "quadratic":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class BitDDPMTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for unconditional diffusion models.
    
    Implements DDPM training with support for:
    - Multiple noise schedules (linear, cosine, quadratic)
    - Different prediction types (epsilon, x0, v-prediction)
    - EMA model tracking
    - DDIM sampling for fast inference
    
    Args:
        model: The denoising model (e.g., UNet)
        image_size: Size of images for sampling
        in_channels: Number of input channels (default: 3)
        
        # Diffusion parameters
        num_timesteps: Number of diffusion timesteps (default: 1000)
        beta_schedule: Noise schedule type (default: "linear")
        beta_start: Starting beta value (default: 0.0001)
        beta_end: Ending beta value (default: 0.02)
        
        # Loss configuration
        loss_type: Loss function - "l1", "l2", or "huber" (default: "l2")
        prediction_type: What model predicts - "epsilon", "x0", or "v" (default: "epsilon")
        
        # Training parameters
        learning_rate: Learning rate (default: 1e-4)
        lr_warmup_steps: Number of warmup steps (default: 1000)
        lr_scheduler: LR schedule - "constant", "linear", or "cosine" (default: "constant")
        max_lr_steps: Maximum steps for lr decay (required for linear/cosine)
        
        # EMA parameters
        use_ema: Use exponential moving average (default: True)
        ema_decay: EMA decay rate (default: 0.9999)
        
        # Sampling parameters
        num_sample_steps: Number of DDIM sampling steps (default: 50)
        sample_every_n_steps: Generate samples every N steps (default: 1000)
        num_samples: Number of samples to generate (default: 16)
        
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
        image_size: int,
        in_channels: int = 3,
        # Diffusion parameters
        num_timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # Loss configuration
        loss_type: Literal["l1", "l2", "huber"] = "l2",
        prediction_type: Literal["epsilon", "x0", "v"] = "epsilon",
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
        sample_every_n_steps: int = 1000,
        num_samples: int = 16,
        # Optimizer parameters
        optimizer: Literal["adam", "adamw"] = "adamw",
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        # Store model and parameters
        self.model = model
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_scheduler_type = lr_scheduler
        self.max_lr_steps = max_lr_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.num_sample_steps = num_sample_steps
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        
        # Setup EMA
        self.ema_model = EMAModel(self.model, decay=ema_decay) if use_ema else None
        
        # Setup diffusion schedule
        self._setup_diffusion_schedule()
        
        # For consistent validation sampling
        self.validation_noise = None
    
    def _setup_diffusion_schedule(self) -> None:
        """Register diffusion schedule parameters as buffers."""
        betas = get_beta_schedule(
            self.beta_schedule,
            self.num_timesteps,
            self.beta_start,
            self.beta_end
        )
        
        # Calculate alpha values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register core schedule parameters
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Precalculated values for forward diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Precalculated values for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).
        
        Args:
            x_start: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Gaussian noise [B, C, H, W]
        
        Returns:
            Noisy images at timestep t
        """
        # Extract coefficients for timestep t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from a 1D tensor for a batch of indices.
        
        Args:
            values: 1D tensor of values
            t: Batch of indices [B]
            x_shape: Shape of tensor to extract for
        
        Returns:
            Extracted values reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        # Reshape to [batch_size, 1, 1, 1] for broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def compute_training_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training target based on prediction type.
        
        Args:
            x_start: Original images [B, C, H, W]
            noise: Gaussian noise [B, C, H, W]
            t: Timesteps [B]
        
        Returns:
            Training target
        """
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "x0":
            return x_start
        elif self.prediction_type == "v":
            # v-parameterization: v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_start
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def predict_x0_from_model_output(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model output to x0 prediction based on parameterization.
        
        Args:
            x_t: Noisy images at timestep t [B, C, H, W]
            t: Timesteps [B]
            model_output: Model prediction [B, C, H, W]
        
        Returns:
            Predicted x0
        """
        if self.prediction_type == "epsilon":
            # x0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
            sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * model_output
        
        elif self.prediction_type == "x0":
            return model_output
        
        elif self.prediction_type == "v":
            # x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * model_output
        
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
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
        """Training step."""
        x_start = batch
        batch_size = x_start.shape[0]
        
        # Sample random timesteps uniformly
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Sample Gaussian noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion process
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict with model
        model_output = self(x_t, t)
        
        # Compute target based on prediction type
        target = self.compute_training_target(x_start, noise, t)
        
        # Compute loss
        loss = self.compute_loss(model_output, target)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
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
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict with model
        model_output = self(x_t, t)
        
        # Compute target
        target = self.compute_training_target(x_start, noise, t)
        
        # Compute loss
        loss = self.compute_loss(model_output, target)
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        temperature: float = 1.0,
        use_ema: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample using DDIM (Denoising Diffusion Implicit Models).
        
        Args:
            batch_size: Number of samples to generate
            num_steps: Number of denoising steps (default: self.num_sample_steps)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
            temperature: Noise temperature for sampling
            use_ema: Use EMA model if available (default: self.use_ema)
            noise: Initial noise (if None, sample random noise)
        
        Returns:
            Generated images [B, C, H, W]
        """
        # Set defaults
        if num_steps is None:
            num_steps = self.num_sample_steps
        if use_ema is None:
            use_ema = self.use_ema
        
        # Setup model
        model = self.model
        if use_ema and self.ema_model is not None:
            self.ema_model.apply_shadow()
        
        model.eval()
        
        # Create timestep schedule for DDIM
        c = self.num_timesteps // num_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c, device=self.device)
        ddim_timesteps_prev = torch.cat([torch.tensor([-1], device=self.device), ddim_timesteps[:-1]])
        
        # Start from noise
        if noise is None:
            x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size, device=self.device)
            x = x * temperature
        else:
            x = noise
        
        # Reverse diffusion process
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            t_prev = ddim_timesteps_prev[i] if i > 0 else torch.tensor(-1, device=self.device)
            
            # Expand timestep to batch
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Model prediction
            model_output = model(x, t_batch)
            
            # Predict x0
            pred_x0 = self.predict_x0_from_model_output(x, t_batch, model_output)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            if i > 0:
                # Get alpha values
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
                
                # Compute variance
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                
                # Predict noise
                if self.prediction_type == "epsilon":
                    pred_noise = model_output
                else:
                    # Reconstruct noise from x0 prediction
                    pred_noise = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
                
                # Compute x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise
                noise = torch.randn_like(x) * temperature if sigma_t > 0 else 0
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                # Final step
                x = pred_x0
        
        # Restore model if using EMA
        if use_ema and self.ema_model is not None:
            self.ema_model.restore()
        
        model.train()
        return x
    
    @torch.no_grad()
    def sample_ddpm(
        self,
        batch_size: int,
        use_ema: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample using original DDPM method (full Markov chain).
        
        Args:
            batch_size: Number of samples to generate
            use_ema: Use EMA model if available
            noise: Initial noise (if None, sample random noise)
        
        Returns:
            Generated images [B, C, H, W]
        """
        return self.sample_ddim(
            batch_size=batch_size,
            num_steps=self.num_timesteps,
            eta=1.0,
            use_ema=use_ema,
            noise=noise
        )

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
                "frequency": 1
            }
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
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
                    progress = (step - self.lr_warmup_steps) / (self.max_lr_steps - self.lr_warmup_steps)
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
                    progress = (step - self.lr_warmup_steps) / (self.max_lr_steps - self.lr_warmup_steps)
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def configure_callbacks(self):
        """Configure callbacks for training."""
        callbacks = super().configure_callbacks() or []
        
        callbacks.extend([
            GradientNormLogger(every_n_steps=100),
            WeightHistogramLogger(),
            DiffusionSampleLogger(
                batch_size=self.num_samples,
                num_steps=self.num_sample_steps,
                use_ema=self.use_ema,
            ),
        ])
        
        return callbacks