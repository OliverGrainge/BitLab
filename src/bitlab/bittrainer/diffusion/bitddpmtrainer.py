"""
PyTorch Lightning Trainer for Unconditional Diffusion Models

This trainer implements DDPM (Denoising Diffusion Probabilistic Models) training
with support for various noise schedules, loss types, and sampling strategies.
"""

import math
from typing import Optional, Literal, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from bitlab.bitmodels.auto import BitAutoModel
from bitlab.bitmodels.unet.config import BitUNetConfig


class EMAModel:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                shadow = self.shadow[name].to(param.device, dtype=param.dtype)
                new_average = (1.0 - self.decay) * param.data + self.decay * shadow
                self.shadow[name] = new_average.detach().clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.detach().clone()
                shadow = self.shadow[name].to(param.device, dtype=param.dtype)
                param.data.copy_(shadow)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
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
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    
    elif schedule == "cosine":
        # Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
        steps = num_timesteps + 1
        s = 0.008
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
        model: The denoising model (e.g., BitUNetModel instance)
        image_size: Size of images for sampling (must match model's expected input)
        in_channels: Number of input channels (must match model's expected input)
        num_timesteps: Number of diffusion timesteps (default: 1000)
        beta_schedule: Noise schedule type - "linear", "cosine", or "quadratic" (default: "linear")
        beta_start: Starting beta value for linear schedule (default: 0.0001)
        beta_end: Ending beta value for linear schedule (default: 0.02)
        loss_type: Loss function - "l1", "l2", or "huber" (default: "l2")
        prediction_type: What model predicts - "epsilon", "x0", or "v" (default: "epsilon")
        learning_rate: Learning rate (default: 1e-4)
        lr_warmup_steps: Number of warmup steps (default: 1000)
        use_ema: Use exponential moving average (default: True)
        ema_decay: EMA decay rate (default: 0.9999)
        num_sample_steps: Number of DDIM sampling steps (default: 50)
        sample_every_n_steps: Generate samples every N steps (default: 1000)
        num_samples: Number of samples to generate (default: 16)
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
        
        # Store model
        self.model = model
        
        # Store parameters
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
        if use_ema:
            self.ema_model = EMAModel(self.model, decay=ema_decay)
        else:
            self.ema_model = None
        
        # Setup diffusion schedule
        self.register_diffusion_schedule(
            beta_schedule,
            num_timesteps,
            beta_start,
            beta_end
        )
        
        # Validation image tracking
        self.validation_z = None
    
    def register_diffusion_schedule(
        self,
        schedule: str,
        num_timesteps: int,
        beta_start: float,
        beta_end: float
    ):
        """Register diffusion schedule parameters as buffers."""
        betas = get_beta_schedule(schedule, num_timesteps, beta_start, beta_end)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers so they're moved to the correct device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
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
        Forward diffusion process: sample x_t from q(x_t | x_0).
        
        Args:
            x_start: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Sampled noise [B, C, H, W]
        
        Returns:
            Noisy images at timestep t
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Get training target based on prediction type."""
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "x0":
            return x_start
        elif self.prediction_type == "v":
            # v-prediction: v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_start
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def predict_x0_from_output(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Convert model output to x0 prediction."""
        if self.prediction_type == "epsilon":
            # x0 = (x_t - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)
            sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t][:, None, None, None]
            sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None]
            return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * model_output
        
        elif self.prediction_type == "x0":
            return model_output
        
        elif self.prediction_type == "v":
            # x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * model_output
        
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on loss type."""
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "l2":
            return F.mse_loss(pred, target)
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x_start = batch
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict
        model_output = self.model(x_t, t)
        
        # Get target
        target = self.get_target(x_start, noise, t)
        
        # Compute loss
        loss = self.get_loss(model_output, target)
        
        # Log
        self.log("train/loss", loss, prog_bar=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training batch."""
        if self.ema_model is not None:
            self.ema_model.update()
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Validation step - compute validation loss."""
        x_start = batch
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict
        model_output = self.model(x_t, t)
        
        # Get target
        target = self.get_target(x_start, noise, t)
        
        # Compute loss
        loss = self.get_loss(model_output, target)
        
        # Log
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        use_ema: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Sample using DDIM (Denoising Diffusion Implicit Models).
        
        Args:
            batch_size: Number of samples to generate
            num_steps: Number of denoising steps (default: uses self.num_sample_steps)
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
            use_ema: Use EMA model if available (default: uses self.use_ema)
        
        Returns:
            Generated images [B, C, H, W]
        """
        if num_steps is None:
            num_steps = self.num_sample_steps
        if use_ema is None:
            use_ema = self.use_ema
            
        model = self.model
        if use_ema and self.ema_model is not None:
            self.ema_model.apply_shadow()
        
        model.eval()
        
        # Create time steps for DDIM
        c = self.num_timesteps // num_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c, device=self.device)
        ddim_timesteps_prev = torch.cat([torch.tensor([0], device=self.device), ddim_timesteps[:-1]])
        
        # Start from random noise
        x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size, device=self.device)
        
        # Iterative denoising
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            t_prev = ddim_timesteps_prev[i]
            
            # Expand timestep to batch dimension
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise/x0/v
            model_output = model(x, t_batch)
            
            # Predict x0
            pred_x0 = self.predict_x0_from_output(x, t_batch, model_output)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
            
            # Compute sigma
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            
            # Compute predicted noise
            if self.prediction_type == "epsilon":
                pred_noise = model_output
            else:
                # Reconstruct noise from x0 prediction
                pred_noise = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
            
            # Compute x_{t-1}
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        if use_ema and self.ema_model is not None:
            self.ema_model.restore()
        
        model.train()
        return x
    
    def on_validation_epoch_end(self):
        """Generate and log sample images at the end of validation."""
        if self.global_step % self.sample_every_n_steps == 0:
            samples = self.sample_ddim(
                batch_size=self.num_samples,
                num_steps=self.num_sample_steps,
                use_ema=self.use_ema
            )
            
            # Normalize to [0, 1] for visualization
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)
            
            # Create grid
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=False)
            
            # Log to tensorboard
            if self.logger is not None:
                self.logger.experiment.add_image(
                    "samples",
                    grid,
                    global_step=self.global_step
                )
    
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
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Create learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.lr_warmup_steps:
                return step / self.lr_warmup_steps
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

