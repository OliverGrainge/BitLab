from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional, Union, Tuple, Literal
import math
import torch.nn as nn 

import torch
import torch.nn.functional as F

from bitlab.bitmodels.tasks import ModelTask


class CausalLMMixin:
    """Shared utilities for causal language models."""

    task: ClassVar[str] = ModelTask.CAUSAL_LM.value

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Greedy or sampling-based continuation for causal LMs."""

        self.eval()
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", None)
        )

        generated = input_ids
        past_key_values: Any = None
        cur_len = input_ids.shape[1]

        for _ in range(max_length - cur_len):
            model_inputs = self.prepare_inputs_for_generation(
                generated, past_key_values=past_key_values
            )
            outputs = self.forward(**model_inputs, use_cache=use_cache)

            if use_cache:
                past_key_values = outputs.get("past_key_values", past_key_values)

            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                kth_values = torch.topk(logits, min(top_k, logits.size(-1)))[0][
                    ..., -1, None
                ]
                logits = logits.masked_fill(logits < kth_values, float("-inf"))

            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            if do_sample:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and torch.all(next_token.eq(eos_token_id)):
                break

        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        *,
        past_key_values: Any = None,
    ) -> dict[str, Any]:
        if past_key_values is None:
            return {"input_ids": input_ids}
        return {"input_ids": input_ids[:, -1:], "past_key_values": past_key_values}


class ImageClassificationMixin:
    """Utility helpers for image classification models."""

    task: ClassVar[str] = ModelTask.IMAGE_CLASSIFICATION.value

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        *,
        return_logits: bool = True,
        return_probabilities: bool = True,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        logits = self.forward(inputs)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        result: dict[str, torch.Tensor] = {"predictions": pred}
        if return_logits:
            result["logits"] = logits
        if return_probabilities:
            result["probabilities"] = probs
        return result


class ImageGenerationMixin:
    """
    Mixin that adds image generation capabilities to diffusion models.
    
    This mixin encapsulates all sampling logic (DDIM, DDPM) and diffusion schedule
    management, providing a simple `generate()` interface for the model.
    
    Usage:
        class MyDiffusionModel(ImageGenerationMixin, nn.Module):
            def __init__(self, ...):
                super().__init__(
                    num_timesteps=1000,
                    beta_schedule="linear",
                    ...
                )
                # Your model architecture here
            
            def forward(self, x, t):
                # Your model forward pass
                pass
        
        # Generate images
        model = MyDiffusionModel(...)
        images = model.generate(batch_size=16, num_steps=50)
    """

    task: ClassVar[str] = "image_generation"

    def __init__(
        self,
        *args,
        # Image parameters
        image_size: int,
        in_channels: int = 3,
        # Diffusion schedule parameters
        num_timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # Prediction type
        prediction_type: Literal["epsilon", "x0", "v"] = "epsilon",
        # Default sampling parameters
        default_num_steps: int = 100,
        default_eta: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the image generation mixin.
        
        Args:
            image_size: Size of generated images (assumed square)
            in_channels: Number of image channels (3 for RGB)
            num_timesteps: Total number of diffusion timesteps
            beta_schedule: Type of noise schedule
            beta_start: Starting beta value for linear/quadratic schedules
            beta_end: Ending beta value for linear/quadratic schedules
            prediction_type: What the model predicts ("epsilon", "x0", or "v")
            default_num_steps: Default number of sampling steps
            default_eta: Default DDIM stochasticity parameter
        """
        super().__init__(*args, **kwargs)
        
        # Store generation parameters
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type
        self.default_num_steps = default_num_steps
        self.default_eta = default_eta
        
        # Setup diffusion schedule
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self) -> None:
        """Initialize and register diffusion schedule buffers."""
        betas = self._get_beta_schedule(
            self.beta_schedule,
            self.num_timesteps,
            self.beta_start,
            self.beta_end,
        )

        # Calculate alpha values
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (will be moved to correct device automatically)
        if isinstance(self, nn.Module):
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
            
            # Precompute values for forward diffusion q(x_t | x_0)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
            )
            self.register_buffer(
                "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
            )
            self.register_buffer(
                "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
            )
        else:
            # Fallback if not using nn.Module
            self.betas = betas
            self.alphas = alphas
            self.alphas_cumprod = alphas_cumprod
            self.alphas_cumprod_prev = alphas_cumprod_prev
            self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    @staticmethod
    def _get_beta_schedule(
        schedule: str,
        num_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> torch.Tensor:
        """Get beta schedule for diffusion process."""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)

        elif schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = num_timesteps + 1
            s = 0.008  # Small offset to prevent beta from being too small near t=0
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        elif schedule == "quadratic":
            return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2

        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def _extract(
        self, values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size
    ) -> torch.Tensor:
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

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).
        
        Args:
            x_start: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Gaussian noise [B, C, H, W]
        
        Returns:
            Noisy images at timestep t
        """
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_training_target(
        self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
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
            sqrt_alphas_cumprod_t = self._extract(
                self.sqrt_alphas_cumprod, t, x_start.shape
            )
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            return (
                sqrt_alphas_cumprod_t * noise
                - sqrt_one_minus_alphas_cumprod_t * x_start
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def predict_x0_from_output(
        self, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor
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
            sqrt_recip_alphas_cumprod_t = self._extract(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape
            )
            sqrt_recipm1_alphas_cumprod_t = self._extract(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
            )
            return (
                sqrt_recip_alphas_cumprod_t * x_t
                - sqrt_recipm1_alphas_cumprod_t * model_output
            )

        elif self.prediction_type == "x0":
            return model_output

        elif self.prediction_type == "v":
            # x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            sqrt_alphas_cumprod_t = self._extract(
                self.sqrt_alphas_cumprod, t, x_t.shape
            )
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            return (
                sqrt_alphas_cumprod_t * x_t
                - sqrt_one_minus_alphas_cumprod_t * model_output
            )

        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_steps: Optional[int] = None,
        method: Literal["ddim", "ddpm"] = "ddim",
        eta: Optional[float] = None,
        temperature: float = 1.0,
        noise: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate images using diffusion sampling.
        
        This is the main generation interface. It handles all sampling logic
        and delegates to the model's forward pass.
        
        Args:
            batch_size: Number of images to generate
            num_steps: Number of denoising steps (None = use default)
            method: Sampling method ("ddim" or "ddpm")
            eta: DDIM stochasticity (0=deterministic, 1=DDPM-like, None=use default)
            temperature: Noise temperature for sampling
            noise: Initial noise tensor (None = sample random noise)
            device: Device to generate on (None = use model device)
            return_intermediates: If True, return all intermediate steps
        
        Returns:
            Generated images [B, C, H, W] or list of intermediates if requested
        """
        # Set defaults
        if num_steps is None:
            num_steps = self.default_num_steps
        if eta is None:
            eta = self.default_eta if method == "ddim" else 1.0
        if device is None:
            device = next(self.parameters()).device if isinstance(self, nn.Module) else torch.device("cpu")

        # Validate method
        if method == "ddpm":
            num_steps = self.num_timesteps
            eta = 1.0
        elif method != "ddim":
            raise ValueError(f"Unknown sampling method: {method}")

        # Generate using DDIM (which generalizes DDPM when eta=1.0)
        return self._sample_ddim(
            batch_size=batch_size,
            num_steps=num_steps,
            eta=eta,
            temperature=temperature,
            noise=noise,
            device=device,
            return_intermediates=return_intermediates,
        )

    @torch.no_grad()
    def _sample_ddim(
        self,
        batch_size: int,
        num_steps: int,
        eta: float,
        temperature: float,
        noise: Optional[torch.Tensor],
        device: torch.device,
        return_intermediates: bool,
    ) -> torch.Tensor:
        """
        Internal DDIM sampling implementation.
        
        Args:
            batch_size: Number of samples to generate
            num_steps: Number of denoising steps
            eta: DDIM stochasticity parameter
            temperature: Noise temperature
            noise: Optional initial noise
            device: Device to sample on
            return_intermediates: Whether to return all intermediate steps
        
        Returns:
            Generated images or list of intermediates
        """
        # Set model to eval mode if it's a Module
        was_training = False
        if isinstance(self, nn.Module):
            was_training = self.training
            self.eval()

        # Create timestep schedule for DDIM
        c = self.num_timesteps // num_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c, device=device)
        ddim_timesteps_prev = torch.cat(
            [torch.tensor([-1], device=device), ddim_timesteps[:-1]]
        )

        # Start from noise
        if noise is None:
            x = torch.randn(
                batch_size,
                self.in_channels,
                self.image_size,
                self.image_size,
                device=device,
            )
            x = x * temperature
        else:
            x = noise.to(device)

        # Store intermediates if requested
        intermediates = [x] if return_intermediates else None

        # Reverse diffusion process
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            t_prev = (
                ddim_timesteps_prev[i]
                if i > 0
                else torch.tensor(-1, device=device)
            )

            # Expand timestep to batch
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Model prediction (calls the model's forward method)
            model_output = self.forward(x, t_batch)

            # Predict x0
            pred_x0 = self.predict_x0_from_output(x, t_batch, model_output)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            if i > 0:
                # Get alpha values
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = (
                    self.alphas_cumprod[t_prev]
                    if t_prev >= 0
                    else torch.tensor(1.0, device=device)
                )

                # Compute variance
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
                )

                # Predict noise
                if self.prediction_type == "epsilon":
                    pred_noise = model_output
                else:
                    # Reconstruct noise from x0 prediction
                    pred_noise = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(
                        1 - alpha_t
                    )

                # Compute x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise
                noise_term = torch.randn_like(x) * temperature if sigma_t > 0 else 0
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise_term
            else:
                # Final step
                x = pred_x0

            if return_intermediates:
                intermediates.append(x)

        # Restore training mode if needed
        if isinstance(self, nn.Module) and was_training:
            self.train()

        return intermediates if return_intermediates else x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Must be implemented by the inheriting class.
        
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]
        
        Returns:
            Model prediction (epsilon, x0, or v depending on prediction_type)
        """
        raise NotImplementedError(
            "Model must implement forward(x, t) method for denoising"
        )
