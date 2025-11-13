from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional, Union, Tuple

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
    Alternative version with even cleaner API.
    
    In this version, schedulers are simpler and the mixin handles more logic.
    """

    task: ClassVar[str] = "image_generation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduler = None

    @property
    def scheduler(self):
        """Get the current scheduler."""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        """Set the scheduler for sampling."""
        self._scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        shape: Optional[Union[Tuple[int, ...], torch.Size]] = None,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Sample from the diffusion model.
        
        Args:
            batch_size: Number of samples to generate (used if shape not provided)
            shape: Complete shape (B, C, H, W) for samples
            num_steps: Number of denoising steps
            device: Device to generate samples on
            generator: Random generator for reproducibility
            return_intermediate: If True, return intermediate samples
            
        Returns:
            Generated samples, or (samples, intermediates) if return_intermediate=True
        """
        self.eval()
        
        # Setup
        device = device or next(self.parameters()).device
        if self.scheduler is None:
            raise ValueError("No scheduler set. Use model.scheduler = scheduler")
        
        # Determine shape
        if shape is None:
            if batch_size is None:
                raise ValueError("Must provide either 'shape' or 'batch_size'")
            shape = self._infer_shape(batch_size)
        
        # Initialize
        sample = torch.randn(shape, device=device, generator=generator)
        if hasattr(self.scheduler, 'scale_noise'):
            sample = self.scheduler.scale_noise(sample)
        
        intermediates = [] if return_intermediate else None
        
        # Denoising loop
        timesteps = self.scheduler.get_timesteps(num_steps, device=device)
        for i, t in enumerate(timesteps):
            # Prepare timestep tensor
            t_tensor = self._prepare_timestep(t, shape[0], device)
            
            # Model forward
            if hasattr(self.scheduler, 'scale_model_input'):
                model_input = self.scheduler.scale_model_input(sample, t)
            else:
                model_input = sample
            
            model_output = self.forward(model_input, t_tensor)
            
            # Scheduler step
            sample = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=sample,
                return_dict=False,
            )
            
            if return_intermediate:
                intermediates.append(sample.clone())
        
        # Post-process
        if hasattr(self.scheduler, 'post_process'):
            sample = self.scheduler.post_process(sample)
        
        if return_intermediate:
            return sample, intermediates
        return sample

    def _infer_shape(self, batch_size: int) -> Tuple[int, ...]:
        """Infer sampling shape from batch_size. Override in subclasses."""
        raise NotImplementedError("Implement _infer_shape() or pass 'shape' argument")

    def _prepare_timestep(
        self, t: Union[int, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Convert timestep to appropriate tensor format."""
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(batch_size)
            return t.to(device)
        else:
            return torch.full((batch_size,), t, device=device, dtype=torch.long)
