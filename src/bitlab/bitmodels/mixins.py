from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional

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
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self.config, "eos_token_id", None
        )

        generated = input_ids
        past_key_values: Any = None
        cur_len = input_ids.shape[1]

        for _ in range(max_length - cur_len):
            model_inputs = self.prepare_inputs_for_generation(generated, past_key_values=past_key_values)
            outputs = self.forward(**model_inputs, use_cache=use_cache)

            if use_cache:
                past_key_values = outputs.get("past_key_values", past_key_values)

            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                kth_values = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits = logits.masked_fill(logits < kth_values, float("-inf"))

            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
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
        return_logits: bool = False,
        return_probabilities: bool = False,
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
    """Helper methods for unconditional diffusion/image generators."""

    task: ClassVar[str] = ModelTask.IMAGE_GENERATION.value

    @torch.no_grad()
    def sample(
        self,
        shape: Iterable[int],
        *,
        timesteps: int = 50,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        **scheduler_kwargs: Any,
    ) -> torch.Tensor:
        self.eval()
        device = device or next(self.parameters()).device
        if scheduler is None:
            raise ValueError(
                "A scheduler implementing init_noise/scale_model_input/step/finalize is required."
            )

        sample = scheduler.init_noise(shape, device=device, **scheduler_kwargs)
        for t in scheduler.timesteps(timesteps):
            model_input = scheduler.scale_model_input(sample, t)
            model_output = self.forward(model_input, t)
            sample = scheduler.step(model_output, t, sample)
        return scheduler.finalize(sample)


