"""
Simplified script to load BitNet weights from HuggingFace and generate text.
"""

import argparse
from contextlib import nullcontext
from typing import Optional
import torch
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file

from model import BitNetConfig, BitNetForCausalLM


def create_config_from_hf(hf_config: AutoConfig) -> BitNetConfig:
    """Convert HuggingFace config to BitNetConfig."""
    return BitNetConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        hidden_act=hf_config.hidden_act,
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        pad_token_id=hf_config.pad_token_id,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        attention_bias=getattr(hf_config, "attention_bias", False),
        quant_type=getattr(hf_config, "quant_type", "ai8ptk_wpt"),
    )


def _resolve_device(device: Optional[str]) -> torch.device:
    """Return a concrete torch.device with sensible defaults."""
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: Optional[str], device: torch.device) -> torch.dtype:
    """Map a user-provided dtype string to a torch dtype with device-aware defaults."""
    if dtype is None:
        if device.type == "cuda":
            if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    dtype = dtype.lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Choose from: {', '.join(sorted(mapping.keys()))}"
        )
    return mapping[dtype]


def load_model(
    hf_model_path: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    compile_model: bool = False,
):
    """Load tokenizer and custom BitNet model with HF weights."""
    torch_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype, torch_device)

    print(f"Using device: {torch_device}")
    print(f"Requested dtype: {torch_dtype}")

    print(f"Loading tokenizer from {hf_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading config from {hf_model_path}...")
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=False)
    
    print("Creating custom BitNet model...")
    config = create_config_from_hf(hf_config)
    model = BitNetForCausalLM(config)
    
    print("Loading weights from HuggingFace...")
    # Try to load from safetensors file
    try:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(repo_id=hf_model_path, filename="model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load weights: {e}")
        print("Model initialized with random weights")
    
    model = model.to(device=torch_device, dtype=torch_dtype)
    model.eval()

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Compiled model with torch.compile for faster inference")
        except Exception as exc:
            print(f"⚠ torch.compile failed ({exc}); continuing without compilation")

    return tokenizer, model, torch_device, torch_dtype


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: torch.device, model_dtype: torch.dtype,
             max_new_tokens: int = 50, temperature: float = 0.8,
             top_k: int = 50, top_p: float = 0.9):
    """Generate text from prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    autocast_dtype = None
    if device.type in {"cuda", "mps"}:
        autocast_dtype = model_dtype if model_dtype in {torch.float16, torch.bfloat16} else torch.float16

    context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    with context:
        generated = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            do_sample=True,
        )
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text with BitNet")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T-bf16",
                        help="HuggingFace model path")
    parser.add_argument("--prompt", default="Write a haiku about AI:",
                        help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--dtype", default=None, help="Model dtype (fp32, bf16, fp16)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model")
    
    args = parser.parse_args()
    
    tokenizer, model, device, model_dtype = load_model(
        args.model,
        args.device,
        args.dtype,
        args.compile,
    )
    
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    output = generate(
        model, tokenizer, args.prompt, device, model_dtype,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print(output)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()