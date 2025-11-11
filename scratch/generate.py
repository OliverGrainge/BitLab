"""
Simplified script to load BitNet weights from HuggingFace and generate text.
"""

import argparse
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


def load_model(hf_model_path: str, device: str = None):
    """Load tokenizer and custom BitNet model with HF weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading tokenizer from {hf_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading config from {hf_model_path}...")
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=False)
    
    print("Creating custom BitNet model...")
    config = create_config_from_hf(hf_config)
    model = BitNetForCausalLM(config).to(device)
    
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
    
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 50, 
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
    """Generate text from prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
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
    
    args = parser.parse_args()
    
    tokenizer, model, device = load_model(args.model, args.device)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    output = generate(
        model, tokenizer, args.prompt, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print(output)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()