"""
Helper script to load BitNet weights from HuggingFace and generate sample texts.
"""

import argparse
import textwrap
from typing import Iterable, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model import BitNetConfig, BitNetForCausalLM


def create_config_from_hf(hf_config: AutoConfig) -> BitNetConfig:
    """Convert HuggingFace config to our BitNetConfig."""
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


def load_models_and_tokenizer(hf_model_path: str, torch_dtype: torch.dtype = torch.float32, device: str | None = None):
    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer '{hf_model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading HuggingFace config '{hf_model_path}' (no remote code)...")
    # Do NOT trust remote code for the config — repo's auto_map points at missing files
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=False)

    hf_model = None
    try:
        # attempt to load the HF model only if the user didn't ask to skip it
        print(f"Attempting to load HF reference model '{hf_model_path}' (may fail if repo is missing code)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,   # still needs remote code to instantiate the HF architecture
        ).to(resolved_device)
        hf_model.eval()
    except Exception as e:
        print("⚠ Could not load HuggingFace reference model (remote model code missing or incompatible).")
        print("  Details:", e)
        print("  Proceeding with only the local custom BitNet implementation. To avoid this message, run with --skip-reference.")
        hf_model = None

    print("Creating custom BitNet model...")
    custom_config = create_config_from_hf(hf_config)
    custom_model = BitNetForCausalLM(custom_config).to(resolved_device)

    if hf_model is not None:
        print("Loading weights into custom model from HF reference ...")
        missing_keys, unexpected_keys = custom_model.load_state_dict(hf_model.state_dict(), strict=False)
        # ... same checks as you had
    else:
        # You still need to load weights from the safetensors file yourself,
        # e.g. via `hf_model_path/model.safetensors` using safetensors or torch.load,
        # or use your conversion path to load the checkpoint into your custom model.
        print("⚠ HF reference model not available — you must load weights into `custom_model` yourself (safetensors/gguf/etc).")

    custom_model.eval()
    return tokenizer, custom_model, hf_model, resolved_device


def format_generation(title: str, prompt: str, completion: str) -> str:
    """Format a prompt/completion pair for pretty printing."""
    wrapped_prompt = textwrap.fill(prompt.strip(), width=88)
    wrapped_completion = textwrap.fill(completion.strip(), width=88)
    return (
        f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"
        f"Prompt:\n{wrapped_prompt}\n\nCompletion:\n{wrapped_completion}\n"
    )


@torch.no_grad()
def generate_with_custom(
    model: BitNetForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
) -> str:
    """Generate text using the custom BitNet implementation."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = model.generate(
        input_ids=input_ids,
        max_length=input_ids.shape[1] + max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def generate_with_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
) -> str:
    """Generate text using the HuggingFace reference implementation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_ids = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_samples(
    tokenizer: AutoTokenizer,
    custom_model: BitNetForCausalLM,
    hf_model: AutoModelForCausalLM | None,
    device: torch.device,
    prompts: Iterable[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
    compare: bool,
) -> None:
    """Generate completions for each prompt and print the results."""
    for idx, prompt in enumerate(prompts, start=1):
        print(f"\nPrompt {idx}: {prompt}")

        custom_completion = generate_with_custom(
            custom_model,
            tokenizer,
            prompt,
            device,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            do_sample,
        )
        print(
            format_generation(
                title="Custom BitNet Output",
                prompt=prompt,
                completion=custom_completion,
            )
        )

        if compare and hf_model is not None:
            hf_completion = generate_with_hf(
                hf_model,
                tokenizer,
                prompt,
                device,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                do_sample,
            )

        elif compare and hf_model is None:
            print("⚠ Skipping HuggingFace reference output (HF model unavailable).")
            print(
                format_generation(
                    title="HuggingFace BitNet Output",
                    prompt=prompt,
                    completion=hf_completion,
                )
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text samples with BitNet models."
    )
    parser.add_argument(
        "--hf-model",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model identifier or local path.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Write a haiku about low-bit neural networks.",
            "Explain why quantization is useful for large language models.",
            "Describe a future where efficient AI models power everyday devices.",
        ],
        help="One or more prompts to generate completions for.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Number of new tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower values make outputs more deterministic.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling value. Set to 0 to disable.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling value. Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Torch dtype to load the HuggingFace model with.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g., 'cuda', 'cpu', 'mps'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip HuggingFace reference generation and only use the custom model.",
    )
    return parser


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[name]


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    top_k = None if args.top_k <= 0 else args.top_k
    top_p = None if args.top_p >= 1.0 else args.top_p
    do_sample = not args.no_sample
    compare = not args.skip_reference

    tokenizer, custom_model, hf_model, device = load_models_and_tokenizer(
        hf_model_path=args.hf_model,
        torch_dtype=dtype,
        device=args.device,
    )

    if compare and hf_model is None:
        print("⚠ HuggingFace model was not loaded successfully; disabling reference comparison.")
        compare = False

    generate_samples(
        tokenizer=tokenizer,
        custom_model=custom_model,
        hf_model=hf_model,
        device=device,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        compare=compare,
    )


if __name__ == "__main__":
    main()

