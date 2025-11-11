"""
Helper script to load BitNet weights from HuggingFace and validate outputs,
with tokenizer debug printing (encode / decode / token list).
"""
import torch
from model import BitNetForCausalLM, BitNetConfig, load_weights_from_hf


def create_config_from_hf(hf_config):
    """Convert HuggingFace config to our BitNetConfig"""
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
        rope_theta=hf_config.rope_theta if hasattr(hf_config, 'rope_theta') else 10000.0,
        attention_bias=hf_config.attention_bias if hasattr(hf_config, 'attention_bias') else False,
        quant_type=getattr(hf_config, 'quant_type', 'ai8ptk_wpt'),
    )


def show_tokenization(tokenizer, text: str, add_bos=False, add_eos=False, return_tensors="pt"):
    """
    Encode a string with the tokenizer and print a readable debug view:
      - original text
      - token strings (as tokenizer uses)
      - token ids
      - decoded text from ids (round-trip)
      - attention mask
    Returns the model-ready dict (input_ids, attention_mask) as torch tensors if requested.
    """
    # optionally add bos/eos tokens if tokenizer/config expects them
    if add_bos and tokenizer.bos_token is not None:
        text_for_encode = tokenizer.bos_token + text
    else:
        text_for_encode = text
    if add_eos and tokenizer.eos_token is not None:
        text_for_encode = text_for_encode + tokenizer.eos_token

    enc = tokenizer(text_for_encode, return_tensors=return_tensors)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    # Convert to plain Python lists for printing
    ids_list = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids_list, skip_special_tokens=False)
    decoded = tokenizer.decode(ids_list, skip_special_tokens=False)

    print("\n" + "-" * 60)
    print("PROMPT DEBUG")
    print("-" * 60)
    print("Original string:")
    print(f"  {text!r}")
    print("\nToken strings (tokenizer.convert_ids_to_tokens):")
    print(f"  {tokens}")
    print("\nToken ids:")
    print(f"  {ids_list}")
    if attention_mask is not None:
        print("\nAttention mask:")
        print(f"  {attention_mask[0].tolist()}")
    print("\nRound-trip decode (tokenizer.decode):")
    print(f"  {decoded!r}")
    print("-" * 60 + "\n")

    return enc


def load_and_validate_model(hf_model_path: str, test_prompts: list = None):
    """
    Load a BitNet model from HuggingFace and validate outputs match.
    Also demonstrates tokenization for provided prompts.

    Args:
        hf_model_path: Path to HuggingFace model (e.g., "microsoft/bitnet-b1.58-2B-4T")
        test_prompts: Optional list of prompt strings for token/debug + validation
    """
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"Loading HuggingFace model from {hf_model_path}...")
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32)

    # Load tokenizer for encoding/decoding
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=True)

    print("Creating custom BitNet model...")
    config = create_config_from_hf(hf_config)
    custom_model = BitNetForCausalLM(config)

    print("Loading weights into custom model...")
    # Direct load_state_dict - keys may match. We keep strict=False to allow partials.
    missing_keys, unexpected_keys = custom_model.load_state_dict(hf_model.state_dict(), strict=False)

    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys}")

    if not missing_keys and not unexpected_keys:
        print("✓ All weights loaded successfully! Keys matched perfectly.")
    else:
        print("⚠ Some keys didn't match - weights partially loaded")

    # If the user passed prompts, show tokenization and validate outputs per prompt
    if test_prompts:
        hf_model.eval()
        custom_model.eval()

        for prompt in test_prompts:
            enc = show_tokenization(tokenizer, prompt)

            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)

            # Move to same device as models (CPU by default here)
            input_ids = input_ids.long()
            if attention_mask is not None:
                attention_mask = attention_mask.long()

            with torch.no_grad():
                # HF model output (ensure we pass attention_mask if present)
                hf_outputs = hf_model(input_ids, attention_mask=attention_mask) if attention_mask is not None else hf_model(input_ids)
                hf_logits = hf_outputs.logits

                # Custom model output (assumes your custom model accepts attention_mask if needed)
                # If your custom_model signature differs, adapt this call accordingly.
                try:
                    custom_outputs = custom_model(input_ids, attention_mask=attention_mask)
                except TypeError:
                    # fallback if custom model expects only input_ids
                    custom_outputs = custom_model(input_ids)

                # custom_outputs expected to be dict-like in your code
                custom_logits = custom_outputs['logits'] if isinstance(custom_outputs, dict) else custom_outputs.logits

                # Compare outputs
                max_diff = (hf_logits - custom_logits).abs().max().item()
                mean_diff = (hf_logits - custom_logits).abs().mean().item()

                print(f"Prompt: {prompt!r}")
                print(f"  Max difference in logits: {max_diff:.6e}")
                print(f"  Mean difference in logits: {mean_diff:.6e}")

                if torch.allclose(hf_logits, custom_logits, atol=1e-4):
                    print("  ✓ Outputs match! Model loaded correctly (within tolerance).")
                else:
                    print("  ⚠ Outputs don't match exactly; this may be numerical or implementation differences.")
                    hf_max = hf_logits.abs().max().item()
                    rel_err = (max_diff / hf_max) if hf_max != 0 else float('inf')
                    print(f"  Relative max error: {rel_err:.6e}")

    return custom_model, hf_model, tokenizer


def compare_single_forward_pass(custom_model, hf_model, input_ids, attention_mask=None):
    """Compare a single forward pass between custom and HF models"""
    custom_model.eval()
    hf_model.eval()

    with torch.no_grad():
        # Custom model
        try:
            custom_outputs = custom_model(input_ids, attention_mask=attention_mask)
        except TypeError:
            custom_outputs = custom_model(input_ids)

        custom_logits = custom_outputs['logits'] if isinstance(custom_outputs, dict) else custom_outputs.logits
        custom_hidden = custom_outputs.get('hidden_states', None)
        if custom_hidden is not None:
            # if custom model returns tuple/list, take last
            if isinstance(custom_hidden, (list, tuple)):
                custom_hidden = custom_hidden[-1]

        # HF model
        hf_outputs = hf_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hf_logits = hf_outputs.logits
        hf_hidden = hf_outputs.hidden_states[-1]  # Last hidden state

        # Compare logits
        logits_diff = (custom_logits - hf_logits).abs()
        print(f"\nLogits comparison:")
        print(f"  Max diff: {logits_diff.max().item():.6e}")
        print(f"  Mean diff: {logits_diff.mean().item():.6e}")
        print(f"  Relative max diff: {(logits_diff.max() / hf_logits.abs().max()).item():.6e}")

        # Compare hidden states (if custom returned them)
        if custom_hidden is not None:
            hidden_diff = (custom_hidden - hf_hidden).abs()
            print(f"\nHidden states comparison:")
            print(f"  Max diff: {hidden_diff.max().item():.6e}")
            print(f"  Mean diff: {hidden_diff.mean().item():.6e}")
            print(f"  Relative max diff: {(hidden_diff.max() / hf_hidden.abs().max()).item():.6e}")
            hidden_max = hidden_diff.max().item()
            hidden_mean = hidden_diff.mean().item()
        else:
            hidden_max = hidden_mean = None
            print("\nCustom model did not return hidden states; skipping hidden comparison.")

        # Check predictions
        custom_preds = custom_logits.argmax(dim=-1)
        hf_preds = hf_logits.argmax(dim=-1)
        matches = (custom_preds == hf_preds).float().mean().item()
        print(f"\nPrediction agreement: {matches*100:.2f}%")

    return {
        'logits_max_diff': logits_diff.max().item(),
        'logits_mean_diff': logits_diff.mean().item(),
        'hidden_max_diff': hidden_max,
        'hidden_mean_diff': hidden_mean,
        'prediction_agreement': matches,
    }


if __name__ == "__main__":
    # Example: Load and validate a BitNet model
    model_path = "microsoft/bitnet-b1.58-2B-4T-bf16"  # Replace with actual model path

    # Example prompts to inspect (you can add more)
    prompts = [
        "Hello, my name is Ollie. What is the weather like today?",
        "Translate to French: The quick brown fox jumps over the lazy dog.",
        "Write a short poem about AI and winter."
    ]

    # Load models + tokenizer and show tokenizations / validate outputs
    custom_model, hf_model, tokenizer = load_and_validate_model(model_path, test_prompts=prompts)

    # If you want to run a detailed comparison on the first prompt:
    enc = tokenizer(prompts[0], return_tensors="pt")
    input_ids = enc["input_ids"].long()
    attention_mask = enc.get("attention_mask", None)
    print("\n" + "="*60)
    print("Detailed comparison of forward pass for first prompt:")
    print("="*60)
    results = compare_single_forward_pass(custom_model, hf_model, input_ids, attention_mask=attention_mask)

    # Save custom model
    print("\nSaving custom model...")
    torch.save(custom_model.state_dict(), "bitnet_custom_weights.pt")
    print("Model saved to bitnet_custom_weights.pt")