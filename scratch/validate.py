"""
Helper script to load BitNet weights from HuggingFace and validate outputs
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


def load_and_validate_model(hf_model_path: str, test_input_ids: torch.LongTensor = None):
    """
    Load a BitNet model from HuggingFace and validate outputs match
    
    Args:
        hf_model_path: Path to HuggingFace model (e.g., "microsoft/bitnet-b1.58-2B-4T")
        test_input_ids: Optional input tensor for validation
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"Loading HuggingFace model from {hf_model_path}...")
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32)
    
    print("Creating custom BitNet model...")
    config = create_config_from_hf(hf_config)
    custom_model = BitNetForCausalLM(config)
    
    print("Loading weights into custom model...")
    # Direct load_state_dict - keys match exactly!
    missing_keys, unexpected_keys = custom_model.load_state_dict(hf_model.state_dict(), strict=False)
    
    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys}")
    
    if not missing_keys and not unexpected_keys:
        print("✓ All weights loaded successfully! Keys matched perfectly.")
    else:
        print("⚠ Some keys didn't match - weights partially loaded")
    
    # Validate outputs if test input provided
    if test_input_ids is not None:
        print("\nValidating outputs...")
        hf_model.eval()
        custom_model.eval()
        
        with torch.no_grad():
            # HF model output
            hf_outputs = hf_model(test_input_ids)
            hf_logits = hf_outputs.logits
            
            # Custom model output
            custom_outputs = custom_model(test_input_ids)
            custom_logits = custom_outputs['logits']
            
            # Compare outputs
            max_diff = (hf_logits - custom_logits).abs().max().item()
            mean_diff = (hf_logits - custom_logits).abs().mean().item()
            
            print(f"Max difference in logits: {max_diff}")
            print(f"Mean difference in logits: {mean_diff}")
            
            # Check if outputs are close
            if torch.allclose(hf_logits, custom_logits, atol=1e-4):
                print("✓ Outputs match! Model loaded correctly.")
            else:
                print("⚠ Outputs don't match exactly, but this might be due to numerical precision.")
                print(f"  Relative error: {(max_diff / hf_logits.abs().max().item()):.6f}")
    
    return custom_model, hf_model


def compare_single_forward_pass(custom_model, hf_model, input_ids):
    """Compare a single forward pass between custom and HF models"""
    custom_model.eval()
    hf_model.eval()
    
    with torch.no_grad():
        # Custom model
        custom_outputs = custom_model(input_ids)
        custom_logits = custom_outputs['logits']
        custom_hidden = custom_outputs['hidden_states']
        
        # HF model
        hf_outputs = hf_model(input_ids, output_hidden_states=True)
        hf_logits = hf_outputs.logits
        hf_hidden = hf_outputs.hidden_states[-1]  # Last hidden state
        
        # Compare logits
        logits_diff = (custom_logits - hf_logits).abs()
        print(f"\nLogits comparison:")
        print(f"  Max diff: {logits_diff.max().item():.6e}")
        print(f"  Mean diff: {logits_diff.mean().item():.6e}")
        print(f"  Relative max diff: {(logits_diff.max() / hf_logits.abs().max()).item():.6e}")
        
        # Compare hidden states
        hidden_diff = (custom_hidden - hf_hidden).abs()
        print(f"\nHidden states comparison:")
        print(f"  Max diff: {hidden_diff.max().item():.6e}")
        print(f"  Mean diff: {hidden_diff.mean().item():.6e}")
        print(f"  Relative max diff: {(hidden_diff.max() / hf_hidden.abs().max()).item():.6e}")
        
        # Check predictions
        custom_preds = custom_logits.argmax(dim=-1)
        hf_preds = hf_logits.argmax(dim=-1)
        matches = (custom_preds == hf_preds).float().mean().item()
        print(f"\nPrediction agreement: {matches*100:.2f}%")
        
    return {
        'logits_max_diff': logits_diff.max().item(),
        'logits_mean_diff': logits_diff.mean().item(),
        'hidden_max_diff': hidden_diff.max().item(),
        'hidden_mean_diff': hidden_diff.mean().item(),
        'prediction_agreement': matches,
    }


if __name__ == "__main__":
    # Example: Load and validate a BitNet model
    model_path = "microsoft/bitnet-b1.58-2B-4T-bf16"  # Replace with actual model path
    
    # Create test input
    test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    # Load and validate
    custom_model, hf_model = load_and_validate_model(model_path, test_input)
    
    # Detailed comparison
    print("\n" + "="*60)
    print("Detailed comparison of forward pass:")
    print("="*60)
    results = compare_single_forward_pass(custom_model, hf_model, test_input)
    
    # Save custom model
    print("\nSaving custom model...")
    torch.save(custom_model.state_dict(), "bitnet_custom_weights.pt")
    print("Model saved to bitnet_custom_weights.pt")