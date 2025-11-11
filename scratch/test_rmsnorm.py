import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - EXACT match to original"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # Match original attribute name

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_rmsnorm_dict(sd): 
    rmsnorm_dict = {}
    rmsnorm_dict["weight"] = sd["model.layers.24.post_attention_layernorm.weight"]
    return rmsnorm_dict


def collect_activations(model, layer_name, tokenizer, test_input):
    """Collect activations from a specific layer in the HuggingFace model"""
    activations = {}
    
    def hook_fn(module, input, output):
        activations['input'] = input[0].detach().clone()
        activations['output'] = output.detach().clone()
    
    # Register hook on the target layer
    target_layer = dict(model.named_modules())[layer_name]
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Run forward pass
    inputs = tokenizer(test_input, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    
    # Remove hook
    handle.remove()
    
    return activations


def test_equivalence(custom_module, hf_model, layer_name, tokenizer, test_inputs, rtol=1e-5, atol=1e-5):
    """Test if custom module produces equivalent outputs to HF model layer"""
    print(f"\n{'='*60}")
    print(f"Testing equivalence for: {layer_name}")
    print(f"{'='*60}\n")
    
    all_passed = True
    
    for i, test_input in enumerate(test_inputs):
        print(f"Test case {i+1}: '{test_input[:50]}...'")
        
        # Collect activations from HF model
        activations = collect_activations(hf_model, layer_name, tokenizer, test_input)
        hf_input = activations['input']
        hf_output = activations['output']
        
        # Run through custom module
        with torch.no_grad():
            custom_output = custom_module(hf_input)
        
        # Compare outputs
        is_close = torch.allclose(custom_output, hf_output, rtol=rtol, atol=atol)
        max_diff = (custom_output - hf_output).abs().max().item()
        mean_diff = (custom_output - hf_output).abs().mean().item()
        
        print(f"  Input shape: {hf_input.shape}")
        print(f"  Output shape: {custom_output.shape}")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Outputs match: {'✓ PASS' if is_close else '✗ FAIL'}")
        print()
        
        if not is_close:
            all_passed = False
            # Additional debugging info
            print(f"  HF output stats - mean: {hf_output.mean():.6f}, std: {hf_output.std():.6f}")
            print(f"  Custom output stats - mean: {custom_output.mean():.6f}, std: {custom_output.std():.6f}")
            print()
    
    print(f"{'='*60}")
    print(f"Overall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"{'='*60}\n")
    
    return all_passed


if __name__ == "__main__": 
    print("Loading BitNet model...")
    hf_model = AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    hf_model.eval()
    
    # Get state dict and extract RMSNorm weights
    sd = hf_model.state_dict()
    rmsnorm_dict = get_rmsnorm_dict(sd)
    
    print("\nRMSNorm parameters:")
    for key, value in rmsnorm_dict.items():
        print(f"  {key}: {value.shape}")
    
    # Create custom RMSNorm module and load weights
    rmsnorm = RMSNorm(rmsnorm_dict["weight"].shape[0])
    rmsnorm.load_state_dict(rmsnorm_dict)
    rmsnorm.eval()
    
    # Test inputs - use varied examples
    test_inputs = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning are transforming the world.",
        "A",  # Short input
        "This is a longer sentence with more tokens to test the normalization behavior across different sequence lengths and contexts."
    ]
    
    # Layer to test
    layer_name = "model.layers.24.post_attention_layernorm"
    
    # Run equivalence test
    passed = test_equivalence(
        custom_module=rmsnorm,
        hf_model=hf_model,
        layer_name=layer_name,
        tokenizer=tokenizer,
        test_inputs=test_inputs,
        rtol=1e-5,
        atol=1e-5
    )
    
    if passed:
        print("🎉 Equivalence check successful! Your RMSNorm matches the BitNet implementation.")
    else:
        print("⚠️  Equivalence check failed. There are differences between implementations.")