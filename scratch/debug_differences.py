"""
Debug script to identify differences between custom and HF BitNet implementations
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from model import BitNetForCausalLM, BitNetConfig
from validate import create_config_from_hf


def compare_module_outputs(custom_module, hf_module, inputs, module_name=""):
    """Compare outputs of two modules"""
    custom_module.eval()
    hf_module.eval()
    
    with torch.no_grad():
        if isinstance(inputs, dict):
            custom_out = custom_module(**inputs)
            hf_out = hf_module(**inputs)
        else:
            custom_out = custom_module(inputs)
            hf_out = hf_module(inputs)
        
        # Handle different output types
        if isinstance(custom_out, dict) and isinstance(hf_out, tuple):
            custom_tensor = custom_out.get('logits') or custom_out.get('hidden_states') or list(custom_out.values())[0]
            hf_tensor = hf_out[0]
        elif isinstance(custom_out, tuple) and isinstance(hf_out, tuple):
            custom_tensor = custom_out[0]
            hf_tensor = hf_out[0]
        elif isinstance(custom_out, torch.Tensor) and isinstance(hf_out, torch.Tensor):
            custom_tensor = custom_out
            hf_tensor = hf_out
        else:
            custom_tensor = custom_out
            hf_tensor = hf_out
        
        diff = (custom_tensor - hf_tensor).abs()
        print(f"\n{module_name}:")
        print(f"  Max diff: {diff.max().item():.6e}")
        print(f"  Mean diff: {diff.mean().item():.6e}")
        if hf_tensor.abs().max() > 0:
            print(f"  Relative max: {(diff.max() / hf_tensor.abs().max()).item():.6e}")
        
        return custom_tensor, hf_tensor, diff


def debug_attention(custom_attn, hf_attn, hidden_states, position_embeddings, attention_mask):
    """Debug attention layer step by step"""
    print("\n" + "="*60)
    print("DEBUGGING ATTENTION LAYER")
    print("="*60)
    
    custom_attn.eval()
    hf_attn.eval()
    
    batch_size, seq_length, _ = hidden_states.shape
    
    with torch.no_grad():
        # Compare Q, K, V projections
        custom_q = custom_attn.q_proj(hidden_states)
        hf_q = hf_attn.q_proj(hidden_states)
        print("\nQ projection:")
        print(f"  Max diff: {(custom_q - hf_q).abs().max().item():.6e}")
        
        custom_k = custom_attn.k_proj(hidden_states)
        hf_k = hf_attn.k_proj(hidden_states)
        print("K projection:")
        print(f"  Max diff: {(custom_k - hf_k).abs().max().item():.6e}")
        
        custom_v = custom_attn.v_proj(hidden_states)
        hf_v = hf_attn.v_proj(hidden_states)
        print("V projection:")
        print(f"  Max diff: {(custom_v - hf_v).abs().max().item():.6e}")
        
        # Reshape
        custom_q = custom_q.view(batch_size, seq_length, custom_attn.num_heads, custom_attn.head_dim).transpose(1, 2)
        hf_q = hf_q.view(batch_size, seq_length, hf_attn.config.num_attention_heads, hf_attn.head_dim).transpose(1, 2)
        
        custom_k = custom_k.view(batch_size, seq_length, custom_attn.num_key_value_heads, custom_attn.head_dim).transpose(1, 2)
        hf_k = hf_k.view(batch_size, seq_length, hf_attn.config.num_key_value_heads, hf_attn.head_dim).transpose(1, 2)
        
        print("\nAfter reshape:")
        print(f"  Q diff: {(custom_q - hf_q).abs().max().item():.6e}")
        print(f"  K diff: {(custom_k - hf_k).abs().max().item():.6e}")
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        from model import apply_rotary_pos_emb
        custom_q_rot, custom_k_rot = apply_rotary_pos_emb(custom_q, custom_k, cos, sin)
        
        # HF also uses apply_rotary_pos_emb - imported from modeling file
        from transformers.models.bitnet.modeling_bitnet import apply_rotary_pos_emb as hf_apply_rotary
        hf_q_rot, hf_k_rot = hf_apply_rotary(hf_q, hf_k, cos, sin)
        
        print("\nAfter RoPE:")
        print(f"  Q diff: {(custom_q_rot - hf_q_rot).abs().max().item():.6e}")
        print(f"  K diff: {(custom_k_rot - hf_k_rot).abs().max().item():.6e}")
        
        # Full forward pass
        custom_out = custom_attn(hidden_states, position_embeddings, attention_mask)
        hf_out = hf_attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)
        
        if isinstance(custom_out, tuple):
            custom_tensor = custom_out[0]
        else:
            custom_tensor = custom_out
            
        if isinstance(hf_out, tuple):
            hf_tensor = hf_out[0]
        else:
            hf_tensor = hf_out
        
        print("\nFinal attention output:")
        print(f"  Max diff: {(custom_tensor - hf_tensor).abs().max().item():.6e}")
        print(f"  Mean diff: {(custom_tensor - hf_tensor).abs().mean().item():.6e}")


def debug_mlp(custom_mlp, hf_mlp, hidden_states):
    """Debug MLP/FFN layer step by step"""
    print("\n" + "="*60)
    print("DEBUGGING MLP LAYER")
    print("="*60)
    
    custom_mlp.eval()
    hf_mlp.eval()
    
    with torch.no_grad():
        # Gate projection
        custom_gate = custom_mlp.gate_proj(hidden_states)
        hf_gate = hf_mlp.gate_proj(hidden_states)
        print("\nGate projection:")
        print(f"  Max diff: {(custom_gate - hf_gate).abs().max().item():.6e}")
        
        # Up projection
        custom_up = custom_mlp.up_proj(hidden_states)
        hf_up = hf_mlp.up_proj(hidden_states)
        print("Up projection:")
        print(f"  Max diff: {(custom_up - hf_up).abs().max().item():.6e}")
        
        # Activation - check what activation function is being used
        print(f"\nCustom activation: {custom_mlp.hidden_act}")
        print(f"HF activation: {hf_mlp.config.hidden_act}")
        
        # Apply activation
        if custom_mlp.hidden_act == "relu2":
            custom_gate_act = F.relu(custom_gate).pow(2)
        else:
            custom_gate_act = F.silu(custom_gate)
        
        # HF uses ACT2FN
        hf_gate_act = hf_mlp.act_fn(hf_gate)
        
        print("After activation:")
        print(f"  Max diff: {(custom_gate_act - hf_gate_act).abs().max().item():.6e}")
        
        # Element-wise multiply
        custom_mult = custom_gate_act * custom_up
        hf_mult = hf_gate_act * hf_up
        print("After gate * up:")
        print(f"  Max diff: {(custom_mult - hf_mult).abs().max().item():.6e}")
        
        # Sub-normalization
        custom_normed = custom_mlp.ffn_sub_norm(custom_mult)
        hf_normed = hf_mlp.ffn_sub_norm(hf_mult)
        print("After sub-norm:")
        print(f"  Max diff: {(custom_normed - hf_normed).abs().max().item():.6e}")
        
        # Down projection
        custom_down = custom_mlp.down_proj(custom_normed)
        hf_down = hf_mlp.down_proj(hf_normed)
        print("After down projection:")
        print(f"  Max diff: {(custom_down - hf_down).abs().max().item():.6e}")
        
        # Full forward
        custom_out = custom_mlp(hidden_states)
        hf_out = hf_mlp(hidden_states)
        print("\nFinal MLP output:")
        print(f"  Max diff: {(custom_out - hf_out).abs().max().item():.6e}")


def debug_rotary_embeddings(custom_rope, hf_rope, hidden_states, position_ids):
    """Debug rotary embeddings"""
    print("\n" + "="*60)
    print("DEBUGGING ROTARY EMBEDDINGS")
    print("="*60)
    
    custom_rope.eval()
    hf_rope.eval()
    
    with torch.no_grad():
        custom_cos, custom_sin = custom_rope(hidden_states, position_ids)
        hf_cos, hf_sin = hf_rope(hidden_states, position_ids)
        
        print("\nCos embeddings:")
        print(f"  Max diff: {(custom_cos - hf_cos).abs().max().item():.6e}")
        print(f"  Mean diff: {(custom_cos - hf_cos).abs().mean().item():.6e}")
        
        print("Sin embeddings:")
        print(f"  Max diff: {(custom_sin - hf_sin).abs().max().item():.6e}")
        print(f"  Mean diff: {(custom_sin - hf_sin).abs().mean().item():.6e}")
        
        # Check inv_freq
        print("\ninv_freq comparison:")
        print(f"  Custom inv_freq shape: {custom_rope.inv_freq.shape}")
        print(f"  HF inv_freq shape: {hf_rope.inv_freq.shape}")
        print(f"  Max diff: {(custom_rope.inv_freq - hf_rope.inv_freq).abs().max().item():.6e}")


def main():
    model_path = "microsoft/bitnet-b1.58-2B-4T-bf16"
    
    print("Loading models...")
    hf_config = AutoConfig.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    
    custom_config = create_config_from_hf(hf_config)
    custom_model = BitNetForCausalLM(custom_config)
    custom_model.load_state_dict(hf_model.state_dict(), strict=False)
    
    # Create test input
    test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    batch_size, seq_length = test_input.shape
    position_ids = torch.arange(seq_length).unsqueeze(0)
    
    print("\n" + "="*60)
    print("COMPARING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Get embeddings
    with torch.no_grad():
        custom_embeds = custom_model.model.embed_tokens(test_input)
        hf_embeds = hf_model.model.embed_tokens(test_input)
        
        print("\nToken embeddings:")
        print(f"  Max diff: {(custom_embeds - hf_embeds).abs().max().item():.6e}")
        print(f"  Are they the same? {torch.allclose(custom_embeds, hf_embeds, atol=1e-6)}")
    
    # Compare rotary embeddings
    debug_rotary_embeddings(
        custom_model.model.rotary_emb,
        hf_model.model.rotary_emb,
        custom_embeds,
        position_ids
    )
    
    # Get position embeddings for attention
    with torch.no_grad():
        position_embeddings = custom_model.model.rotary_emb(custom_embeds, position_ids)
        
        # Create attention mask
        attention_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf')),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length)
    
    # Debug first decoder layer
    print("\n" + "="*60)
    print("DEBUGGING FIRST DECODER LAYER")
    print("="*60)
    
    custom_layer = custom_model.model.layers[0]
    hf_layer = hf_model.model.layers[0]
    
    # Debug attention
    debug_attention(
        custom_layer.self_attn,
        hf_layer.self_attn,
        custom_embeds,
        position_embeddings,
        attention_mask
    )
    
    # Debug MLP
    debug_mlp(
        custom_layer.mlp,
        hf_layer.mlp,
        custom_embeds
    )
    
    # Compare full layer forward
    print("\n" + "="*60)
    print("FULL LAYER FORWARD")
    print("="*60)
    
    with torch.no_grad():
        custom_layer_out = custom_layer(
            custom_embeds,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        
        hf_layer_out = hf_layer(
            hf_embeds,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        
        print("\nLayer output:")
        print(f"  Max diff: {(custom_layer_out - hf_layer_out).abs().max().item():.6e}")
        print(f"  Mean diff: {(custom_layer_out - hf_layer_out).abs().mean().item():.6e}")


if __name__ == "__main__":
    main()