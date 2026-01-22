import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.models import MODELS_REGISTRY 
from src.models.tokenizers import TOKENIZERS_REGISTRY 

def load_model(model_name, device=None): 
    """
    Load a model either from registry or from a checkpoint path.
    
    Args:
        model_name: Either a registry key or a path to a model checkpoint
        device: Device to load model on (auto-detected if None)
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if it's a registry model
    if model_name in MODELS_REGISTRY:
        model = MODELS_REGISTRY[model_name]()
    else:
        # Treat as a path (local or HuggingFace model ID)
        if os.path.exists(model_name) or "/" in model_name:
            print(f"Loading model from path: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(
                f"Model '{model_name}' not found in registry and not a valid path.\n"
                f"Available registry models: {list(MODELS_REGISTRY.keys())}"
            )
    
    model = model.to(device)
    model.eval()
    return model, device

def load_tokenizer(model_name): 
    """
    Load a tokenizer either from registry or from a checkpoint path.
    
    Args:
        model_name: Either a registry key or a path to a model checkpoint
    """
    # Check if it's a registry tokenizer
    if model_name in TOKENIZERS_REGISTRY:
        tokenizer = TOKENIZERS_REGISTRY[model_name]()
    else:
        # Treat as a path (local or HuggingFace model ID)
        # Try to load tokenizer from the same path as the model
        if os.path.exists(model_name) or "/" in model_name:
            print(f"Loading tokenizer from path: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(
                f"Tokenizer '{model_name}' not found in registry and not a valid path.\n"
                f"Available registry tokenizers: {list(TOKENIZERS_REGISTRY.keys())}"
            )
    
    return tokenizer

def chat(model, tokenizer, prompt, device="cpu", max_new_tokens=512, temperature=0.7, use_chat_template=True): 
    """
    Send a message to the model and get a response.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: User's message/prompt
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        use_chat_template: Whether to use chat template (False for base models)
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        # Apply chat template and tokenize
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # Handle both tensor and dict returns
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
    else:
        # Plain tokenization without chat template
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    input_ids = input_ids.to(device)
    
    # Create attention mask (all ones since we're not padding)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Help prevent repetition loops, especially for base models
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (remove the input prompt)
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chat with a language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a registry model:
  python -m src.chat qwen2_5_05_instruct "Hello!"
  
  # Use a local checkpoint:
  python -m src.chat ./checkpoints/my_model "Hello!"
  
  # Use a HuggingFace model:
  python -m src.chat Qwen/Qwen2.5-0.5B-Instruct "Hello!"
        """
    )
    parser.add_argument("model_name", 
                       help="Model name from registry, local path, or HuggingFace model ID")
    parser.add_argument("message", nargs="+", help="Your message to the model")
    parser.add_argument("--no-chat-template", action="store_true", 
                       help="Don't use chat template (useful for base models)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum number of tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    user_message = " ".join(args.message)
    use_chat_template = not args.no_chat_template
    
    print(f"Loading model: {model_name}...")
    model, device = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    print(f"Model loaded on device: {device}")
    print(f"Using chat template: {use_chat_template}\n")
    
    print(f"You: {user_message}\n")
    print("Model: ", end="", flush=True)
    
    response = chat(
        model, 
        tokenizer, 
        user_message, 
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        use_chat_template=use_chat_template
    )
    print(response)

if __name__ == "__main__":
    main()