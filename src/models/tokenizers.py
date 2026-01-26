from transformers import AutoTokenizer
import os


def _get_cache_dir():
    """Get the HuggingFace cache directory from environment variables."""
    return os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))


def load_qwen2_5_05B_instruct_tokenizer(): 
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir = _get_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)
    return tokenizer


def load_qwen2_5_05B_pt_tokenizer(): 
    model_name = "Qwen/Qwen2.5-0.5B"
    cache_dir = _get_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)
    return tokenizer



    
def load_qwen3_06B_pt_tokenizer():
    model_name = "Qwen/Qwen3-0.6B"
    cache_dir = _get_cache_dir()
    # Point to the *local snapshot*, not the hub ID
    snap = os.path.join(
        cache_dir,
        "models--Qwen--Qwen3-0.6B",
        "snapshots",
        "c1899de289a04d12100db370d81485cdf75e47ca"
    )

    return AutoTokenizer.from_pretrained(
        snap,
        local_files_only=True,
        use_fast=True,
        # optional: makes intent explicit
        trust_remote_code=False,
    )

TOKENIZERS_REGISTRY = {
    "qwen2_5_05B_instruct": load_qwen2_5_05B_instruct_tokenizer,
    "qwen2_5_05B_pt": load_qwen2_5_05B_pt_tokenizer,
    "qwen3_06B_pt": load_qwen3_06B_pt_tokenizer,
}

def load_bitlab_tokenizer(tokenizer_name: str): 
    if tokenizer_name not in TOKENIZERS_REGISTRY: 
        raise ValueError(f"Tokenizer {tokenizer_name} not found")
    return TOKENIZERS_REGISTRY[tokenizer_name]()