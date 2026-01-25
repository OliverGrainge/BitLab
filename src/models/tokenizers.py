from transformers import AutoTokenizer


def load_qwen2_5_05B_instruct_tokenizer(): 
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_qwen2_5_05B_pt_tokenizer(): 
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_qwen3_06B_pt_tokenizer(): 
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

TOKENIZERS_REGISTRY = {
    "qwen2_5_05B_instruct": load_qwen2_5_05B_instruct_tokenizer,
    "qwen2_5_05B_pt": load_qwen2_5_05B_pt_tokenizer,
    "qwen3_06B_pt": load_qwen3_06B_pt_tokenizer,
}

def load_bitlab_tokenizer(tokenizer_name: str): 
    if tokenizer_name not in TOKENIZERS_REGISTRY: 
        raise ValueError(f"Tokenizer {tokenizer_name} not found")
    return TOKENIZERS_REGISTRY[tokenizer_name]()