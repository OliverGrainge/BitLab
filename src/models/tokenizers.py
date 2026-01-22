from transformers import AutoTokenizer

def load_qwen2_5_05_instruct_tokenizer(): 
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


TOKENIZERS_REGISTRY = {
    "qwen2_5_05_instruct": load_qwen2_5_05_instruct_tokenizer,
}