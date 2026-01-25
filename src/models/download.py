from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


def download_qwen2_5_05B_instruct():
    """
    Download the Qwen2.5-0.5B-Instruct model and tokenizer from the Hugging Face Hub.
    Returns:
        A tuple containing (model, tokenizer).
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



def download_qwen2_5_05B_pt():
    """
    Download the Qwen/Qwen2.5-0.5B model and tokenizer from the Hugging Face Hub.
    Returns:
        A tuple containing (model, tokenizer).
    """
    model_name = "Qwen/Qwen2.5-0.5B"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def download_qwen3_06B_pt():
    """
    Download the Qwen/Qwen3-0.6B model and tokenizer from the Hugging Face Hub.
    Returns:
        A tuple containing (model, tokenizer).
    """
    model_name = "Qwen/Qwen3-0.6B"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



DOWNLOAD_MODELS_REGISTRY = {
    "qwen2_5_05B_instruct": download_qwen2_5_05B_instruct,
    "qwen2_5_05B_pt": download_qwen2_5_05B_pt,
    "qwen3_06B_pt": download_qwen3_06B_pt,
}


def download_bitlab_model(model_name: str):
    if model_name not in DOWNLOAD_MODELS_REGISTRY:
        raise ValueError(f"Model {model_name} not found")
    return DOWNLOAD_MODELS_REGISTRY[model_name]()