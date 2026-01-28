from huggingface_hub import snapshot_download


def download_qwen2_5_05B_instruct():
    """
    Download the Qwen2.5-0.5B-Instruct model and tokenizer from the Hugging Face Hub.
    Only downloads files; does not load model into memory.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    print(f"Downloaded {model_name} successfully")



def download_qwen2_5_05B_pt():
    """
    Download the Qwen/Qwen2.5-0.5B model and tokenizer from the Hugging Face Hub.
    Only downloads files; does not load model into memory.
    """
    model_name = "Qwen/Qwen2.5-0.5B"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    print(f"Downloaded {model_name} successfully")


def download_qwen3_06B_pt():
    """
    Download the Qwen/Qwen3-0.6B model and tokenizer from the Hugging Face Hub.
    Only downloads files; does not load model into memory.
    """
    model_name = "Qwen/Qwen3-0.6B"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    print(f"Downloaded {model_name} successfully")


def download_smollm2_135m_pt(): 
    model_name = "HuggingFaceTB/SmolLM2-135M"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    print(f"Downloaded {model_name} successfully")


def download_smollm2_360m_pt(): 
    model_name = "HuggingFaceTB/SmolLM2-360M"
    # Use snapshot_download to ensure all files (config, tokenizer, model) are cached
    snapshot_download(repo_id=model_name, repo_type="model")
    print(f"Downloaded {model_name} successfully")



        



DOWNLOAD_MODELS_REGISTRY = {
    "qwen2_5_05B_instruct": download_qwen2_5_05B_instruct,
    "qwen2_5_05B_pt": download_qwen2_5_05B_pt,
    "qwen3_06B_pt": download_qwen3_06B_pt,
    "smollm2_135m_pt": download_smollm2_135m_pt,
    "smollm2_360m_pt": download_smollm2_360m_pt,
}


def download_bitlab_model(model_name: str):
    if model_name not in DOWNLOAD_MODELS_REGISTRY:
        raise ValueError(f"Model {model_name} not found")
    return DOWNLOAD_MODELS_REGISTRY[model_name]()