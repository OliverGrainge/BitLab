from transformers import AutoModelForCausalLM, AutoTokenizer


def download_qwen2_5_05B_instruct():
    """
    Download the Qwen2.5-0.5B-Instruct model and tokenizer from the Hugging Face Hub.
    Returns:
        A tuple containing (model, tokenizer).
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
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
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



DOWNLOAD_MODELS_REGISTRY = {
    "qwen2_5_05B_instruct": download_qwen2_5_05B_instruct,
    "qwen2_5_05B_pt": download_qwen2_5_05B_pt,
    "qwen3_06B_pt": download_qwen3_06B_pt,
}