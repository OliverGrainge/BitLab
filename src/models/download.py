from transformers import AutoModelForCausalLM, AutoTokenizer


def download_qwen2_5_05_instruct():
    """
    Download the Qwen2.5-0.5B-Instruct model and tokenizer from the Hugging Face Hub.
    Returns:
        A tuple containing (model, tokenizer).
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



def download_qwen2_5_05_pt():
    """
    Download the Qwen/Qwen2.5-0.5B model and tokenizer from the Hugging Face Hub.
    Returns:
        A tuple containing (model, tokenizer).
    """
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



DOWNLOAD_MODELS_REGISTRY = {
    "qwen2_5_05_instruct": download_qwen2_5_05_instruct,
    "qwen2_5_05_pt": download_qwen2_5_05_pt,
}