from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen2_5_05B_instruct(): 
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

def load_qwen2_5_05B_pt(): 
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


def load_qwen3_06B_pt(): 
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


MODELS_REGISTRY = {
    "qwen2_5_05B_instruct": load_qwen2_5_05B_instruct,
    "qwen2_5_05B_pt": load_qwen2_5_05B_pt,
    "qwen3_06B_pt": load_qwen3_06B_pt,
}

def load_bitlab_model(model_name: str): 
    if model_name not in MODELS_REGISTRY: 
        raise ValueError(f"Model {model_name} not found")
    return MODELS_REGISTRY[model_name]()