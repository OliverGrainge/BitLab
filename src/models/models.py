from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen2_5_05_instruct(): 
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

def load_qwen2_5_05_pt(): 
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

MODELS_REGISTRY = {
    "qwen2_5_05_instruct": load_qwen2_5_05_instruct,
    "qwen2_5_05_pt": load_qwen2_5_05_pt,
}