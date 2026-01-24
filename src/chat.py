import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.models import MODELS_REGISTRY
from src.models.tokenizers import TOKENIZERS_REGISTRY
from src.training.trainers import SFTTrainer, BitDistillTrainer, TRAINERS_REGISTRY


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_name: str, checkpoint: str | None = None, device: str | None = None):
    """
    Supports:
      - Registry models (MODELS_REGISTRY)
      - HF model id / local folder
      - Lightning .ckpt either as model_name OR as weights to load into a registry model
    Returns: (model, device, tokenizer_model_name)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_model_name = model_name

    # Case A: model_name itself is a Lightning checkpoint
    if model_name.endswith(".ckpt") and os.path.exists(model_name) and checkpoint is None:
        print(f"Loading Lightning module from checkpoint: {model_name}")
        
        # Try to determine trainer type from checkpoint
        ckpt = torch.load(model_name, map_location="cpu")
        trainer_class = None
        
        # Check hyperparameters for trainer type
        if "hyper_parameters" in ckpt:
            hparams = ckpt["hyper_parameters"]
            if isinstance(hparams, dict) and "_target_" in hparams:
                # Lightning 2.0+ format
                target = hparams["_target_"]
                if "BitDistillTrainer" in target:
                    trainer_class = BitDistillTrainer
                elif "SFTTrainer" in target:
                    trainer_class = SFTTrainer
            elif isinstance(hparams, dict):
                # Try to infer from class name in state_dict keys
                state_keys = ckpt.get("state_dict", {}).keys()
                if any("student" in k for k in state_keys):
                    trainer_class = BitDistillTrainer
                else:
                    trainer_class = SFTTrainer
        
        # Default to SFTTrainer if can't determine
        if trainer_class is None:
            trainer_class = SFTTrainer
        
        trainer = trainer_class.load_from_checkpoint(model_name, map_location=device)
        
        # Extract model - BitDistillTrainer uses student, SFTTrainer uses model
        if isinstance(trainer, BitDistillTrainer):
            model = trainer.student
        else:
            model = trainer.model

        # try to recover base model name for tokenizer
        if hasattr(trainer, "model_name"):
            tokenizer_model_name = trainer.model_name
        elif hasattr(trainer, "hparams") and isinstance(trainer.hparams, dict) and "model_name" in trainer.hparams:
            tokenizer_model_name = trainer.hparams["model_name"]
        else:
            raise ValueError(
                "Could not infer tokenizer model name from checkpoint. "
                "Pass --tokenizer-model-name explicitly."
            )

    # Case B: registry model
    elif model_name in MODELS_REGISTRY:
        model = MODELS_REGISTRY[model_name]()

        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise ValueError(f"Checkpoint not found: {checkpoint}")
            print(f"Loading weights from checkpoint: {checkpoint}")

            ckpt = torch.load(checkpoint, map_location="cpu")
            if checkpoint.endswith(".ckpt"):
                # Lightning checkpoint stores weights under 'state_dict'
                if "state_dict" not in ckpt:
                    raise ValueError(f"No state_dict in Lightning checkpoint: {checkpoint}")
                state = ckpt["state_dict"]
                # Handle different checkpoint formats:
                # - SFTTrainer: "model.model.xxx" (double prefix from self.model)
                # - BitDistillTrainer: "model.xxx" (single prefix, matches HuggingFace format)
                # HuggingFace models expect "model.xxx", so:
                # - If double prefix "model.model.xxx", strip one "model."
                # - If single prefix "model.xxx", use as-is (no modification needed)
                if any(k.startswith("model.model.") for k in state.keys()):
                    # SFTTrainer format - strip one "model." prefix
                    state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
                # BitDistillTrainer format already has correct "model.xxx" format, use as-is
            else:
                # raw state_dict
                state = ckpt

            model.load_state_dict(state, strict=True)
            print(f"Successfully loaded weights from {checkpoint}")

    # Case C: HF model id or local folder
    else:
        if os.path.exists(model_name) or "/" in model_name:
            print(f"Loading HF model from: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if checkpoint is not None:
                print(f"Loading weights from checkpoint: {checkpoint}")
                state = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(state, strict=False)
        else:
            raise ValueError(f"Model '{model_name}' not found in registry and not a path/HF id.")

    model = model.to(device).eval()
    return model, device, tokenizer_model_name


def load_tokenizer(model_name: str):
    if model_name in TOKENIZERS_REGISTRY:
        return TOKENIZERS_REGISTRY[model_name]()
    if os.path.exists(model_name) or "/" in model_name:
        print(f"Loading tokenizer from: {model_name}")
        return AutoTokenizer.from_pretrained(model_name)
    raise ValueError(f"Tokenizer '{model_name}' not found in registry and not a path/HF id.")


def get_stop_token_ids(tokenizer, use_chat_template: bool) -> list[int] | int | None:
    """
    For chat models, it's common that the model ends turns with a special token
    like <|im_end|> rather than tokenizer.eos_token. We include both if present.
    """
    stop_ids = set()

    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    if use_chat_template:
        # Qwen2.5 Instruct commonly uses <|im_end|> as end-of-turn.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            stop_ids.add(int(im_end_id))

    if not stop_ids:
        return None
    if len(stop_ids) == 1:
        return next(iter(stop_ids))
    return sorted(stop_ids)


def build_inputs(tokenizer, prompt: str, use_chat_template: bool):
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return ids
    else:
        return tokenizer(prompt, return_tensors="pt")["input_ids"]


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    use_chat_template: bool = True,
):
    input_ids = build_inputs(tokenizer, prompt, use_chat_template).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    stop_ids = get_stop_token_ids(tokenizer, use_chat_template=use_chat_template)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        eos_token_id=stop_ids,
    )

    gen = out[0, input_ids.shape[1]:].tolist()

    # Correct “stopped at EOS” check
    if stop_ids is None:
        stopped_at_eos = False
    else:
        stop_set = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids)
        stopped_at_eos = (len(gen) > 0) and (gen[-1] in stop_set)

    text_with_special = tokenizer.decode(gen, skip_special_tokens=False)
    text_clean = tokenizer.decode(gen, skip_special_tokens=True)

    return {
        "tokens": gen,
        "stopped_at_eos": stopped_at_eos,
        "last_token_id": gen[-1] if gen else None,
        "text_with_special": text_with_special,
        "text_clean": text_clean,
        "stop_ids": stop_ids,
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("model_name", nargs="?", default=None)
    p.add_argument("message", nargs="*", default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--tokenizer-model-name", type=str, default=None)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config)
        model_name = cfg["model_name"]
        prompt = cfg.get("message", "")
        checkpoint = cfg.get("checkpoint", None)
        use_chat_template = bool(cfg.get("use_chat_template", True))
        max_tokens = int(cfg.get("max_tokens", 256))
        temperature = float(cfg.get("temperature", 0.7))
        tokenizer_model_name = cfg.get("tokenizer_model_name", None)
    else:
        if args.model_name is None or not args.message:
            raise SystemExit("Usage: python -m src.chat <model_name> <message...>")
        model_name = args.model_name
        prompt = " ".join(args.message)
        checkpoint = args.checkpoint
        use_chat_template = not args.no_chat_template
        max_tokens = args.max_tokens
        temperature = args.temperature
        tokenizer_model_name = args.tokenizer_model_name

    print(f"Loading model: {model_name}...")
    if checkpoint:
        print(f"Will load checkpoint: {checkpoint}")

    model, device, inferred_tok_name = load_model(model_name, checkpoint=checkpoint)
    tok_name = tokenizer_model_name or inferred_tok_name
    tokenizer = load_tokenizer(tok_name)

    print(f"Model loaded on device: {device}")
    print(f"Tokenizer: {tok_name}")
    print(f"Using chat template: {use_chat_template}\n")

    print(f"You: {prompt}\n")

    if use_chat_template:
        r1 = generate_once(
            model, tokenizer, prompt, device=device,
            max_new_tokens=max_tokens, temperature=temperature, use_chat_template=True
        )
        print(f"  Stop IDs: {r1['stop_ids']}")
        print(f"  Tokens: {len(r1['tokens'])}/{max_tokens}")
        print(f"  Stopped at EOS: {r1['stopped_at_eos']}")
        print(f"  Last token ID: {r1['last_token_id']}")
        print(f"  Output: {repr(r1['text_with_special'])}\n")

    else:
        r = generate_once(
            model, tokenizer, prompt, device=device,
            max_new_tokens=max_tokens, temperature=temperature, use_chat_template=False
        )
        print(f"Stop IDs: {r['stop_ids']}")
        print(f"Tokens: {len(r['tokens'])}/{max_tokens}")
        print(f"Stopped at EOS: {r['stopped_at_eos']}")
        print(f"Last token ID: {r['last_token_id']}")
        print(f"Output: {repr(r['text_with_special'])}")


if __name__ == "__main__":
    main()