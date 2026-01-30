#!/usr/bin/env python3
"""
Script to load a Lightning trainer from a training YAML and chat with the model.
Uses the trainer's chat() method so quantized / structure-changed models load correctly.
Pass the training YAML path (e.g. runs/training/base/bitdistill_base.yaml). The YAML
must have a 'chat:' section with prompt, generation_params, etc. The best checkpoint
under checkpoint.dirpath is auto-loaded unless chat.checkpoint_path is set.
"""
from dotenv import load_dotenv
load_dotenv()

import re
import sys
from pathlib import Path
from typing import Optional

import torch

from src.training.trainers import load_bitlab_trainer
from src.utils import load_config


def _resolve_dirpath(dirpath: str, repo_root: Path) -> Path:
    """Resolve checkpoint dirpath to an absolute path (repo root if relative)."""
    p = Path(dirpath)
    if not p.is_absolute():
        p = repo_root / dirpath
    return p.resolve()


def find_best_checkpoint(
    dirpath: Path,
    monitor: str,
    mode: str,
    save_last: bool = True,
) -> Optional[str]:
    """
    Find the best checkpoint in dirpath by parsing the monitor metric from filenames.
    Lightning ModelCheckpoint saves files like "name-metric=value.ckpt".
    Returns the path to the best .ckpt file, or last.ckpt if save_last and no metric-named files.
    """
    if not dirpath.exists():
        return None
    pattern = re.compile(re.escape(monitor) + r"=([\d.]+)")
    candidates = []
    for f in dirpath.glob("*.ckpt"):
        if f.name == "last.ckpt":
            continue
        m = pattern.search(f.name)
        if m is not None:
            try:
                value = float(m.group(1))
                candidates.append((value, str(f)))
            except ValueError:
                pass
    if not candidates:
        if save_last:
            last_ckpt = dirpath / "last.ckpt"
            if last_ckpt.exists():
                return str(last_ckpt)
        return None
    candidates.sort(key=lambda x: x[0], reverse=(mode == "max"))
    return candidates[0][1]


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.chat <training_config.yaml>")
        print("  Pass the training YAML (e.g. runs/training/base/bitdistill_base.yaml).")
        print("  It must contain a 'chat:' section with prompt, generation_params, etc.")
        print("  Best checkpoint is auto-loaded from checkpoint.dirpath unless chat.checkpoint_path is set.")
        sys.exit(1)

    config_path = sys.argv[1]
    repo_root = Path(__file__).resolve().parent.parent
    if not Path(config_path).is_absolute():
        config_path = str(repo_root / config_path)

    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)

    # Allow a thin wrapper that only points to the training YAML
    if "training_config" in config:
        training_config_path = config["training_config"]
        if not Path(training_config_path).is_absolute():
            training_config_path = str(repo_root / training_config_path)
        print(f"Loading training configuration from {training_config_path}...")
        config = load_config(training_config_path)

    chat_section = config.get("chat")
    if not chat_section:
        raise ValueError(
            "Training config must contain a 'chat:' section with at least 'prompt' and optionally "
            "generation_params, show_tokens, use_chat_template, checkpoint_path."
        )

    prompt = chat_section["prompt"]
    generation_params = chat_section.get("generation_params", {})
    show_tokens = chat_section.get("show_tokens", True)
    use_chat_template = chat_section.get("use_chat_template", False)
    checkpoint_path_override = chat_section.get("checkpoint_path")
    load_checkpoint = chat_section.get("load_checkpoint", True)

    trainer_cfg = config["trainer"].copy()
    trainer_type = trainer_cfg.pop("type", None)
    if not trainer_type:
        raise ValueError("Training config 'trainer' section must have 'type' (e.g. bitdistillgptqpretrainer)")

    checkpoint_cfg = config.get("checkpoint", {})

    # Resolve checkpoint: explicit path, or auto best, or skip if load_checkpoint is false
    checkpoint_path = None
    if not load_checkpoint:
        pass
    elif checkpoint_path_override is not None and checkpoint_path_override != "":
        checkpoint_path = checkpoint_path_override
        if isinstance(checkpoint_path, str) and not Path(checkpoint_path).is_absolute():
            checkpoint_path = str(repo_root / checkpoint_path)
    else:
        dirpath = checkpoint_cfg.get("dirpath")
        if dirpath:
            resolved_dir = _resolve_dirpath(dirpath, repo_root)
            monitor = checkpoint_cfg.get("monitor", "train_loss")
            mode = checkpoint_cfg.get("mode", "min")
            save_last = checkpoint_cfg.get("save_last", True)
            checkpoint_path = find_best_checkpoint(resolved_dir, monitor, mode, save_last)
            if checkpoint_path:
                print(f"Auto-selected best checkpoint: {checkpoint_path}")

    print(f"\nTrainer: {trainer_type}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    else:
        print("Checkpoint: none (base model only)")
    print(f"Prompt: {prompt}")
    print(f"Use chat template: {use_chat_template}")
    print(f"Generation parameters: {generation_params}")
    print(f"Show tokens: {show_tokens}\n")

    print("Instantiating Lightning trainer from training config...")
    pl_module = load_bitlab_trainer(trainer_type, **trainer_cfg)

    print("Preparing QAT structure so checkpoint keys match...")
    pl_module.prepare_qat()

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        pl_module.load_state_dict(ckpt["state_dict"], strict=True)
        print("Checkpoint loaded successfully.")

    pl_module.eval()

    print("\nGenerating...\n")
    pl_module.chat(
        prompt=prompt,
        generation_params=generation_params,
        show_tokens=show_tokens,
        use_chat_template=use_chat_template,
    )


if __name__ == "__main__":
    main()