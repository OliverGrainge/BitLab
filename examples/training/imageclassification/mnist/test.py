"""
Evaluate a trained MNIST classifier checkpoint.

Example
-------
python test.py config.yaml
python test.py config_a.yaml config_b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from bitlab.bittrainer.classification import BitImageClassifierTrainer

from datamodule import MNISTClassificationDataModule
from main import MNISTMLP, get_linear_layer, load_config


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def build_datamodule(
    config: Dict[str, Any], *, batch_size_override: int | None = None
) -> MNISTClassificationDataModule:
    val_batch_size = batch_size_override or config["val_batch_size"]

    datamodule = MNISTClassificationDataModule(
        train_batch_size=config["train_batch_size"],
        val_batch_size=val_batch_size,
        num_workers=config["num_workers"],
        pin_memory=False,
        use_augmentation=False,
        seed=config["seed"],
    )
    return datamodule


def build_model(config: Dict[str, Any]) -> MNISTMLP:
    linear_layer = get_linear_layer(
        config["model"]["layer_type"],
        config["model"].get("quant_type"),
    )

    hidden_dims: Iterable[int] | None = config["model"].get("hidden_dims")

    model = MNISTMLP(
        hidden_dims=hidden_dims,
        num_classes=config["num_classes"],
        linear_layer=linear_layer,
    )
    return model


def build_lit_module(
    config: Dict[str, Any], model: MNISTMLP, datamodule: MNISTClassificationDataModule
) -> BitImageClassifierTrainer:
    # Ensure the datamodule is ready so we can compute the number of steps.
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    steps_per_epoch = len(datamodule.train_dataloader())
    max_steps = steps_per_epoch * config["max_epochs"]

    lit_module = BitImageClassifierTrainer(
        model=model,
        num_classes=config["num_classes"],
        loss_type=config["loss_type"],
        label_smoothing=config.get("label_smoothing", 0.0),
        learning_rate=config["learning_rate"],
        lr_warmup_steps=config["lr_warmup_steps"],
        lr_scheduler=config["lr_scheduler"],
        max_lr_steps=max_steps,
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
        sgd_momentum=config.get("sgd_momentum", 0.9),
        sgd_nesterov=config.get("sgd_nesterov", False),
        top_k=config.get("top_k", [1]),
        compute_per_class_metrics=config.get("compute_per_class_metrics", False),
        log_samples_every_n_epochs=config.get("log_samples_every_n_epochs", 5),
        num_samples_to_log=config.get("num_samples_to_log", 16),
    )
    return lit_module


def load_checkpoint(
    lit_module: BitImageClassifierTrainer, checkpoint_path: Path, device: torch.device
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at '{checkpoint_path}' does not contain a 'state_dict' entry."
        )

    missing, unexpected = lit_module.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if missing:
        raise RuntimeError(f"Missing keys when loading state dict: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading state dict: {unexpected}")


def run_evaluation(
    lit_module: BitImageClassifierTrainer,
    datamodule: MNISTClassificationDataModule,
    *,
    split: str,
    accelerator: str,
    devices: Any,
    precision: str,
) -> Dict[str, Any]:
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    if split == "val":
        results = trainer.validate(lit_module, datamodule=datamodule, verbose=False)
    else:
        results = trainer.test(lit_module, datamodule=datamodule, verbose=False)

    if not results:
        return {}
    return results[0]


def infer_checkpoint_path(config_path: Path) -> Path:
    config_name = config_path.stem
    candidates = [
        Path.cwd(),
        config_path.resolve().parent,
        Path(__file__).resolve().parent,
    ]
    for base in candidates:
        candidate = base / "checkpoints" / config_name / f"{config_name}.ckpt"
        if candidate.exists():
            return candidate
    return Path.cwd() / "checkpoints" / config_name / f"{config_name}.ckpt"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MNIST classifier checkpoint."
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="+",
        help="Path(s) to the YAML configuration used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help=(
            "Path to the PyTorch Lightning checkpoint (.ckpt). Defaults to "
            "checkpoints/<config>/<config>.ckpt. Provide multiple --checkpoint "
            "arguments to match multiple configs."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the evaluation batch size (defaults to config).",
    )
    return parser.parse_args(argv)


def print_results(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    metrics: Dict[str, Any],
) -> None:
    title = f"Results for {config_path} [{split}]"
    separator = "-" * len(title)
    print(f"\n{title}\n{separator}")
    print(f"Checkpoint : {checkpoint_path}")

    if not metrics:
        print("No metrics were returned during evaluation.\n")
        return

    metric_keys = sorted(metrics.keys())
    width = max((len(key) for key in metric_keys), default=0)

    for key in metric_keys:
        value = metrics[key]
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        print(f"{key:<{width}} : {value_str}")
    print()


def validate_checkpoints(
    configs: Sequence[Path], checkpoints: Sequence[Path] | None
) -> list[Path | None]:
    if checkpoints is None:
        return [None] * len(configs)

    if len(checkpoints) == 1 and len(configs) > 1:
        raise ValueError(
            "Multiple configs provided but only one --checkpoint supplied. "
            "Either supply matching --checkpoint values or rely on automatic discovery."
        )

    if len(checkpoints) not in (0, len(configs)):
        raise ValueError(
            "Number of --checkpoint arguments does not match number of configs."
        )

    return list(checkpoints) if checkpoints else [None] * len(configs)


def evaluate_single_config(
    config_path: Path,
    checkpoint_override: Path | None,
    *,
    split: str,
    batch_size: int | None,
    device: torch.device,
) -> int:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    checkpoint_path = checkpoint_override or infer_checkpoint_path(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(str(config_path))

    datamodule = build_datamodule(config, batch_size_override=batch_size)
    model = build_model(config)
    lit_module = build_lit_module(config, model, datamodule)

    lit_module.to(device)
    load_checkpoint(lit_module, checkpoint_path, device=device)

    stage = "validate" if split == "val" else "test"
    datamodule.setup(stage=stage)

    metrics = run_evaluation(
        lit_module,
        datamodule,
        split=split,
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
    )

    print_results(config_path, checkpoint_path, split, metrics)
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    torch.set_float32_matmul_precision("medium")
    device = get_device()

    checkpoints = validate_checkpoints(args.config, args.checkpoint)

    exit_code = 0
    for config_path, checkpoint_override in zip(
        args.config, checkpoints, strict=False
    ):
        try:
            result = evaluate_single_config(
                config_path,
                checkpoint_override,
                split=args.split,
                batch_size=args.batch_size,
                device=device,
            )
            exit_code = max(exit_code, result)
        except Exception as exc:  # noqa: BLE001
            exit_code = 1
            print(f"\n[{config_path}] Error: {exc}\n")

    return exit_code


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    sys.exit(main(sys.argv[1:]))
