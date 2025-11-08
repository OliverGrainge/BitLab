# MNIST Training Pipeline

This directory now mirrors the CIFAR-10 training example layout. It contains a PyTorch Lightning pipeline for MNIST digit classification built on BitLab components.

## Layout

- `main.py` – training entrypoint that loads a YAML config and launches PyTorch Lightning with `BitImageClassifierTrainer`.
- `datamodule.py` – Lightning datamodule wrapping the Hugging Face `mnist` dataset with optional data augmentation.
- `config_*.yaml` – ready-to-run configurations covering standard and BitLinear variants.

## Quick Start

```bash
cd examples/training/imageclassification/mnist

# Standard float MLP
python main.py config_standard.yaml

# BitLinear (ai8pc_wpt) variant
python main.py config_ai8pc_wpt.yaml
```

Each run writes checkpoints to `./checkpoints/` and logs (including WandB artifacts) to `./logs/` in the current working directory.

## Configuration Fields

All configs share the same structure as the CIFAR-10 example. Key fields:

- `run_name`, `wandb_project`, `wandb_entity` – logging metadata.
- `train_batch_size`, `val_batch_size`, `num_workers`, `use_augmentation` – dataloader behaviour.
- `optimizer`, `learning_rate`, `lr_scheduler`, `weight_decay` – optimiser setup.
- `model.layer_type` – `standard` or `bitlinear`.
- `model.quant_type` – required when `layer_type` is `bitlinear` (`ai8pc_wpt`, `ai8pg128_wpt`, `ai8pg256_wpt`).
- `model.hidden_dims` – hidden-layer widths for the MLP.

Adjust these YAML files or create new ones to explore different hyper-parameters, quantisation schemes, or logging preferences.

## Notes

- The first and final linear layers in `main.py` remain in full precision for stability; intermediate layers can be quantised via `BitLinear`.
- `use_augmentation: true` enables mild affine jitter (rotation + translation) to improve robustness.
- WandB logging is enabled by default; set `wandb_entity` or disable logging at the PyTorch Lightning trainer level if required.

