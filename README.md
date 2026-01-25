# BitLab

BitLab is a PyTorch Lightning–based training and inference toolkit for causal language models. It supports **BitDistill** (continual pretraining with BitLinear-style quantization), **SFT** (supervised fine-tuning), and interactive **chat** over registry or custom models.

## Setup

```bash
pip install -r requirements.txt
```

From the project root, run commands as `python -m src.<module>`.

## Downloading datasets and models

List available datasets and models:

```bash
python -m src.download --list
```

Download by name:

```bash
python -m src.download --datasets alpaca fineweb-edu falcon-refinedweb --models qwen3_06B_pt
python -m src.download --all
```

**Datasets:** `alpaca`, `mnli`, `fineweb-edu`, `falcon-refinedweb`  
**Models:** `qwen2_5_05B_instruct`, `qwen2_5_05B_pt`, `qwen3_06B_pt`

### Where files are stored

- **HuggingFace cache** (models, tokenizers, Hub datasets like `alpaca` / `mnli`): use standard HuggingFace env vars. The download script does not pass `cache_dir`, so their defaults apply.
  - **`HF_HOME`** — Base for all HuggingFace data (hub + datasets cache). Default: `~/.cache/huggingface`.
  - **`HF_HUB_CACHE`** — Hub cache only (models, tokenizers). Default: `$HF_HOME/hub`.
  - **`HF_DATASETS_CACHE`** — Datasets library cache only.

- **BitLab data** (partial dataset downloads: `fineweb-edu`, `falcon-refinedweb`): use **`BITLAB_DATA_DIR`**. This is the root; datasets are stored in subdirs (e.g. `fineweb-edu`, `falcon-refinedweb`). Default: `data`.

**Example — custom locations:**

```bash
export HF_HOME=/mnt/cache/huggingface
export BITLAB_DATA_DIR=/mnt/data/bitlab
python -m src.download --datasets fineweb-edu falcon-refinedweb --models qwen3_06B_pt
```

Training and chat use the same env vars when loading data or models, so set them before running those too.

## Training

Training is driven by YAML configs under `runs/training/`. Pass a config path:

```bash
python -m src.train runs/training/experiments/bitdistill_qwen3_pt.yaml
```

Configs specify `datamodule` (e.g. `alpaca-sft`, `fineweb-edu-pt`, `falcon-refinedweb-pt`), `trainer` (e.g. `sfttrainer`, `bitdistillpretrainer`), logger (TensorBoard / Wandb), checkpointing, and `pl_trainer` (epochs, GPU, precision, etc.).

## Chat / inference

Chat with a model by name and message:

```bash
python -m src.chat qwen2_5_05B_instruct "What is the capital of France?"
```

Or use a chat config (model, optional checkpoint, message, etc.):

```bash
python -m src.chat --config runs/chat/experiments/sft_qwen05_pt.yaml
```

Supported model inputs: registry names, HuggingFace model IDs, local paths, or Lightning `.ckpt` paths. Use `--checkpoint` to load weights into a base model, and `--no-chat-template` for plain completion.

## Project layout

```
BitLab/
├── src/
│   ├── download.py       # CLI for datasets/models
│   ├── train.py          # Training entrypoint
│   ├── chat.py           # Chat/inference CLI
│   ├── data/             # Datasets, download logic
│   ├── models/           # Model/tokenizer registries, download
│   ├── training/         # Datamodules, trainers (SFT, BitDistill)
│   └── utils.py          # Config loading, get_data_dir, data_path
├── runs/
│   ├── training/         # Training configs (base + experiments)
│   └── chat/             # Chat configs
├── requirements.txt
└── README.md
```

Checkpoints and TensorBoard logs are written to paths specified in each experiment config (e.g. `checkpoints/...`, `logs/tensorboard/...`).
