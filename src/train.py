from src.utils import load_env

load_env()

import os
import sys
from typing import Optional

import pytorch_lightning as pl
import torch

from src.training.dataloaders import load_bitlab_datamodule
from src.training.trainers import load_bitlab_trainer
from src.utils import load_config


def load_datamodule(config: dict): 
    datamodule_cfg = config["datamodule"] 
    datamodule_type = datamodule_cfg.pop("type", None)  
    return load_bitlab_datamodule(datamodule_type, **datamodule_cfg)

def load_trainer(config: dict): 
    trainer_cfg = config["trainer"] 
    trainer_type = trainer_cfg.pop("type", None)
    return load_bitlab_trainer(trainer_type, **trainer_cfg)

def load_logger(config: dict): 
    logger_cfg = config.get("logger")
    if logger_cfg is None:
        return None
    logger_cfg = logger_cfg.copy()  # Don't modify original config
    logger_type = logger_cfg.pop("type", None) 

    if logger_type == "tensorboard":
        return pl.loggers.TensorBoardLogger(**logger_cfg)
    elif logger_type == "wandb":
        return pl.loggers.WandbLogger(**logger_cfg)
    else:
        raise ValueError(f"Logger {logger_type} not found")

def load_checkpointer(config: dict): 
    checkpointer_cfg = config.get("checkpoint")
    if checkpointer_cfg is None:
        return None
    return pl.callbacks.ModelCheckpoint(**checkpointer_cfg)

def _resolve_max_tokens_to_max_steps(config: dict) -> None:
    """If pl_trainer has max_tokens, set max_steps from it and clear max_epochs so step limit applies."""
    pl_cfg = config.get("pl_trainer") or {}
    max_tokens = pl_cfg.get("max_tokens")
    if max_tokens is None:
        return
    dm_cfg = config.get("datamodule") or {}
    batch_size = int(dm_cfg.get("batch_size", 1))
    max_length = int(dm_cfg.get("max_length", 256))
    accumulate_grad_batches = int(pl_cfg.get("accumulate_grad_batches", 1))
    tokens_per_step = batch_size * max_length * accumulate_grad_batches
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive (batch_size * max_length * accumulate_grad_batches)")
    max_steps = int(max_tokens // tokens_per_step)
    if max_steps < 1:
        raise ValueError(
            f"max_tokens={max_tokens} yields max_steps={max_steps} (tokens_per_step={tokens_per_step}). "
            "Increase max_tokens or use max_epochs / max_steps instead."
        )
    config["pl_trainer"]["max_steps"] = max_steps
    config["pl_trainer"]["max_epochs"] = -1  # Let max_steps be the limit
    # Remove max_tokens so Trainer doesn't see an unknown kwarg
    config["pl_trainer"].pop("max_tokens", None)


def load_pl_trainer(config: dict, logger: Optional["pl.loggers.logger.Logger"] = None, callbacks: Optional[list] = None):
    _resolve_max_tokens_to_max_steps(config)
    pl_trainer_cfg = config["pl_trainer"]
    pl_trainer = pl.Trainer(**pl_trainer_cfg, logger=logger, callbacks=callbacks)
    return pl_trainer

def main(): 
    torch.set_float32_matmul_precision('high')

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    config = load_config(config_path)
    datamodule = load_datamodule(config) 
    trainer = load_trainer(config) 
    # Compile the model, not the Lightning module, to avoid pickling issues with multiprocessing
    logger = load_logger(config)
    if logger is not None:
        logger.log_hyperparams(trainer.hparams)
    #trainer.model = torch.compile(trainer.model)
    checkpointer = load_checkpointer(config)
    callbacks = [checkpointer] if checkpointer is not None else None
    pl_trainer = load_pl_trainer(config, logger, callbacks)
    pl_trainer.fit(trainer, datamodule) 



if __name__ == "__main__":
    main() 