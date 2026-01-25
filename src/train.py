from src.utils import load_env 
load_env()

from src.training.trainers import load_bitlab_trainer 
from src.training.dataloaders import load_bitlab_datamodule
from src.utils import load_config 
import pytorch_lightning as pl
import sys 
import os 
from typing import Optional
import torch 



def load_datamodule(config: dict): 
    datamodule_cfg = config["datamodule"] 
    datamodule_type = datamodule_cfg.pop("type", None)  
    return load_bitlab_datamodule(datamodule_type, **datamodule_cfg)

def load_trainer(config: dict): 
    trainer_cfg = config["trainer"] 
    trainer_type = trainer_cfg.pop("type", None)
    return load_bitlab_trainer(trainer_type, **trainer_cfg)

def load_logger(config: dict): 
    logger_cfg = config["logger"].copy()  # Don't modify original config
    logger_type = logger_cfg.pop("type", None) 

    if logger_type == "tensorboard":
        return pl.loggers.TensorBoardLogger(**logger_cfg)
    elif logger_type == "wandb":
        return pl.loggers.WandbLogger(**logger_cfg)
    else:
        raise ValueError(f"Logger {logger_type} not found")

def load_checkpointer(config: dict): 
    checkpointer_cfg = config["checkpoint"] 
    return pl.callbacks.ModelCheckpoint(**checkpointer_cfg)

def load_pl_trainer(config: dict, logger: Optional["pl.loggers.logger.Logger"] = None, callbacks: Optional[list] = None):
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
    logger = load_logger(config)    
    callbacks = []
    if "checkpoint" in config:
        checkpointer = load_checkpointer(config)
        callbacks.append(checkpointer)
    pl_trainer = load_pl_trainer(config, logger, callbacks)
    pl_trainer.fit(trainer, datamodule) 



if __name__ == "__main__":
    main() 