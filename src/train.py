from src.training.trainers import TRAINERS_REGISTRY 
from src.training.dataloaders import DATALOADERS_REGISTRY
from src.utils import load_config 
import pytorch_lightning as pl
import sys 
import os 
from typing import Optional
import torch 



def load_datamodule(config: dict): 
    datamodule_cfg = config["datamodule"] 
    datamodule_type = datamodule_cfg.pop("type", None)  
    if datamodule_type not in DATALOADERS_REGISTRY:
        raise ValueError(f"DataModule {datamodule_type} not found")
    return DATALOADERS_REGISTRY[datamodule_type](**datamodule_cfg)

def load_trainer(config: dict): 
    trainer_cfg = config["trainer"] 
    trainer_type = trainer_cfg.pop("type", None)
    if trainer_type not in TRAINERS_REGISTRY:
        raise ValueError(f"Trainer {trainer_type} not found")
    return TRAINERS_REGISTRY[trainer_type](**trainer_cfg)

def load_logger(config: dict): 
    logger_cfg = config["logger"].copy()  # Don't modify original config
    logger_type = logger_cfg.pop("type", None) 

    if logger_type == "tensorboard":
        # Use experiment_name from config as the name parameter
        # TensorBoardLogger structure: save_dir/name/version_n/
        # Remove from logger_cfg so it's never passed to TensorBoardLogger/SummaryWriter
        logger_cfg.pop("experiment_name", None)
        experiment_name = config.get("experiment_name")
        if experiment_name:
            # If save_dir ends with experiment_name, use parent as save_dir
            save_dir = logger_cfg.get("save_dir", "logs/tensorboard")
            if save_dir.endswith(experiment_name):
                # Extract parent directory
                parent_dir = "/".join(save_dir.split("/")[:-1]) if "/" in save_dir else "logs/tensorboard"
                logger_cfg["save_dir"] = parent_dir
            elif "save_dir" not in logger_cfg:
                logger_cfg["save_dir"] = "logs/tensorboard"
            
            # Set name to experiment_name for specific experiment folder
            logger_cfg["name"] = experiment_name
        
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