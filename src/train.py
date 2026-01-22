from src.training.trainers import TRAINERS_REGISTRY 
from src.training.dataloaders import DATALOADERS_REGISTRY
from src.utils import load_config 
import pytorch_lightning as pl
import sys 
import os 
from typing import Optional



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
    logger_cfg = config["logger"] 
    logger_type = logger_cfg.pop("type", None) 

    if logger_type == "tensorboard":
        return pl.loggers.TensorBoardLogger(**logger_cfg)
    elif logger_type == "wandb":
        return pl.loggers.WandbLogger(**logger_cfg)
    else:
        raise ValueError(f"Logger {logger_type} not found")


def load_pl_trainer(config: dict, logger: Optional["pl.loggers.logger.Logger"] = None):
    pl_trainer_cfg = config["pl_trainer"] 
    pl_trainer = pl.Trainer(**pl_trainer_cfg, logger=logger)
    return pl_trainer

def main(): 
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    config = load_config(config_path)
    datamodule = load_datamodule(config) 
    trainer = load_trainer(config) 
    logger = load_logger(config)    
    pl_trainer = load_pl_trainer(config, logger)
    pl_trainer.fit(trainer, datamodule) 



if __name__ == "__main__":
    main() 