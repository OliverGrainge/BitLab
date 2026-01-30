from .sfttrainer import SFTTrainer
from .bitdistill import BitDistillPreTrainer
from typing import Optional 

TRAINERS_REGISTRY = {
    "sfttrainer": SFTTrainer,
    "bitdistillpretrainer": BitDistillPreTrainer,
}


def load_bitlab_trainer(trainer_name: str, checkpoint_path: Optional[str] = None, **kwargs): 
    if checkpoint_path is not None: 
        module_cls = TRAINERS_REGISTRY[trainer_name]
        module = module_cls.load_from_checkpoint(checkpoint_path)
        return module
    if trainer_name not in TRAINERS_REGISTRY:
        raise ValueError(f"Trainer {trainer_name} not found")
    return TRAINERS_REGISTRY[trainer_name](**kwargs)

