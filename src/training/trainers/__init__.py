from .sfttrainer import SFTTrainer
from .bitdistillpretrainer import BitDistillPreTrainer
from .bitdistillawqpretrainer import BitDistillAWQPreTrainer
from .bitdistiallptqpretrainer import BitDistillPTQPreTrainer
from .bitdistillgptqpretrainer import BitDistillGPTQPreTrainer

TRAINERS_REGISTRY = {
    "sfttrainer": SFTTrainer,
    "bitdistillpretrainer": BitDistillPreTrainer,
    "bitdistillawqpretrainer": BitDistillAWQPreTrainer,
    "bitdistillptqpretrainer": BitDistillPTQPreTrainer,
    "bitdistillgptqpretrainer": BitDistillGPTQPreTrainer,
}


def load_bitlab_trainer(trainer_name: str, **kwargs): 
    if trainer_name not in TRAINERS_REGISTRY:
        raise ValueError(f"Trainer {trainer_name} not found")
    return TRAINERS_REGISTRY[trainer_name](**kwargs)

