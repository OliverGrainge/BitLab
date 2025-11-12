import torch
from transformers import AutoTokenizer

from bitlab.bitmodels.auto import (BitAutoModel, BitAutoModelForCausalLM,
                                   BitAutoModelForImageClassification,
                                   BitAutoModelForImageGeneration)
from bitlab.bitmodels.causallm import BitNetConfig, BitNetForCausalLM
from bitlab.bitmodels.imageclassification import (BitMLPConfig, BitMLPModel,
                                                  BitResNetConfig,
                                                  BitResNetModel)
from bitlab.bitmodels.imagegeneration import BitUNetConfig, BitUNetModel

resnet_cfg = BitResNetConfig()
unet_cfg = BitUNetConfig()
mlp_cfg = BitMLPConfig()
bitnet_cfg = BitNetConfig()

resnet_model = BitAutoModelForCausalLM.from_pretrained("bitnet:base")

print(resnet_model)
