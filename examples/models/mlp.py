import torch

from bitmodels import AutoBitModel
from bitmodels.mlp import BitMLPConfig, BitMLPModel


def main():
    # Create config
    config = BitMLPConfig(
        n_layers=3,
        in_channels=256,
        hidden_dim=128,
        out_channels=10,
        dropout=0.1,
    )

    # Method 1: AutoBitModel (cleanest!)
    model = AutoBitModel.from_config(config)
    print(f"Auto model type: {type(model).__name__}")

    # Method 2: Direct instantiation
    model_direct = BitMLPModel(config)

    # Test forward pass
    x = torch.randn(1, 256)
    output = model(x)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
