import torch
from vtf import VariationalTransformer


def create_model(device: torch.device) -> VariationalTransformer:
    model = VariationalTransformer(
        d_input=1,
        d_output=1,
        d_embed=32,
        d_latent=2,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        batch_first=True,
        device=device,
    )
    return model
