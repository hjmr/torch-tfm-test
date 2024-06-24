import torch
from vtf import VariationalTransformer


def create_model(device: torch.device) -> VariationalTransformer:
    model = VariationalTransformer(
        d_input=1,
        d_output=1,
        d_embed=16,
        d_latent=2,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        batch_first=True,
        device=device,
    )
    return model
