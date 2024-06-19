import torch
from vtf import VariationalTransformer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"device:{device}")

model = VariationalTransformer(
    d_input=1,
    d_output=1,
    d_embed=8,
    d_latent=16,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    batch_first=True,
    device=device,
)
model.load_state_dict(torch.load("tf_test.pth"))

model.train(False)
z = torch.randn((1, 1, 16), device=device)
max_len = 100
output = model.generate(z, max_len)
for y in output:
    print(y)
