from tf import TransformerModel

import torch
from torch.optim import Adam
from torch.nn import MSELoss

model = TransformerModel(
    d_input=1,
    d_output=1,
    d_embed=8,
    d_latent=16,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    batch_first=True,
)
optimizer = Adam(model.parameters(), lr=0.001)
rec_loss = MSELoss()

sin_data = torch.sin(torch.arange(0, 100, 0.1)).unsqueeze(-1)
cos_data = torch.cos(torch.arange(0, 100, 0.1)).unsqueeze(-1)

model.train(True)
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(sin_data, cos_data)
    loss = rec_loss(output, cos_data)
    loss += model.kl_loss
    loss.backward()
    optimizer.step()
    print(f"epoch:{epoch}, loss:{loss.item()}, kl_loss:{model.kl_loss.item()}")

z = torch.randn((1, 10, 16))
max_len = 100
output = model.generate(z, max_len)
for y in output:
    print(y)
