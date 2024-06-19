import torch
from torch.optim import Adam
from torch.nn import MSELoss

from tf import VariationalTransformer

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
optimizer = Adam(model.parameters(), lr=0.001)
rec_loss = MSELoss()

sin_data = torch.sin(torch.arange(0, 100, 0.1)).unsqueeze(-1).to(device)
cos_data = torch.cos(torch.arange(0, 100, 0.1)).unsqueeze(-1).to(device)

model.train(True)
for epoch in range(100):
    optimizer.zero_grad()
    output = model(sin_data, cos_data)
    loss = rec_loss(output, cos_data)
    loss += model.kl_loss
    loss.backward()
    optimizer.step()
    print(f"epoch:{epoch}, loss:{loss.item()}, kl_loss:{model.kl_loss.item()}")

torch.save(model.state_dict(), "tf_test.pth")

model.train(False)
z = torch.randn((1, 1, 16), device=device)
max_len = 100
output = model.generate(z, max_len)
for y in output:
    print(y)
