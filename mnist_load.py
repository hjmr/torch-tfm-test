import torch
import matplotlib.pyplot as plt
from vtf import VariationalTransformer


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"device:{device}")

model = VariationalTransformer(
    d_input=784,
    d_output=784,
    d_embed=128,
    d_latent=2,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    batch_first=True,
    device=device,
)
model.load_state_dict(torch.load("vtf_mnist.pth"))

n = 16

model.train(False)
z1 = torch.linspace(-0, 1, n)
z2 = torch.zeros_like(z1) + 2
z = torch.stack([z1, z2], dim=1).view(n, 1, -1).to(device)
output = model.generate(z, 1)
output = torch.sigmoid(output)

fig, axs = plt.subplots(1, n, figsize=(n, 1))
for i in range(n):
    axs[i].imshow(output[i].view(28, 28).detach().cpu().numpy(), cmap="gray")
    axs[i].axis("off")

plt.show()
