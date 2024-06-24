import torch
import matplotlib.pyplot as plt
from mnist_model import create_model


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"device:{device}")

model = create_model(device)
model.load_state_dict(torch.load("vtf_mnist.pth"))

n = 16

model.train(False)
z1 = torch.linspace(-0, 1, n)
z2 = torch.zeros_like(z1) + 2
z = torch.stack([z1, z2], dim=1).view(n, 1, -1).to(device)
tgt = torch.zeros((n, 28 * 28, model.d_output), device=device)
output = model.decode(tgt, z)
output = torch.sigmoid(output)

fig, axs = plt.subplots(1, n, figsize=(n, 1))
for i in range(n):
    axs[i].imshow(output[i].view(28, 28).detach().cpu().numpy(), cmap="gray")
    axs[i].axis("off")

plt.show()
