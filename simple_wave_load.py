import numpy as np
import torch
import matplotlib.pyplot as plt
from simple_wave_model import create_model


def generate(model, z, max_len):
    model.train(False)

    y_input = torch.zeros((z.size(0), 1, model.d_output), device=device)
    for _ in range(max_len):
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        gen = model.decode(y_input, z, tgt_mask)
        gen = torch.tanh(gen)

        next_item = gen[:, -1:]
        y_input = torch.cat([y_input, next_item], dim=1)

    return y_input


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"device:{device}")

model = create_model(device)
model.load_state_dict(torch.load("wave_test.pth"))

n = 16
max_len = 100

z1 = torch.linspace(-1, 1, n)
z2 = torch.zeros_like(z1) + 2
z = torch.stack([z1, z2], dim=1).view(n, -1, 2).to(device)
output = generate(model, z, max_len)

fig, axs = plt.subplots(1, n, figsize=(n, 1))
x = np.arange(0, max_len + 1, 1)
for i in range(n):
    axs[i].plot(x, output[i].view(-1).detach().cpu().numpy(), color="blue")

plt.show()
