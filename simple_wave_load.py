import numpy as np
import torch
import matplotlib.pyplot as plt
from simple_wave_model import create_model


def generate(model, start_y, z, max_len):
    model.train(False)

    y_input = start_y.clone()
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

m = 10
n = 10
max_len = 100

z1 = torch.linspace(-5, 5, n)
z2 = torch.zeros_like(z1) + 1.0
z = torch.stack([z1, z2], dim=1).view(n, -1, 2).to(device)

y = torch.linspace(-1, 1, m)

output = []
for i in range(m):
    start_y = torch.zeros(n, 1, 1).fill_(y[i]).to(device)
    output.append(generate(model, start_y, z, max_len))

fig, axs = plt.subplots(m, n, tight_layout=True, figsize=(m, n))
x = np.arange(0, max_len + 1, 1)
for i in range(m):
    for j in range(n):
        axs[i, j].plot(x, output[i][j].view(-1).detach().cpu().numpy(), color="blue")
        axs[i, j].tick_params(labelsize=4)
        axs[i, j].set_title(f"y:{y[i]:.2f}, z1:{z1[j]:.2f}", fontsize=6)

plt.show()
