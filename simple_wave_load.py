import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from simple_wave_model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="wave_test.pth")
    return parser.parse_args()


def main(model_file: str, device: torch.device):

    model = create_model(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.train(False)

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
        output.append(model.generate(start_y, z, max_len, device))

    fig, axs = plt.subplots(m, n, tight_layout=True, figsize=(m, n))
    x = np.arange(0, max_len + 1, 1)
    for i in range(m):
        for j in range(n):
            axs[i, j].plot(x, output[i][j].view(-1).detach().cpu().numpy(), color="blue")
            axs[i, j].tick_params(labelsize=4)
            axs[i, j].set_title(f"y:{y[i]:.2f}, z1:{z1[j]:.2f}", fontsize=6)

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device:{device}, model:{args.model}")
    main(args.model, device)
