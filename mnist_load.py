import argparse
import torch
import matplotlib.pyplot as plt
from mnist_model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="vtf_mnist.pth")
    parser.add_argument("--num", type=int, default=16)
    return parser.parse_args()


def main(model_file: str, num: int, device: torch.device):
    model = create_model(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))

    model.train(False)
    z1 = torch.linspace(-0, 1, num)
    z2 = torch.zeros_like(z1) + 2
    z = torch.stack([z1, z2], dim=1).view(num, 1, -1).to(device)
    tgt = torch.zeros((num, 28 * 28, model.d_output), device=device)
    output = model.decode(tgt, z)
    output = torch.sigmoid(output)

    fig, axs = plt.subplots(1, num, figsize=(num, 1))
    for i in range(num):
        axs[i].imshow(output[i].view(28, 28).detach().cpu().numpy(), cmap="gray")
        axs[i].axis("off")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device:{device}, model:{args.model}")
    main(args.model, args.num, device)
