import argparse
import torch
from torchvision import datasets
from torchvision.transforms import v2

from mnist_model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="vtf_mnist.pth")
    parser.add_argument("--num_epochs", type=int, default=50)
    return parser.parse_args()


def train(model, optimizer, rec_loss, data_loader, device: torch.device, prev_updates=0):
    model.train(True)

    for batch_idx, (data, _) in enumerate(data_loader):
        n_upd = prev_updates + batch_idx

        data = data.to(device)
        dummy = torch.zeros(data.size(), device=device)
        target = data.clone()

        optimizer.zero_grad()
        output = model(data, dummy)
        output = torch.sigmoid(output)
        rec_loss = rec_loss(output, target)
        loss = rec_loss + model.kl_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if n_upd % 100 == 0:
            print(f"Step:{n_upd:,} (N samples: {n_upd * data_loader.batch_size:,}) Loss: {loss.item():.4f} (Recon: {rec_loss.item():.  4f}, KL:{model.kl_loss.item():.4f})")  # fmt: skip
    return prev_updates + len(data_loader)


def main(num_epochs: int, model_file: str, device: torch.device):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x.view(-1, 1)),
        ]
    )

    train_data = datasets.MNIST("data/MNIST", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data/MNIST", train=False, download=True, transform=transform)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = create_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    rec_loss = torch.nn.MSELoss()
    # rec_loss = torch.nn.BCELoss()

    prev_updates = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        prev_updates = train(model, optimizer, rec_loss, train_loader, device, prev_updates)

    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device:{device}, model:{args.model}, num_epochs:{args.num_epochs}")
    main(args.num_epochs, args.model, device)
