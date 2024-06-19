import torch
from torchvision import datasets
from torchvision.transforms import v2

from vtf import VariationalTransformer


batch_size = 128
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(28, 28) - 0.5),
    ]
)

train_data = datasets.MNIST("data/MNIST", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data/MNIST", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"device:{device}")

model = VariationalTransformer(
    d_input=28,
    d_output=28,
    d_embed=128,
    d_latent=2,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=1024,
    batch_first=True,
    device=device,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
rec_loss_func = torch.nn.MSELoss()


def train(model, data_loader, optimizer, rec_loss_func, prev_updates=0):
    model.train(True)

    for batch_idx, (data, _) in enumerate(train_loader):
        n_upd = prev_updates + batch_idx

        data = data.to(device)
        target = data.clone()

        optimizer.zero_grad()
        output = model(data, target)
        rec_loss = rec_loss_func(output, target)
        loss = rec_loss + model.kl_loss
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if n_upd % 100 == 0:
            print(f"Step:{n_upd:,} (N samples: {n_upd * batch_size:,}) Loss: {loss.item():.4f} (Recon: {rec_loss.item():.4f}, KL:{model.kl_loss.item():.4f})")  # fmt: skip
    return prev_updates + len(train_loader)


num_epochs = 100

prev_updates = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    prev_updates = train(model, train_loader, optimizer, rec_loss_func, prev_updates)

torch.save(model.state_dict(), "vtf_mnist.pth")
