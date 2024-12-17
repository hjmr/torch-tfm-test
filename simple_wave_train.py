import argparse
import random
import numpy as np
import torch
from scipy import signal

from simple_wave_model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="wave_test.pth")
    parser.add_argument("--num_epochs", type=int, default=500)
    return parser.parse_args()


def generate_data(source: torch.tensor, target: torch.tensor, unit_num: int, unit_length: int, device: torch.device):
    inp_data = []
    tgt_data = []

    for _ in range(unit_num):
        idx = random.randint(0, len(source) - unit_length - 1)
        inp_data.append(source[idx : idx + unit_length].tolist())
        tgt_data.append(target[idx : idx + unit_length].tolist())
    inp_data = torch.tensor(inp_data, dtype=torch.float32, device=device).reshape(unit_num, unit_length, 1)
    tgt_data = torch.tensor(tgt_data, dtype=torch.float32, device=device).reshape(unit_num, unit_length, 1)
    return inp_data, tgt_data


def main(num_epochs: int, model_file: str, device: torch.device):
    # fmt: off
    sin_data = torch.sin(torch.arange(0, 12, 0.05)).unsqueeze(-1).to(device)
    square_data = torch.tensor(signal.square(torch.arange(0, 12, 0.05)), dtype=torch.float32).unsqueeze(-1).to(device)
    sawtooth_data = torch.tensor(signal.sawtooth(torch.arange(0, 12, 0.05)), dtype=torch.float32).unsqueeze(-1).to(device)
    # fmt: on

    sin_inp, sin_tgt = generate_data(sin_data, sin_data, 1000, 50, device)
    squ_inp, squ_tgt = generate_data(square_data, square_data, 1000, 50, device)
    saw_inp, saw_tgt = generate_data(sawtooth_data, sawtooth_data, 1000, 50, device)

    input = torch.cat([sin_inp, squ_inp, saw_inp], dim=0)
    target = torch.cat([sin_tgt, squ_tgt, saw_tgt], dim=0)

    dataset = torch.utils.data.TensorDataset(input, target)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    model = create_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    rec_loss_func = torch.nn.MSELoss()

    model.train(True)
    for epoch in range(num_epochs):
        for inp, tgt in train_loader:
            optimizer.zero_grad()
            tgt_inp = tgt[:, :-1]
            tgt_tgt = tgt[:, 1:]
            tgt_mask = model.get_tgt_mask(tgt_inp.size(1)).to(device)

            output = model(inp, tgt_inp, tgt_mask)
            output = torch.tanh(output)
            rec_loss = rec_loss_func(output, tgt_tgt)
            loss = rec_loss + model.kl_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"epoch:{epoch}, loss:{rec_loss.item()}, kl_loss:{model.kl_loss.item()}")

    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device:{device}, model:{args.model}, num_epochs:{args.num_epochs}")
    main(args.num_epochs, args.model, device)
