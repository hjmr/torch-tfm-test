import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import LayerNorm


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_signal: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        d_latent: int,
        dropout: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_signal = d_signal
        self.d_model = d_model
        self.nhead = nhead
        self.device = device

        # Encoder
        self.input_converter = nn.Linear(d_signal, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, device=device)
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        self.encoder_mu = nn.Linear(d_model, d_latent)
        self.encoder_ln_var = nn.Linear(d_model, d_latent)

        # Decoder
        self.decoder_z = nn.Linear(d_latent, d_model)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, device=device)
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)
        self.output_converter = nn.Linear(d_model, d_signal)

        # Distribution Loss
        self.kl_func = nn.KLDivLoss(reduction="batchmean")
        self.kl_loss = 0.0

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_converter.bias.data.zero_()
        self.input_converter.weight.data.uniform_(-initrange, initrange)
        self.encoder_mu.bias.data.zero_()
        self.encoder_mu.weight.data.uniform_(-initrange, initrange)
        self.encoder_ln_var.bias.data.zero_()
        self.encoder_ln_var.weight.data.uniform_(-initrange, initrange)
        self.output_converter.bias.data.zero_()
        self.output_converter.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        is_batched = src.dim() == 3
        if src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_signal or tgt.size(-1) != self.d_signal:
            raise RuntimeError("the feature number of src and tgt must be equal to d_signal")

        mu, _ = self.encode(src)
        output = self.decode(tgt, mu)
        return output

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.input_converter(src)
        memory = self.transformer_encoder(emb)
        mu = self.encoder_mu(memory)
        ln_var = self.encoder_ln_var(memory)
        self.kl_loss = self.kl_func(mu, ln_var)
        return mu, ln_var

    def decode(self, tgt: Tensor, z: Tensor) -> Tensor:
        memory = self.decoder_z(z)
        emb = self.transformer_decoder(tgt, memory)
        output = self.output_converter(emb)
        return output

    def generate(self, z: Tensor, max_len: int) -> Tensor:
        tgt = torch.zeros((self.d_signal, max_len, self.d_signal), device=self.device)
        tgt[:, 0, :] = self.decoder_z(z)
        for t in range(1, max_len):
            output = self.transformer_decoder(tgt[:, :t, :], tgt[:, : t - 1, :])
            output = self.output_converter(output)
            tgt[:, t, :] = output[:, -1, :]
        return tgt


if __name__ == "__main__":
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out = transformer_decoder(tgt, memory)  # model = TransformerModel(16, 512, 8, 6, 6, 2048, 64)

    # src = torch.rand((10, 32, 16))
    # tgt = torch.rand((20, 32, 16))
    # output = model(src, tgt)
    # print(f"output:{output.size()}")
    # print(f"kl_loss:{model.kl_loss}")
