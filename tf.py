import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import LayerNorm


class TransformerModel(nn.Module):
    def __init__(
        self,
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
        self.d_model = d_model
        self.nhead = nhead
        self.device = device

        # Encoder
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

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        is_batched = src.dim() == 3
        if src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        output = self.decode(tgt, self.encode(src)[0])
        return output

    def encode(self, src: Tensor) -> Tensor:
        memory = self.transformer_encoder(src)
        mu = self.encoder_mu(memory)
        ln_var = self.encoder_ln_var(memory)
        return mu, ln_var

    def decode(self, tgt: Tensor, z: float) -> Tensor:
        memory = self.decoder_z(z)
        output = self.transformer_decoder(tgt, memory)
        return output

    def generate(self, z: float, max_len: int) -> Tensor:
        tgt = torch.zeros((self.d_model, max_len, self.d_model), device=self.device)
        tgt[:, 0, :] = self.decoder_z(z)
        for t in range(1, max_len):
            output = self.transformer_decoder(tgt[:, :t, :], tgt[:, : t - 1, :])
            tgt[:, t, :] = output[:, -1, :]
        return tgt
