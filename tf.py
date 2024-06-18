import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import LayerNorm
from torch.distributions import kl, MultivariateNormal


class TransformerModel(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_embed: int,
        d_latent: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.5,
        batch_first: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_input = d_input
        self.d_embed = d_embed
        self.d_output = d_output
        self.nhead = nhead
        self.batch_first = batch_first
        self.device = device

        # Encoder
        self.input_converter = nn.Linear(d_input, d_embed)
        encoder_layers = TransformerEncoderLayer(
            d_embed, nhead, dim_feedforward, dropout, batch_first=batch_first, device=device
        )
        encoder_norm = LayerNorm(d_embed)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        self.encoder_mu = nn.Linear(d_embed, d_latent)
        self.encoder_ln_var = nn.Linear(d_embed, d_latent)
        self.soft_plus = nn.Softplus()

        # Decoder
        self.decoder_z = nn.Linear(d_latent, d_embed)
        self.target_converter = nn.Linear(d_output, d_embed)
        decoder_layers = TransformerDecoderLayer(
            d_embed, nhead, dim_feedforward, dropout, batch_first=batch_first, device=device
        )
        decoder_norm = LayerNorm(d_embed)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)
        self.output_converter = nn.Linear(d_embed, d_output)

        # Distribution Loss
        # self.kl_func = nn.KLDivLoss(reduction="batchmean")
        self.kl_func = kl.kl_divergence
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
        self.target_converter.bias.data.zero_()
        self.target_converter.weight.data.uniform_(-initrange, initrange)
        self.output_converter.bias.data.zero_()
        self.output_converter.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_input or tgt.size(-1) != self.d_output:
            raise RuntimeError("the feature number of src and tgt must be equal to d_input and d_output")

        dist = self.encode(src)
        output = self.decode(tgt, dist.rsample())
        return output

    def encode(self, src: Tensor, eps: float = 1e-8) -> MultivariateNormal:
        emb = self.input_converter(src)
        memory = self.transformer_encoder(emb)
        mu = self.encoder_mu(memory)
        ln_var = self.encoder_ln_var(memory)

        scale = self.soft_plus(ln_var) + eps
        scale_tril = torch.diag_embed(scale)
        dist = MultivariateNormal(mu, scale_tril=scale_tril)
        z = dist.rsample()
        std_normal = MultivariateNormal(torch.zeros_like(z), scale_tril=torch.eye(z.size(-1)))

        self.kl_loss = self.kl_func(dist, std_normal).mean()
        return dist

    def decode(self, tgt: Tensor, z: Tensor) -> Tensor:
        emb = self.target_converter(tgt)
        memory = self.decoder_z(z)
        output = self.transformer_decoder(emb, memory)
        output = self.output_converter(output)
        return output

    def generate(self, z: Tensor, max_len: int) -> Tensor:
        tgt = torch.zeros((self.d_output, max_len, self.d_output), device=self.device)
        tgt[:, 0, :] = self.decoder_z(z)
        for t in range(1, max_len):
            output = self.transformer_decoder(tgt[:, :t, :], tgt[:, : t - 1, :])
            output = self.output_converter(output)
            tgt[:, t, :] = output[:, -1, :]
        return tgt


if __name__ == "__main__":
    model = TransformerModel(
        d_input=16,
        d_output=24,
        d_embed=16,
        d_latent=16,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        batch_first=True,
    )
    src = torch.rand((32, 10, 16))
    tgt = torch.rand((32, 20, 24))
    output = model(src, tgt)
    print(f"output:{output.size()}")
    print(f"kl_loss:{model.kl_loss}")
