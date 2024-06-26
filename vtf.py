import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import LayerNorm
from torch.distributions import kl, MultivariateNormal


class VariationalTransformer(nn.Module):

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
        self.input_embedding = nn.Linear(d_input, d_embed, device=device)
        encoder_layers = TransformerEncoderLayer(
            d_embed, nhead, dim_feedforward, dropout, batch_first=batch_first, device=device
        )
        encoder_norm = LayerNorm(d_embed, device=device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        self.encoder_mu = nn.Linear(d_embed, d_latent, device=device)
        self.encoder_ln_var = nn.Linear(d_embed, d_latent, device=device)
        self.soft_plus = nn.Softplus()

        # Decoder
        self.decoder_z = nn.Linear(d_latent, d_embed, device=device)
        self.target_embedding = nn.Linear(d_output, d_embed, device=device)
        decoder_layers = TransformerDecoderLayer(
            d_embed, nhead, dim_feedforward, dropout, batch_first=batch_first, device=device
        )
        decoder_norm = LayerNorm(d_embed, device=device)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)
        self.output_converter = nn.Linear(d_embed, d_output, device=device)

        # Distribution Loss
        self.kl_func = kl.kl_divergence
        self.kl_loss = 0.0

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embedding.bias.data.zero_().to(self.device)
        self.input_embedding.weight.data.uniform_(-initrange, initrange).to(self.device)
        self.encoder_mu.bias.data.zero_().to(self.device)
        self.encoder_mu.weight.data.uniform_(-initrange, initrange).to(self.device)
        self.encoder_ln_var.bias.data.zero_().to(self.device)
        self.encoder_ln_var.weight.data.uniform_(-initrange, initrange).to(self.device)
        self.target_embedding.bias.data.zero_().to(self.device)
        self.target_embedding.weight.data.uniform_(-initrange, initrange).to(self.device)
        self.output_converter.bias.data.zero_().to(self.device)
        self.output_converter.weight.data.uniform_(-initrange, initrange).to(self.device)

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor = None) -> Tensor:
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_input or tgt.size(-1) != self.d_output:
            raise RuntimeError("the feature number of src and tgt must be equal to d_input and d_output")

        dist = self.encode(src)
        output = self.decode(tgt, dist.rsample(), tgt_mask=tgt_mask)
        return output

    def encode(self, src: Tensor, eps: float = 1e-8) -> MultivariateNormal:
        emb = self.input_embedding(src)
        memory = self.transformer_encoder(emb)
        mu = self.encoder_mu(memory)
        ln_var = self.encoder_ln_var(memory)

        scale = self.soft_plus(ln_var) + eps
        scale_tril = torch.diag_embed(scale)
        dist = MultivariateNormal(mu, scale_tril=scale_tril)
        z = dist.rsample()
        std_normal = MultivariateNormal(
            torch.zeros_like(z, device=self.device), scale_tril=torch.eye(z.size(-1), device=self.device)
        )

        self.kl_loss = self.kl_func(dist, std_normal).mean()
        return dist

    def decode(self, tgt: Tensor, z: Tensor, tgt_mask: Tensor = None) -> Tensor:
        emb = self.target_embedding(tgt)
        memory = self.decoder_z(z)
        output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
        output = self.output_converter(output)
        return output

    def get_tgt_mask(self, size: int) -> Tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask


if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device:{device}")
    model = VariationalTransformer(
        d_input=4,
        d_output=4,
        d_embed=16,
        d_latent=32,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        batch_first=True,
        device=device,
    )
    src = torch.randn((32, 10, 4), device=device)
    tgt = torch.randn((32, 10, 4), device=device)
    output = model(src, tgt)
    print(output.size())
    print(model.kl_loss)

    z = torch.randn((1, 10, 32), device=device)
    max_len = 10
    output = model.generate(z, max_len)
    print(output.size())
    print(output)
