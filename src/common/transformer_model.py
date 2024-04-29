import torch
from lightning.pytorch.demos.transformer import PositionalEncoding
from torch import nn

from src.common.feed_forward import FeedForward


class TransformerModel(nn.Module):
    def __init__(
            self, input_dim, embed_dim, output_dim, num_layers, num_heads, dropout, head_hidden_layers,
            head_hidden_dimension,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout, dim_feedforward=embed_dim * 4
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.ffw = nn.Linear(
            in_features=embed_dim,
            out_features=output_dim,
        )

    def forward(self, x):
        # x is (batch_size, seq_len, input_dim)s

        x = x.flatten(-2)

        x = self.embedding(x)

        classification_token = torch.ones_like(x[:, :1, :])
        x = torch.cat((x, classification_token), dim=1)

        x = self.pos_encoding(x)

        x = self.encoder(x)
        classification_output = x[:, -1, :]

        return self.ffw(classification_output)
