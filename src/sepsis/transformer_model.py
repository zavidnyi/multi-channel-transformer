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
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout, dim_feedforward=embed_dim * 4
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.ffw = FeedForward(
            input_dimension=embed_dim,
            hidden_dim=head_hidden_dimension,
            output_dim=output_dim,
            hidden_layers=head_hidden_layers,
            dropout=dropout,
        )

    def forward(self, x):
        # x is (batch_size, seq_len, input_dim)
        # embedded = self.embedding(x)
        # x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        x = self.embedding(x)
        classification_token = torch.ones_like(x[:, :1, :])
        x = torch.cat((classification_token, x), dim=1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        classification_output = x[:, 0, :]
        # output = self.dropout(output)
        # output = output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        # output = torch.mean(output, dim=0)  # Global average pooling
        x = self.ffw(classification_output)  # Take the last hidden state
        # return torch.sigmoid(torch.squeeze(output, -1)[...,-1])
        return torch.squeeze(x, -1)
