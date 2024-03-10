import torch
from lightning.pytorch.demos.transformer import PositionalEncoding
from torch import nn


class TransformerModel(nn.Module):
    def __init__(
        self, input_dim, embed_dim, output_dim, num_layers, num_heads, dropout
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
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
