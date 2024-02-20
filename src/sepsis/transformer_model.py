import torch
from torch import nn

from src.common.feed_forward import FeedForward


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        # self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.ffw = FeedForward(input_dim, output_dim)

    def forward(self, x):
        # x is (batch_size, seq_len, input_dim)
        # embedded = self.embedding(x)
        # x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        x = self.encoder(x)
        # output = self.dropout(output)
        # output = output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        # output = torch.mean(output, dim=0)  # Global average pooling
        x = self.ffw(x)  # Take the last hidden state
        # return torch.sigmoid(torch.squeeze(output, -1)[...,-1])
        return torch.squeeze(x, -1).mean(dim=1)
