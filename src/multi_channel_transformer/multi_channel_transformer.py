import copy

import torch
from lightning.pytorch.demos.transformer import PositionalEncoding
from torch import nn

from src.common.feed_forward import FeedForward


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, number_of_channels, number_of_heads, dropout):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        # same as regular TransformerEncoderLayer, but the values and keys depend on the output of the other channels
        self.sa = nn.MultiheadAttention(
            input_dimension,
            number_of_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ffwd = FeedForward(
            input_dimension=input_dimension,
            hidden_dim=input_dimension * 4,
            hidden_layers=0,
            output_dim=input_dimension,
            dropout=dropout,
        )
        self.agg = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(input_dimension), requires_grad = True)
                for _ in range(number_of_channels - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dimension)

    def forward(self, x, other_channels_output):
        x_agg = []
        for i, x_i in enumerate(other_channels_output):
            x_agg.append(torch.mul(x_i, self.agg[i]))
        x_agg = torch.stack(x_agg)
        x_agg = torch.sum(x_agg, dim=0)
        x = x + self.dropout(self.sa(x, x_agg, x_agg, need_weights=False)[0])
        x = self.ln(x)
        x = x + self.dropout(self.ffwd(x))
        return x


class MultiChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, number_of_channels, number_of_heads, channel_dimension, dropout):
        super(MultiChannelTransformerEncoderLayer, self).__init__()
        # first for each channel apply regular transformer encoder layer
        # then for each channel apply cross-channel transformer encoder layer
        self.channel_wise_self_encoder_layer = nn.ModuleList(
            [
                copy.deepcopy(
                    nn.TransformerEncoderLayer(
                        d_model=channel_dimension,
                        dim_feedforward=channel_dimension * 4,
                        nhead=number_of_heads,
                        batch_first=True,
                        norm_first=True,
                        dropout=dropout,
                    )
                )
                for _ in range(number_of_channels)
            ]
        )
        self.channel_wise_ln = nn.ModuleList(
            [
                copy.deepcopy(nn.LayerNorm(channel_dimension))
                for _ in range(number_of_channels)
            ]
        )
        self.cross_channel_encoder_layer = nn.ModuleList(
            [
                copy.deepcopy(
                    CrossChannelTransformerEncoderLayer(
                        channel_dimension, number_of_channels, number_of_heads, dropout
                    )
                )
                for _ in range(number_of_channels)
            ]
        )

    # x is (channel, batch_size, seq_len, channel_dim)
    def forward(self, x):
        x_clone = []
        for i in range(len(self.channel_wise_self_encoder_layer)):
            x_clone.append(
                self.channel_wise_ln[i](self.channel_wise_self_encoder_layer[i](x[i]))
            )
        x = torch.stack(x_clone)

        x_clone = []
        for i in range(len(self.cross_channel_encoder_layer)):
            x_clone.append(
                self.cross_channel_encoder_layer[i](
                    x[i], [h for j, h in enumerate(x) if i != j]
                )
            )
        x = torch.stack(x_clone)
        return x


class MultiChannelTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers, input_dimension):
        super(MultiChannelTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(number_of_layers)]
        )
        self.norm = nn.LayerNorm(input_dimension)

    def forward(self, x):
        x = x.permute(
            2, 0, 1, 3
        )  # (channel, batch_size, seq_len, channel_dim)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 2, 0, 3)  # (batch_size, seq_len, channel, channel_dim)
        x = x.flatten(-2)
        x = self.norm(x)
        return x


class MultiChannelTransformerClassifier(nn.Module):
    def __init__(
            self,
            vocabulary_size,
            embed_dim,
            output_dim,
            number_of_channels,
            number_of_layers,
            number_of_heads,
            dropout=0.1,
    ):
        super(MultiChannelTransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embed_dim
        )
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        channel_input_dim = embed_dim * number_of_channels
        self.encoder = MultiChannelTransformerEncoder(
            MultiChannelTransformerEncoderLayer(
                number_of_channels,
                number_of_heads,
                embed_dim,
                dropout=dropout,
            ),
            number_of_layers,
            input_dimension=channel_input_dim,
        )
        self.linear = nn.Linear(channel_input_dim, output_dim)

    def forward(self, x):
        # x is (batch_size, seq_len, channels, channel_dim)
        if x.dim() == 3:
            # in case channel dimension is one, we need to remap it 4 dimension manually
            x = x.unsqueeze(-1)

        x = self.embedding(x)
        x = x.flatten(-2)
        classification_token = torch.ones_like(x[:, :1, :])
        x = torch.cat((classification_token, x), dim=1)

        x = x.permute(
            2, 0, 1, 3
        )  # (channel, batch_size, seq_len, channel_dim)
        x_copy = []
        for i in range(x.size(0)):
            x_copy.append(self.positional_encoding(x[i]))
        x = torch.stack(x_copy)
        x = x.permute(1, 2, 0, 3)  # (batch_size, seq_len, channel, channel_dim)

        x = self.encoder(x)
        x = x[:, 0, :]  # Take the last hidden state
        x = x.flatten(1)
        x = self.linear(x)
        x = x.squeeze(-1)
        return x
