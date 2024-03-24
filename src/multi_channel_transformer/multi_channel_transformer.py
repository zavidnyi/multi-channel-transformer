import copy

import torch
from torch import nn

from src.common.feed_forward import FeedForward


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, number_of_channels, number_of_heads):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        # same as regular TransformerEncoderLayer, but the values and keys depend on the output of the other channels
        self.sa = nn.MultiheadAttention(
            input_dimension, number_of_heads, batch_first=True
        )
        self.ffwd = FeedForward(
            input_dimension=input_dimension,
            hidden_dim=input_dimension,
            hidden_layers=0,
            output_dim=input_dimension,
            dropout=0.1,
        )
        self.agg = nn.ModuleList(
            [
                copy.deepcopy(nn.Linear(input_dimension, input_dimension, bias=False))
                for _ in range(number_of_channels - 1)
            ]
        )
        self.ln2 = nn.LayerNorm(input_dimension)

    def forward(self, x, other_channels_output):
        x_agg = torch.zeros(x.shape).to(x.device)
        for i, x_i in enumerate(other_channels_output):
            x_agg += self.agg[i](x_i)
        x = x + self.sa(x, x_agg, x_agg, need_weights=False)[0]
        x = x + self.ln2(x)
        x = x + self.ffwd(x)
        return x


class MultiChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, number_of_channels, number_of_heads, channel_dimension):
        super(MultiChannelTransformerEncoderLayer, self).__init__()
        # first for each channel apply regular transformer encoder layer
        # then for each channel apply cross-channel transformer encoder layer
        self.channel_wise_self_encoder_layer = nn.ModuleList(
            [
                copy.deepcopy(
                    nn.TransformerEncoderLayer(
                        d_model=channel_dimension,
                        nhead=number_of_heads,
                        batch_first=True,
                        norm_first=True,
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
                        channel_dimension, number_of_channels, number_of_heads
                    )
                )
                for _ in range(number_of_channels)
            ]
        )

    # x is (batch_size, seq_len, channel, channel_dim)
    def forward(self, x):
        x = x.permute(
            2, 0, 1, 3
        )  # (channel, batch_size, seq_len, channel_dim) REVIEW: is this the correct permutation?
        x_clone = x.clone()
        x_clone_ = []
        for i in range(len(self.channel_wise_self_encoder_layer)):
            x_clone_.append(
                self.channel_wise_ln[i](self.channel_wise_self_encoder_layer[i](x[i]))
            )
        x_clone = torch.stack(x_clone_)

        x_ = []
        for i in range(len(self.cross_channel_encoder_layer)):
            x_.append(
                self.cross_channel_encoder_layer[i](
                    x[i], [h for i, h in enumerate(x_clone) if i != i]
                )
            )
        x = torch.stack(x_)
        x = x.permute(1, 2, 0, 3)
        return x


class MultiChannelTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers, input_dimension):
        super(MultiChannelTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(number_of_layers)]
        )
        self.norm = nn.LayerNorm(input_dimension)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class MultiChannelTransformerClassifier(nn.Module):
    def __init__(
        self,
        channel_dimension,
        channel_hidden_dimension,
        output_dim,
        number_of_channels,
        number_of_layers,
        number_of_heads,
        dropout=0.1,
        head_hidden_layers=2,
        head_hidden_dimension=512,
    ):
        super(MultiChannelTransformerClassifier, self).__init__()
        self.channel_wise_embedding = nn.ModuleList(
            [
                copy.deepcopy(
                    nn.Linear(channel_dimension, channel_hidden_dimension, bias=False)
                )
                for _ in range(number_of_channels)
            ]
        )
        self.encoder = MultiChannelTransformerEncoder(
            MultiChannelTransformerEncoderLayer(
                number_of_channels, number_of_heads, channel_hidden_dimension
            ),
            number_of_layers,
            channel_hidden_dimension,
        )
        channel_input_dim = channel_hidden_dimension * number_of_channels
        self.ffw = FeedForward(
            input_dimension=channel_input_dim,
            hidden_dim=head_hidden_dimension,
            hidden_layers=head_hidden_layers,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x):
        # x is (batch_size, seq_len, channels, channel_dim)
        if x.dim() == 3:
            # in case channel dimension is one, we need to remap it 4 dimension manually
            x = x.unsqueeze(-1)

        x = x.permute(2, 0, 1, 3)  # (channel, batch_size, seq_len, channel_dim)
        x_clone_ = []
        for i in range(len(self.channel_wise_embedding)):
            x_clone_.append(self.channel_wise_embedding[i](x[i]))
        x = torch.stack(x_clone_)

        x = x.permute(1, 2, 0, 3)  # (batch_size, seq_len, channel, channel_dim)

        classification_token = torch.ones_like(x[:, :1, :])
        x = torch.cat((classification_token, x), dim=1)

        x = self.encoder(x)
        x = x.squeeze(-1)
        x = x[:, 0, :] # Take the last hidden state
        x = x.flatten(1)
        x = self.ffw(x)
        x = x.squeeze(-1)
        return x
