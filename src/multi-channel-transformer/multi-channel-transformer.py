import torch
from torch import nn
import copy


class FeedForward(nn.Module):
    def __init__(self, input_dimension, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, 4 * input_dimension),
            nn.ReLU(),
            nn.Linear(4 * input_dimension, input_dimension),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dimension, number_of_channels, number_of_heads):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        # same as regular TransformerEncoderLayer, but the values and keys depend on the output of the other channels
        self.sa = nn.MultiheadAttention(input_dimension, number_of_heads)
        self.ffwd = FeedForward(input_dimension)
        self.agg = nn.ModuleList(
            [copy.deepcopy(nn.Linear(input_dimension, input_dimension, bias=False)) for _ in
             range(number_of_channels - 1)])
        self.ln = nn.LayerNorm(input_dimension)

    def forward(self, x, other_channels_output):
        x_agg = torch.zeros(x.shape)
        for i, x_i in enumerate(other_channels_output):
            x_agg += self.agg[i](x_i)
        x = x + self.sa(x, x_agg, x_agg)
        x = x + self.ffwd(self.ln2(x))
        return x


class MultiChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, number_of_channels, number_of_heads, input_dimension):
        super(MultiChannelTransformerEncoderLayer, self).__init__()
        # first for each channel apply regular transformer encoder layer
        # then for each channel apply cross-channel transformer encoder layer
        self.channel_wise_self_encoder_layer = nn.ModuleList(
            [copy.deepcopy(nn.TransformerEncoderLayer(d_model=input_dimension, nhead=number_of_heads)) for _ in
             range(number_of_channels)])
        self.cross_channel_encoder_layer = nn.ModuleList(
            [copy.deepcopy(CrossChannelTransformerEncoderLayer(input_dimension, number_of_channels, number_of_heads))
             for _ in
             range(number_of_channels)])

    def forward(self, x):
        x = x.permute(2, 0, 1)  # (channel, batch_size, seq_len) REVIEW: is this the correct permutation?
        for i in range(len(self.channel_wise_self_encoder_layer)):
            x[i] = self.channel_wise_self_encoder_layer[i](x[i])
        for i in range(len(self.cross_channel_encoder_layer)):
            x[i] = self.cross_channel_encoder_layer[i](x[i], [h for i, h in enumerate(x) if i != i])
        return x.permute(1, 2, 0)  # (batch_size, seq_len, channel)


class MultiChannelTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers, input_dimension):
        super(MultiChannelTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(number_of_layers)])
        self.norm = nn.LayerNorm(input_dimension)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class MultiChannelTransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, number_of_channels, number_of_layers, number_of_heads):
        super(MultiChannelTransformerClassifier, self).__init__()
        # self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder = MultiChannelTransformerEncoder(
            MultiChannelTransformerEncoderLayer(number_of_channels, number_of_heads),
            number_of_layers
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # embedded = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        output = self.encoder(x)
        output = self.dropout(output)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        output = self.fc(output[:, -1, :])  # Take the last hidden state
        return output
