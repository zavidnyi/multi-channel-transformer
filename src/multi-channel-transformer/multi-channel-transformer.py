from torch import nn


class CrossChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossChannelTransformerEncoderLayer, self).__init__()
        # same as regular TransformerEncoderLayer, but the values and keys depend on the output of the other channels

    def forward(self, x):
        pass


class MultiChannelTransformerEncoderLayer(nn.Module):
    def __init__(self, number_of_channels, number_of_heads):
        super(MultiChannelTransformerEncoderLayer, self).__init__()
        # first for each channel apply regular transformer encoder layer
        # then for each channel apply cross-channel transformer encoder layer

    def forward(self, x):
        pass


class MultiChannelTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super(MultiChannelTransformerEncoder, self).__init__()
        # same as regular Encoder, but uses custom encoder layer

    def forward(self, x):
        pass


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
