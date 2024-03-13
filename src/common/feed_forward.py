from torch import nn


class FeedForward(nn.Module):
    def __init__(
            self, input_dimension, hidden_dim, hidden_layers, output_dim, dropout=0.5
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dimension, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
