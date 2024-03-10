from torch import nn


class FeedForward(nn.Module):
    def __init__(self, input_dimension, output_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, input_dimension // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dimension // 3, input_dimension // 3),
            nn.ReLU(),
            nn.Linear(input_dimension // 3, output_dim),
        )

    def forward(self, x):
        return self.net(x)
