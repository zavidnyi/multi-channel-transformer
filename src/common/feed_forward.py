from torch import nn


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
