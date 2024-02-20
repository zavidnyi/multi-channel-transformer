#!/usr/bin/env python
import numpy as np
import torch

from src.sepsis.transformer_model import TransformerModel

device = "mps"
input_dim = 40  # Replace with the actual input dimension
output_dim = 1  # Replace with the actual number of output classes
learning_rate = 1e-4
weight_decay = 0.01
hidden_dim = 256
num_layers = 3
num_heads = 4
batch_size = 32
num_epochs = 5000
eval_interval = 500
eval_iters = 200
torch.manual_seed(6469)


def get_sepsis_score(data, model):
    data = torch.tensor(np.nan_to_num(data), dtype=torch.float32).to(device)
    prediction = torch.sigmoid(model(data.unsqueeze(0)))
    return prediction, prediction > 0.5


def load_sepsis_model():
    model = TransformerModel(input_dim, output_dim, hidden_dim, num_layers, num_heads)
    model.to(device)
    model.load_state_dict(torch.load('transformer_model_large-1000_l_3e4_adamw_val_stop.pth'))
    model.eval()
    return model
