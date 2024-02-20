import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.multi_channel_transformer.multi_channel_transformer import MultiChannelTransformerClassifier

data_directory = "data/challenge-2019/processed_samples"  # Replace with the actual path to your data directory
data = {}
targets = {}
non_sepsis_samples = 1_000  # Number of samples to load
sepsis_samples = 1_000  # Number of samples to load

for filename in tqdm(
        os.listdir("data/challenge-2019/processed_samples/sepsis")[:sepsis_samples], desc="Loading data"
):
    if filename.endswith(".csv"):
        patient_id, sequence_id = filename.split("_")[:2]
        patient_sample = pd.read_csv(os.path.join(data_directory, filename))
        last_row = patient_sample.iloc[-1]
        sepsis_label = last_row["SepsisLabel"]
        targets[(patient_id, sequence_id)] = 1 if sepsis_label == 1 else 0
        patient_sample.drop(columns=["SepsisLabel"], inplace=True)
        patient_sample = patient_sample.iloc[1:]  # Drop the first row
        data[(patient_id, sequence_id)] = patient_sample.fillna(0)

for filename in tqdm(
        os.listdir("data/challenge-2019/processed_samples/no-sepsis")[:non_sepsis_samples],
        desc="Loading data",
):
    if filename.endswith(".csv"):
        patient_id, sequence_id = filename.split("_")[:2]
        patient_sample = pd.read_csv(os.path.join(data_directory, filename))
        last_row = patient_sample.iloc[-1]
        sepsis_label = last_row["SepsisLabel"]
        targets[(patient_id, sequence_id)] = 1 if sepsis_label == 1 else 0
        patient_sample.drop(columns=["SepsisLabel"], inplace=True)
        patient_sample = patient_sample.iloc[1:]  # Drop the first row
        data[(patient_id, sequence_id)] = patient_sample.fillna(0)

number_of_sepsis_labels: int = np.sum(list(targets.values()))
pos_weight = torch.tensor(
    (len(targets) - number_of_sepsis_labels) / number_of_sepsis_labels
)
print(
    f"Total number of sepsis occurances: {number_of_sepsis_labels}/{len(targets)} = {(number_of_sepsis_labels / len(targets)) * 100}%\n"
)



# Define hyperparameters
device = "mps"
input_dim = 40  # Replace with the actual input dimension
output_dim = 1  # Replace with the actual number of output classes
learning_rate = 1e-4
weight_decay = 0.01
hidden_dim = 256
num_layers = 3
num_heads = 1
batch_size = 10
num_epochs = 1_000
eval_interval = 100
eval_iters = 100
early_stop_iters = 10
torch.manual_seed(6469)

# Create the model
model = MultiChannelTransformerClassifier(input_dim,1, output_dim, 40, num_layers, num_heads)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Create dataset and dataloader
class SepsisDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


dataset = SepsisDataset(
    np.array([x.to_numpy() for x in list(data.values())]),
    np.array(list(targets.values())),
)
train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):
            dl = dataloader if (split == "train") else dataloader_val
            inputs, labels = next(iter(dl))
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(
                torch.float32
            ).to(device)
            outputs = model(inputs)
            losses[k] = F.binary_cross_entropy_with_logits(
                torch.squeeze(outputs, -1),
                labels,
                pos_weight=pos_weight
            ).item()
            accuracies[k] = (
                ((outputs > 0.5) == labels).float().mean().item()
            )
        out[split] = losses.mean()
        out[split + "_acc"] = accuracies.mean()
    model.train()
    return out

loss_didnt_improve = 0
last_best_loss = float("inf")
# Training loop
for epoch in range(num_epochs):
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        if losses["val"] < last_best_loss:
            last_best_loss = losses["val"]
            loss_didnt_improve = 0
        else:
            loss_didnt_improve += 1
        print(
            f"step {epoch}: train loss {losses['train']:.4f} acc {losses['train_acc']:.4f}, val loss {losses['val']:.4f} acc {losses['val_acc']:.4f}"
        )
        if loss_didnt_improve > early_stop_iters:
            break
    inputs, labels = next(iter(dataloader))
    # for inputs, labels in dataloader:
    inputs = inputs.to(torch.float32).to(device)
    labels = labels.to(torch.float32).to(device)
    optimizer.zero_grad()
    # i = torch.randn(1, 2, requires_grad=True)
    # t = torch.empty(1, dtype=torch.long).random_(2)
    # e = F.cross_entropy(i, t)
    outputs = model(inputs)
    # loss = F.cross_entropy(outputs, labels.to(torch.long))
    loss = F.binary_cross_entropy_with_logits(torch.squeeze(outputs, -1), labels, pos_weight=pos_weight)
    # loss.requires_grad = True
    # optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the trained model
torch.save(model.state_dict(), "multi_transformer_model_large-1000_l_3e4_adamw_val_stop.pth")