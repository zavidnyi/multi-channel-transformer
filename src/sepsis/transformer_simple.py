import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F

data_directory = 'data/challenge-2019/processed_samples'  # Replace with the actual path to your data directory
data = {}
targets = {}
non_sepsis_samples = 30_000  # Number of samples to load

for filename in tqdm(os.listdir('data/challenge-2019/processed_samples/sepsis'), desc="Loading data"):
    if filename.endswith('.csv'):
        patient_id, sequence_id = filename.split('_')[:2]
        patient_sample = pd.read_csv(os.path.join(data_directory, filename))
        last_row = patient_sample.iloc[-1]
        sepsis_label = last_row['SepsisLabel']
        targets[(patient_id, sequence_id)] = 1 if sepsis_label == 1 else 0
        patient_sample.drop(columns=['SepsisLabel'], inplace=True)
        patient_sample = patient_sample.iloc[1:]  # Drop the first row
        data[(patient_id, sequence_id)] = patient_sample.fillna(0)

for filename in tqdm(os.listdir('data/challenge-2019/processed_samples/no-sepsis')[:non_sepsis_samples], desc="Loading data"):
    if filename.endswith('.csv'):
        patient_id, sequence_id = filename.split('_')[:2]
        patient_sample = pd.read_csv(os.path.join(data_directory, filename))
        last_row = patient_sample.iloc[-1]
        sepsis_label = last_row['SepsisLabel']
        targets[(patient_id, sequence_id)] = 1 if sepsis_label == 1 else 0
        patient_sample.drop(columns=['SepsisLabel'], inplace=True)
        patient_sample = patient_sample.iloc[1:]  # Drop the first row
        data[(patient_id, sequence_id)] = patient_sample.fillna(0)

number_of_sepsis_labels: int = np.sum(list(targets.values()))
pos_weight = torch.tensor((len(targets) - number_of_sepsis_labels) / number_of_sepsis_labels)
print(
    f"Total number of sepsis occurances: {number_of_sepsis_labels}/{len(targets)} = {(number_of_sepsis_labels / len(targets)) * 100}%\n")


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        # self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
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


# Define hyperparameters
device = 'mps'
input_dim = 40  # Replace with the actual input dimension
output_dim = 1  # Replace with the actual number of output classes
learning_rate = 3e-4
hidden_dim = 256
num_layers = 3
num_heads = 4
batch_size = 128
num_epochs = 1000
eval_interval = 10
eval_iters = 100
torch.manual_seed(6469)

# Create the model
model = TransformerModel(input_dim, output_dim, hidden_dim, num_layers, num_heads)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


# Create dataset and dataloader
class SepsisDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


dataset = SepsisDataset(np.array([x.to_numpy() for x in list(data.values())]), np.array(list(targets.values())))
train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            dl = dataloader if (split == "train") else dataloader_val
            inputs, labels = next(iter(dl))
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            outputs = model(inputs)
            losses[k] = F.binary_cross_entropy_with_logits(torch.squeeze(outputs, -1), labels,
                                                           pos_weight=pos_weight).item()
        out[split] = losses.mean()
    model.train()
    return out


# Training loop
for epoch in range(num_epochs):
    # inputs, labels = next(iter(dataloader))
    for inputs, labels in dataloader:
        inputs = inputs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        # optimizer.zero_grad()
        # i = torch.randn(1, 2, requires_grad=True)
        # t = torch.empty(1, dtype=torch.long).random_(2)
        # e = F.cross_entropy(i, t)
        outputs = model(inputs)
        # loss = F.cross_entropy(outputs, labels.to(torch.long))
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(outputs, -1), labels, pos_weight=pos_weight)
        # loss.requires_grad = True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'transformer_model.pth')
