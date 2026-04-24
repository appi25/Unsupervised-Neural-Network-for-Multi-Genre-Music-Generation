"""Task 1: LSTM Autoencoder Training (Algorithm 1).

Loss: L_AE = sum_t ||x_t - x_hat_t||^2

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


class MusicDataset(Dataset):
    """Loads .npy piano-roll batches from disk."""
    def __init__(self, folder):
        self.files = [os.path.join(folder, f)
                      for f in os.listdir(folder) if f.endswith('.npy')]
        self.data = []
        for f in self.files[:5]:
            self.data.append(np.load(f))
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()


def train_ae(model, train_dir, device, epochs=5, lr=1e-3, batch_size=64):
    """Algorithm 1: Train LSTM Autoencoder with BCE loss."""
    dataset = MusicDataset(train_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        losses.append(avg)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg:.4f}")
    return losses
