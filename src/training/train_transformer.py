"""Task 3: Music Transformer Training (Algorithm 3).

L_TR = -sum_t log p_theta(x_t | x_<t)
Perplexity = exp(L_TR / T)

Historical training log (20 epochs, 21,005 samples):
    Epoch 1/20, Loss: 0.7641
    Epoch 10/20, Loss: 0.5712
    Epoch 20/20, Loss: 0.4967
    Final Perplexity: 1.4093
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """Sliding-window dataset over REMI token sequences."""
    def __init__(self, data, seq_len=128):
        self.samples = []
        for seq in data:
            if len(seq) > seq_len:
                for i in range(0, len(seq) - seq_len, seq_len // 2):
                    self.samples.append(seq[i : i + seq_len + 1])
            elif len(seq) > 5:
                padded = seq + [0] * (seq_len + 1 - len(seq))
                self.samples.append(padded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s[:-1]), torch.tensor(s[1:])


def train_transformer(model, tokenized_data, device, epochs=20, lr=1e-4,
                      batch_size=16, seq_len=128):
    """Algorithm 3: Autoregressive Transformer training."""
    dataset = TokenDataset(tokenized_data, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")
    return losses
