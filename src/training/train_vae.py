"""Task 2: VAE Training with ELBO objective (Algorithm 2).

L_VAE = L_recon + beta * D_KL(q_phi(z|X) || p(z))
"""
import torch
import torch.nn.functional as F
import torch.optim as optim


def vae_loss_function(recon_x, x, mu, logvar, beta, criterion_recon):
    """ELBO = reconstruction + beta * KL divergence."""
    recon_loss = criterion_recon(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_vae(model, loader, device, epochs=20, lr=1e-3, beta=0.1):
    """Algorithm 2: Train VAE with reparameterization and KL loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_recon = torch.nn.BCEWithLogitsLoss(
        reduction='sum', pos_weight=torch.tensor([10.0]).to(device))
    model.train()

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(batch)
            loss = vae_loss_function(logits, batch, mu, logvar, beta, criterion_recon)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / batch.size(0)
        avg = total_loss / len(loader)
        losses.append(avg)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg:.4f}")
    return losses
