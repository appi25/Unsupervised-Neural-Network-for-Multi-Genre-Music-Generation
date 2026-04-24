"""Latent-space sampling for AE and VAE music generation."""
import torch
import numpy as np


def sample_from_vae(model, n_samples=1, device='cpu'):
    """Sample z ~ N(0, I) and decode to piano-roll logits."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(device)
        z_expanded = model.decoder_input(z).unsqueeze(1).repeat(1, model.seq_len, 1)
        recon_out, _ = model.decoder_lstm(z_expanded)
        logits = model.output_layer(recon_out)
    return torch.sigmoid(logits).cpu().numpy()


def interpolate_latent(model, z1, z2, n_steps=8, device='cpu'):
    """Linear interpolation between two latent vectors.

    z_alpha = (1 - alpha) * z1 + alpha * z2, alpha in [0, 1]
    """
    model.eval()
    results = []
    for alpha in np.linspace(0, 1, n_steps):
        z = (1 - alpha) * z1 + alpha * z2
        z_exp = model.decoder_input(z.to(device)).unsqueeze(1).repeat(1, model.seq_len, 1)
        out, _ = model.decoder_lstm(z_exp)
        logits = model.output_layer(out)
        results.append(torch.sigmoid(logits).cpu().numpy())
    return results
