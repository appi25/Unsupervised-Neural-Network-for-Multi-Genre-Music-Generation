"""Task 2: Variational Autoencoder for multi-genre music generation.

Architecture (from project specification):
    Encoder: q_phi(z|X) = N(mu(X), sigma(X))
    Sampling: z = mu + sigma * epsilon, epsilon ~ N(0, I)
    Loss: L_VAE = L_recon + beta * D_KL(q_phi(z|X) || p(z))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, latent_dim=128, seq_len=64):
        super(MusicVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: q_phi(z|X) = N(mu, sigma)
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                                     bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder: p_theta(X|z)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """z = mu + sigma * epsilon, epsilon ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        mu = self.fc_mu(hidden_cat)
        logvar = self.fc_logvar(hidden_cat)
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decode
        z_expanded = self.decoder_input(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        recon_out, _ = self.decoder_lstm(z_expanded)
        logits = self.output_layer(recon_out)
        return logits, mu, logvar
