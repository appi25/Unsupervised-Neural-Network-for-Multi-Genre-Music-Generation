"""Task 1: LSTM Autoencoder for single-genre music generation.

Architecture (from project specification):
    Encoder: z = f_phi(X)
    Decoder: X_hat = g_theta(z)
    Loss: L_AE = sum_t ||x_t - x_hat_t||^2
"""
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128, seq_len=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Encoder: X -> z
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> X_hat
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        z = self.to_latent(hidden[-1])
        # Decode
        z_expanded = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        recon_out, _ = self.decoder_lstm(z_expanded)
        x_hat = torch.sigmoid(self.output_layer(recon_out))
        return x_hat, z
