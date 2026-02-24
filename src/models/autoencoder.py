#C:\Users\ASUS\fintech-project\src\models\autoencoder.py
import torch
import torch.nn as nn


class MarketAutoencoder(nn.Module):
    """
    Market Risk Autoencoder
    -----------------------
    Purpose:
    - Learn latent RISK STRUCTURE of the market
    - NOT return forecasting
    - NOT alpha generation

    Input: risk-based features only
    Output: reconstruction + latent risk factors
    """

    def __init__(self, input_dim: int, latent_dim: int = 3):
        super().__init__()

        # =========================
        # Encoder: compress risk space
        # =========================
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )

        # =========================
        # Decoder: reconstruct risk signals
        # =========================
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        """
        Forward pass
        Returns:
        - x_hat: reconstructed risk features
        - z: latent risk representation
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
