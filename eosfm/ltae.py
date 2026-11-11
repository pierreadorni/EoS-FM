"""Lightweight Temporal Attention Encoder (LTAE) for image time series.

Adapted from Garnot & Landrieu, "Lightweight Temporal Self-Attention for
Classifying Satellite Image Time Series", ICASSP 2021.

This spatial variant applies the temporal attention independently per
spatial position, making it compatible with dense prediction (segmentation).
Given feature maps of shape [B, T, C, H, W], it produces [B, C_out, H, W].
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GN1d(nn.Module):
    """GroupNorm(1, C) that accepts 2-D inputs [N, C] by unsqueezing/squeezing."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C]
        return self.gn(x.unsqueeze(-1)).squeeze(-1)


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder for 2D feature maps.

    Processes a sequence of T feature maps [B, T, C, H, W] with multi-head
    self-attention along the temporal axis, independently per spatial position.
    A learnable master query replaces per-pixel queries to keep the model light.

    Args:
        in_channels: Number of input channels (C).
        n_head: Number of attention heads.
        d_k: Key/query dimension per head.
        mlp_channels: List of output channel sizes for the MLP applied after
            temporal pooling. The last value is the output channel count.
        dropout: Dropout rate applied after attention weights.
        T: Maximum number of time steps (used for positional encoding).
        return_att: If True, also return attention weights.
        positional_encoding: If True, add sinusoidal positional encoding.
    """

    def __init__(
        self,
        in_channels: int,
        n_head: int = 4,
        d_k: int = 4,
        mlp_channels: Optional[list[int]] = None,
        dropout: float = 0.1,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.d_k = d_k
        self.return_att = return_att

        if mlp_channels is None:
            mlp_channels = [in_channels, in_channels]

        # Master query: one vector per head, shape [n_head, d_k]
        self.master_query = nn.Parameter(torch.randn(n_head, d_k))
        nn.init.xavier_uniform_(self.master_query.unsqueeze(0))

        # Key projection: C -> n_head * d_k
        self.fc_k = nn.Linear(in_channels, n_head * d_k)

        if positional_encoding:
            self.positional_enc = PositionalEncoding(in_channels, T=T)
        else:
            self.positional_enc = None

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_channels)

        # MLP to map concatenated head outputs -> output channels
        # Input: n_head * in_channels (after weighted sum per head)
        mlp_in = n_head * in_channels
        layers = []
        for out_c in mlp_channels:
            # _GN1d wraps GroupNorm(1,C) to handle 2-D inputs [N, C]
            layers += [nn.Linear(mlp_in, out_c), _GN1d(out_c), nn.ReLU()]
            mlp_in = out_c
        self.mlp = nn.Sequential(*layers)
        self.out_channels = mlp_channels[-1]

    def forward(
        self, x: torch.Tensor, batch_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C, H, W]
            batch_positions: [B, T] optional integer day-of-year indices
                             (ignored if positional_encoding=False)
        Returns:
            out: [B, C_out, H, W]
            (optional) attn: [B, n_head, T, H, W]
        """
        B, T, C, H, W = x.shape

        # Positional encoding (applied channel-wise on features)
        if self.positional_enc is not None:
            pe = self.positional_enc(B, T, x.device)  # [B, T, C]
            x = x + pe[:, :, :, None, None]

        # Layer norm over channel dim
        x_norm = self.layer_norm(x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C))
        # x_norm: [B*H*W, T, C]

        # Keys: [B*H*W, T, n_head, d_k]
        keys = self.fc_k(x_norm).reshape(B * H * W, T, self.n_head, self.d_k)
        keys = keys.permute(0, 2, 1, 3)  # [B*H*W, n_head, T, d_k]

        # Master query: [1, n_head, 1, d_k] -> broadcast over B*H*W
        query = self.master_query.unsqueeze(0).unsqueeze(2)  # [1, n_head, 1, d_k]

        # Attention scores: [B*H*W, n_head, 1, T]
        scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)  # [B*H*W, n_head, 1, T]
        attn = self.dropout(attn)

        # Values are the raw (normalised) input features
        values = x_norm  # [B*H*W, T, C]
        values = values.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        # [B*H*W, n_head, T, C]

        # Weighted sum: [B*H*W, n_head, 1, C]
        agg = torch.matmul(attn, values)
        # Concatenate heads: [B*H*W, n_head * C]
        agg = agg.squeeze(2).reshape(B * H * W, self.n_head * C)

        # MLP: [B*H*W, C_out]
        out = self.mlp(agg)
        C_out = out.shape[-1]
        out = out.reshape(B, H, W, C_out).permute(0, 3, 1, 2)  # [B, C_out, H, W]

        if self.return_att:
            # [B*H*W, n_head, T] -> [B, n_head, T, H, W]
            attn_map = attn.squeeze(2).reshape(B, H, W, self.n_head, T)
            attn_map = attn_map.permute(0, 3, 4, 1, 2)
            return out, attn_map

        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences.

    Generates an encoding of shape [B, T, C] based on integer time indices.
    If no indices are provided, sequential integers 0..T-1 are used.
    """

    def __init__(self, d_model: int, T: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.T = T
        # Pre-compute sinusoidal table for positions 0..T-1
        pe = torch.zeros(T, d_model)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [T, d_model]

    def forward(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Return [B, T, d_model] positional encodings for sequential steps."""
        # Use first T positions
        enc = self.pe[:T].unsqueeze(0).expand(B, -1, -1)  # [B, T, d_model]
        return enc.to(device)


class MultiScaleLTAE(nn.Module):
    """Apply independent LTAE2d modules to each pyramid level.

    Given a list of L feature maps each of shape [B, T, C_l, H_l, W_l],
    returns a list of L fused feature maps [B, C_out_l, H_l, W_l].

    Args:
        channel_list: List of channel counts at each pyramid level.
        n_head: Attention heads (shared across levels).
        d_k: Key dimension per head.
        d_model_factor: Each level's MLP output = channel_list[l] (identity
            by default if d_model_factor=1).
        dropout: Dropout applied to attention weights.
        T: Max time steps.
    """

    def __init__(
        self,
        channel_list: list[int],
        n_head: int = 4,
        d_k: int = 4,
        dropout: float = 0.1,
        T: int = 1000,
    ):
        super().__init__()
        self.ltae_list = nn.ModuleList(
            [
                LTAE2d(
                    in_channels=c,
                    n_head=n_head,
                    d_k=d_k,
                    mlp_channels=[c, c],
                    dropout=dropout,
                    T=T,
                )
                for c in channel_list
            ]
        )
        self.out_channels = channel_list  # same as input (identity dim)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: list of L tensors, each [B, T, C_l, H_l, W_l]
        Returns:
            list of L tensors, each [B, C_l, H_l, W_l]
        """
        return [ltae(feat) for ltae, feat in zip(self.ltae_list, features)]
