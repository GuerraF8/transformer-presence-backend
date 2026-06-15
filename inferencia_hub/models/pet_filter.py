"""Clasificador Transformer para estimar si un movimiento corresponde a una persona."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PetMotionTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        context_length: int,
        *,
        hidden_size: int = 32,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.context_length = int(context_length)
        self.hidden_size = int(hidden_size)
        self.projection = nn.Linear(self.input_size, self.hidden_size)
        self.position = nn.Parameter(
            torch.zeros(1, self.context_length, self.hidden_size)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=heads,
            dim_feedforward=self.hidden_size * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.normalization = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(values) + self.position[:, : values.shape[1]]
        encoded = self.encoder(hidden)
        pooled = self.normalization(encoded[:, -1, :])
        return self.head(pooled).squeeze(-1)
