"""Transformer de ocupación independiente del nombre y cantidad de habitaciones."""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeOccupancyTransformer(nn.Module):
    """Puntúa una habitación candidata usando características relativas."""

    def __init__(
        self,
        *,
        input_size: int = 12,
        context_length: int = 28,
        count_classes: int = 5,
        hidden_size: int = 48,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.context_length = context_length
        self.count_classes = count_classes
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(
            torch.zeros(1, context_length, hidden_size)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.room_head = nn.Linear(hidden_size, 1)
        self.count_head = nn.Linear(hidden_size, count_classes)

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = values[:, -self.context_length :, :]
        projected = self.input_projection(sequence)
        projected = projected + self.position[:, -projected.shape[1] :, :]
        encoded = self.encoder(projected)
        hidden = encoded[:, -1, :]
        return self.room_head(hidden).squeeze(-1), self.count_head(hidden)
