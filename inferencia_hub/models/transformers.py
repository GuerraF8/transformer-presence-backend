from __future__ import annotations

import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

from ..domain import TRANSFORMER_LAGS_SEQUENCE, TRANSFORMER_MODEL_CONTEXT_LENGTH


class NextRoomTransformer(nn.Module):
    def __init__(self, input_size: int, room_count: int) -> None:
        super().__init__()
        self.config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=TRANSFORMER_MODEL_CONTEXT_LENGTH,
            lags_sequence=TRANSFORMER_LAGS_SEQUENCE,
            input_size=input_size,
            num_time_features=4,
            d_model=40,
            encoder_layers=2,
            decoder_layers=1,
            dropout=0.1,
            attention_dropout=0.1,
        )
        self.model = TimeSeriesTransformerModel(self.config)
        self.head = nn.Linear(40, room_count)

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=torch.ones_like(past_values),
            future_values=None,
            future_time_features=future_time_features,
        )
        return self.head(outputs.encoder_last_hidden_state[:, -1, :])


class OccupancyTransformer(nn.Module):
    def __init__(self, input_size: int, room_count: int, count_classes: int) -> None:
        super().__init__()
        self.config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=TRANSFORMER_MODEL_CONTEXT_LENGTH,
            lags_sequence=TRANSFORMER_LAGS_SEQUENCE,
            input_size=input_size,
            num_time_features=4,
            d_model=48,
            encoder_layers=2,
            decoder_layers=1,
            dropout=0.12,
            attention_dropout=0.12,
        )
        self.model = TimeSeriesTransformerModel(self.config)
        self.room_head = nn.Linear(48, room_count)
        self.count_head = nn.Linear(48, count_classes)

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=torch.ones_like(past_values),
            future_values=None,
            future_time_features=future_time_features,
        )
        hidden = outputs.encoder_last_hidden_state[:, -1, :]
        return self.room_head(hidden), self.count_head(hidden)
