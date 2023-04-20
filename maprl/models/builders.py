from typing import Sequence, cast
import torch
import torch.nn as nn
import torch.nn.functional as F

from d3rlpy.models.encoders import Encoder, EncoderFactory
from d3rlpy.models.torch.imitators import DeterministicRegressor
from maprl.models.encoders import SkipVectorEncoderFactory


class ConditionedImitator(nn.Module):

    def __init__(self, encoder: Encoder, action_size: int, normalized=False, g_mean=0, g_std=1):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)
        self.normalized = normalized

        if not isinstance(g_mean, torch.Tensor):
            g_mean = torch.Tensor(g_mean)
        if not isinstance(g_std, torch.Tensor):
            g_std = torch.Tensor(g_std)

        self.g_mean = torch.nn.Parameter(g_mean, requires_grad=False)
        self.g_std = torch.nn.Parameter(g_std, requires_grad=False)

    def __call__(self, x: torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, g))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.normalized:
            g = (g - self.g_mean) / self.g_std

        x = torch.cat([x, g], dim=-1)
        h = self._encoder(x)
        h = self._fc(h)
        return torch.tanh(h)

    def compute_error(
        self, x: torch.Tensor, g: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(self.forward(x, g), action)


def create_mapgoal_imitator(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    normalized: bool = False,
    g_mean=0,
    g_std=1,
    additional_config=None
) -> ConditionedImitator:
    # encoder = encoder_factory.create(observation_shape)
    encoder = SkipVectorEncoderFactory(**additional_config).create(observation_shape)
    return ConditionedImitator(encoder, action_size, normalized=normalized, g_mean=g_mean, g_std=g_std)
