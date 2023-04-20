from abc import ABCMeta, abstractmethod
from typing import Tuple, cast, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.torch import Encoder, EncoderWithAction


def create_state_converter(
        observation_shape: Sequence[int],
        map_size: int,
        encoder_factory: EncoderFactory):
    enc_s = encoder_factory.create(observation_shape)
    state_converter = StateConverter(enc_s, map_size)
    return state_converter


def create_action_converter(
        map_size: int,
        action_size: int,
        encoder_factory: EncoderFactory):
    enc_a = encoder_factory.create_with_action([map_size], action_size)
    action_converter = ActionConverter(enc_a, map_size)
    return action_converter


class StateConverter(nn.Module):
    _encoder: Encoder
    _fc: nn.Linear
    def __init__(
        self,
        encoder: Encoder,
        map_size: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._map_size = map_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._map_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fc(self._encoder(x))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


class ActionConverter(nn.Module):
    _encoder: EncoderWithAction
    _fc: nn.Linear
    def __init__(
        self,
        encoder: EncoderWithAction,
        map_size: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._map_size = map_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._map_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._fc(self._encoder(x, action))

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))
