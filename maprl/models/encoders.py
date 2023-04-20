from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union
import torch

from d3rlpy.models.encoders import EncoderFactory, PixelEncoderFactory, VectorEncoderFactory, _create_activation
from d3rlpy.models.torch import (
    Encoder,
    EncoderWithAction,
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)

from d3rlpy.models.encoders import register_encoder_factory


class SkipVectorEncoder(VectorEncoder):
    def __init__(self, *args, use_skip: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_skip = use_skip

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, fc in enumerate(self._fcs):
            h_input = h
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
            if self._use_skip:
                if h.shape[-1] == h_input.shape[-1]:
                    h = h + h_input
        return h


class SkipVectorEncoderFactory(VectorEncoderFactory):
    """Vector encoder factory class with skip connections.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): list of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_dense (bool): flag to use DenseNet architecture.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "skipvector"
    _use_skip: bool

    def __init__(self, *args, use_skip: bool = False, **kargs,
    ):
        super().__init__(*args, **kargs)
        self._use_skip = use_skip

    def create(self, observation_shape: Sequence[int]) -> VectorEncoder:
        assert len(observation_shape) == 1
        return SkipVectorEncoder(
            observation_shape=observation_shape,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            use_skip=self._use_skip,
            activation=_create_activation(self._activation),
        )


class FeatureSizeDefaultEncoderFactory(EncoderFactory):
    """Default encoder factory class with feature size specification.

    This encoder factory returns an encoder based on observation shape with output feature size.

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "default_fsize"
    _feature_size: int
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]

    def __init__(
        self,
        feature_size: float = 10,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        self._feature_size = feature_size
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate

    def create(self, observation_shape: Sequence[int], feature_size: int = 0) -> Encoder:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if not feature_size:
            feature_size = self._feature_size
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
                feature_size=feature_size
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
                hidden_units=[256, feature_size]
            )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
        feature_size: int = 0,
    ) -> EncoderWithAction:
        if not feature_size:
            feature_size = self._feature_size
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
                feature_size=feature_size,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
                hidden_units=[256, feature_size],
            )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
        }


register_encoder_factory(FeatureSizeDefaultEncoderFactory)