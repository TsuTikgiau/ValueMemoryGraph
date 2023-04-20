import os
from typing import Any, Dict, List, Optional, Sequence, Union, Generator, Tuple

from tqdm import tqdm
import numpy as np
import torch

from d3rlpy.argument_utility import (
    ActionScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import TransitionMiniBatch, MDPDataset
from d3rlpy.gpu import Device
from d3rlpy.logger import LOG
from d3rlpy.iterators import RandomIterator
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from maprl.algos.base import ImprovedAlgoBase, CorrectedSummaryWriter
from maprl.algos.mapworld_impl import MapBaseImpl, MapImpl


class _MapBase(ImprovedAlgoBase):
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _impl: Optional[MapBaseImpl]
    _action_translator_config: dict

    def __init__(
        self,
        *,
        map_size: int = 10,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        batch_size: int = 100,
        n_frames: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        impl: Optional[MapBaseImpl] = None,
        f_goal: bool = False,
        action_step: float = 1,
        K: int = 10,
        action_translator_config: dict = {'hidden_units': [256, 256], 'use_skip': False},
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=1,
            gamma=1.0,
            scaler=scaler,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )
        self._map_size = map_size
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder("default")
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        self._f_goal = f_goal
        self._action_step = action_step
        self._action_translator_config = action_translator_config
        self._K = K

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss_dict = \
            self._impl.update_modules(batch.observations, batch.actions,
                                      batch.next_observations, self._get_goal_target(batch, future_steps=self._K))
        return loss_dict

    def _update_2nd(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        loss_dict = \
            self._impl.update_modules_2nd(
                batch.observations, batch.actions,
                self._get_goal_target_2nd_stage(batch))
        return loss_dict

    def _get_goal_target(self, batch: TransitionMiniBatch, future_steps: int=10):
        """

        :param batch:
        :param future_steps: how many future steps we consider when we search the neighbors
        :return: the neighbors with the highest value
        """
        targets_list = []
        for transition in batch.transitions:
            future_t = np.random.randint(future_steps)
            before_goal_transition = transition
            for t in range(future_t):
                if before_goal_transition.next_transition is not None:
                    before_goal_transition = before_goal_transition.next_transition
                else:
                    before_goal_transition = transition
            targets_list.append(before_goal_transition.next_observation)
        targets = np.stack(targets_list)
        return targets

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> np.ndarray:
        """value prediction is not supported by BC algorithms."""
        raise NotImplementedError("BC does not support value estimation.")

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> None:
        """sampling action is not supported by BC algorithm."""
        raise NotImplementedError("BC does not support sampling action.")

    @property
    def _map_graph(self):
        return self._impl._map_graph

    def init_map_graph(self, *args, **kwargs) -> None:
        self._impl.init_map_graph(*args, **kwargs)

    @torch.no_grad()
    def map_surfing(self, source, target):
        assert self._map_graph is not None
        f_s, _ = self._impl.compute_map(source)
        _, path_f_s, on_graph = self._map_graph.plan_path(f_s.cpu().numpy(), target)

        # if the current location is on the graph, we go to the next f_s.
        # If not, we go the closest f_s first
        if len(path_f_s) == 1:
            next_subtarget = path_f_s[0]
        else:
            next_subtarget = path_f_s[int(on_graph)]
        f_a = torch.tensor(next_subtarget, dtype=torch.float32) - f_s
        action = self._impl._map_policy(
            torch.cat([f_s, f_a], dim=-1)).cpu().numpy()
        return action

    @torch.no_grad()
    def distance_to_node(self, source):
        f_s, _ = self._impl.compute_map(source)
        _, _, distance = self._map_graph.get_close_node(f_s.cpu().numpy())
        return distance

    def draw_map(self, pos=None):
        self._map_graph.draw(pos=pos)

    def draw_episode(self, state, color='r-'):
        f_s, _ = self._impl.compute_map(state)
        self._map_graph.draw_episode(f_s, color)

    def goal_conditioned_action(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.impl.goal_conditioned_action(obs, goal)

    def reset(self, observation=None, **kwargs):
        self._impl._map_graph.current_top = 0
        # self._map_graph.init_node_value(state=observation,
        #                                 device='cuda:{}'.format(self._use_gpu.get_id()))
        ## for debug

    @property
    def action_mode(self):
        if hasattr(self, '_impl'):
            return self._impl.action_mode
        else:
            print('impl is not initalized yet')
            return None

    def set_action_mode(self, action_mode):
        if hasattr(self, '_impl'):
            self._impl.action_mode = action_mode


class MapWorld(_MapBase):
    r"""Main part of VMG.


    .. math::
    Args:
        map_size (int): number of dimensions in the learned metric space,
        learning_rate (float): learing rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action scaler. The available options are ``['min_max']``.
        impl (algos.mapworld_impl.MapImpl):
            implemenation of the algorithm.

    """

    _impl: Optional[MapImpl]

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int, **kwargs,
    ) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.

        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self._create_impl(observation_shape, action_size, **kwargs)

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int, **kwargs
    ) -> None:
        self._impl = MapImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            map_size=self._map_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            f_goal=self._f_goal,
            action_translator_config=self._action_translator_config,
            action_step=self._action_step,
            **kwargs
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def rebuild_goal_imitator(self):
        self._f_goal = True
        self._impl.rebuild_goal_imitator()

