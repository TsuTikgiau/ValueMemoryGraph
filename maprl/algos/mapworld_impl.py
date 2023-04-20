import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Dict, Any
import glob

import numpy as np
import torch
from torch.optim import Optimizer
import networkx as nx

from d3rlpy.gpu import Device
from d3rlpy.models.builders import (
    create_deterministic_regressor,
    create_discrete_imitator,
    create_probablistic_regressor,
)
from d3rlpy.models.encoders import EncoderFactory, Encoder, EncoderWithAction
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.torch import DeterministicRegressor, DiscreteImitator, Imitator
from d3rlpy.preprocessing import ActionScaler, Scaler
from d3rlpy.torch_utility import torch_api, train_api, eval_api, map_location, _get_attributes
from d3rlpy.algos.torch.base import TorchImplBase

from maprl.models.models import create_action_converter, create_state_converter
from maprl.models.builders import create_mapgoal_imitator, ConditionedImitator
from maprl.algos.mapgraph import MapGraph
from maprl.algos.utils import get_reward_label


class MapBaseImpl(TorchImplBase, metaclass=ABCMeta):

    _map_size: int
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _imitator: Optional[Imitator]
    _goal_imitator: Optional[ConditionedImitator]
    _enc_s: Optional[Encoder]
    _enc_a: Optional[EncoderWithAction]
    _map_policy: Optional[Imitator]
    _optim: Optional[Optimizer]
    _map_graph: Optional[MapGraph]
    _action_translator_config: dict

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        map_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        action_mode='top',
        neighbor_step=15,
        min_future_step=1,
        f_goal: bool=False,
        action_step=1,
        action_translator_config: dict = {'hidden_units': [256, 256], 'use_skip': False},
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=None,
        )
        self._map_size = map_size
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._use_gpu = use_gpu

        self._action_step = action_step

        # initialized in build
        self._imitator = None
        self._goal_imitator = None
        self._enc_s = None
        self._enc_a = None
        self._optim = None
        self._map_graph = None
        self.action_mode = action_mode
        self._neighbor_step = neighbor_step
        self._min_future_step = min_future_step
        self.f_goal = f_goal
        self._action_translator_config = action_translator_config

    def build(self) -> None:
        self._build_network()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def _build_optim(self) -> None:
        assert self._imitator is not None
        self._optim = self._optim_factory.create(
            [{'params': self._imitator.parameters()},
             {'params': self._goal_imitator.parameters()},
             {'params': self._enc_s.parameters()},
             {'params': self._enc_a.parameters()},
             {'params': self._map_policy.parameters()},
             ],
            lr=self._learning_rate
        )

    def init_map_graph(
            self,
            state: np.ndarray,
            action: np.ndarray,
            episode_terminals: np.ndarray,
            reward: np.ndarray=None,
            cluster_thresh: float=0.6,
            edge_thresh: float=1.6,
            value_discount=0.95,
            env_name: str='',
            model_path='',
            merge_reward_style='avg'
    ) -> None:
        f_s, f_a = self.compute_map(state, action)
        files = glob.glob(os.path.join(model_path, 'graph_*.npz'))
        if len(files):
            saved_graph = np.load(files[0])
            reduced_f_s = saved_graph['reduced_f_s']
            labels = saved_graph['labels']
        else:
            reduced_f_s = None
            labels = None

        self._map_graph = MapGraph(f_s.cpu().numpy(),
                                   episode_terminals=episode_terminals,
                                   rewards=reward,
                                   reward_label=get_reward_label(env_name, state, episode_terminals),
                                   cluster_thresh=cluster_thresh,
                                   edge_thresh=edge_thresh,
                                   value_discount=value_discount,
                                   env_name=env_name,
                                   device=self.device,
                                   reduced_f_s=reduced_f_s,
                                   labels=labels,
                                   merge_reward_style=merge_reward_style)

        self.map_f_s = torch.tensor(self._map_graph.reduced_f_s, device=self.device)
        reduced_state_idx = np.array([np.where(self._map_graph.labels==i)[0][0]
                                      for i in range(len(self.map_f_s))])
        self.map_state = state[reduced_state_idx]

    def update_graph(self):
        top_node = self._map_graph.top_nodes[-1]
        select_nodes = list(nx.ancestors(self._map_graph.graph, top_node).\
            union({top_node}))
        self.map_f_s = self.map_f_s[select_nodes]
        self.map_state = self.map_state[select_nodes]
        self._map_graph.reduced_f_s = self.map_f_s.cpu().numpy()

        sub_graph = self._map_graph.graph.subgraph(select_nodes)
        node_mapping = {v: i for i, v in enumerate(select_nodes)}
        new_graph = nx.relabel_nodes(sub_graph, node_mapping)
        nodes_value = np.array([new_graph.nodes[i]['value'] for i in range(len(node_mapping))])
        self._map_graph.top_nodes = np.argsort(nodes_value)[-1:-30:-1]

        self._map_graph.graph = new_graph

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_modules(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_next: torch.Tensor, goal: torch.Tensor
    ) -> Sequence[np.ndarray]:
        assert self._optim is not None

        self._optim.zero_grad()

        bc_loss, loss_action_recon, loss_contrastive, loss_action_regular, cbc_loss = \
            self.compute_loss(obs_t, act_t, obs_next, goal)

        (bc_loss + loss_action_recon + loss_contrastive + loss_action_regular + cbc_loss).backward()
        self._optim.step()

        loss_dict = {"bc_loss": bc_loss.cpu().detach().numpy(),
                     "cbc_loss": cbc_loss.cpu().detach().numpy(),
                     "loss_action_recon": loss_action_recon.cpu().detach().numpy(),
                     "loss_contrastive": loss_contrastive.cpu().detach().numpy(),
                     "loss_action_regular": loss_action_regular.cpu().detach().numpy(),
                     }

        return loss_dict

    @torch_api(scaler_targets=["state"], action_scaler_targets=["action"])
    def compute_map(
            self, state: torch.Tensor, action: torch.Tensor=None
    ):
        with torch.no_grad():
            f_s = self._enc_s(state)
            if action is not None:
                f_a = self._enc_a(f_s, action)
            else:
                f_a = None
        return f_s, f_a

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_next: torch.Tensor, goal: torch.Tensor
    ) -> Sequence[torch.Tensor]:
        assert self._imitator is not None
        assert self._enc_s is not None and self._enc_a is not None

        f_s = self._enc_s(obs_t)
        f_a = self._enc_a(f_s, act_t)
        f_s_next = self._enc_s(obs_next)

        # reconstruction loss
        loss_action_recon = self._map_policy.compute_error(torch.cat([f_s, f_a], dim=-1), act_t)

        # contrastive loss
        square_distance = torch.square((f_s + f_a)[:, None] - f_s_next[None, :]).sum(-1)
        pos_H = torch.diagonal(square_distance)
        neg_H_m = (2 * self._action_step - square_distance).clamp(min=0)
        neg_H = (neg_H_m.sum(1) - torch.diagonal(neg_H_m)) / (neg_H_m.shape[1] - 1)
        loss_contrastive = (pos_H + neg_H).mean()

        # neg_H = (square_distance.sum(1) - pos_H) / (square_distance.shape[1] - 1)
        # loss_contrastive = (pos_H + (2 * self._action_step - neg_H).clamp(min=0)).mean()

        # regularization loss
        loss_action_regular = (torch.square(f_a).sum(-1) - self._action_step).clamp(min=0).mean()

        processed_goal = self.process_goal(goal)
        # conditional bc loss
        cbc_loss = self._goal_imitator.compute_error(obs_t, processed_goal, act_t)

        # orig_bc_loss (need to remove later)
        bc_loss = self._imitator.compute_error(obs_t, act_t)

        return bc_loss, loss_action_recon, loss_contrastive, loss_action_regular, cbc_loss

    def _obtain_orig_state(self, map_f_s):
        min_idx = torch.argmin(torch.norm(map_f_s - self.map_f_s, dim=-1))
        orig_state = torch.tensor(self.map_state[min_idx], dtype=torch.float32, device=self.device)
        return orig_state

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        if self.action_mode == 'bc':
            # pure behavior cloning without any VMG thing. will be removed
            assert self._imitator is not None
            return self._imitator(x)
        elif self.action_mode == 'top':
            return self.action2top_nodes(x)
        elif self.action_mode == 'neighbor':
            return self.action_to_best_neighbor(x)
        else:
            raise KeyError('unknown action mode')

    @torch.no_grad()
    def action2top_nodes(self, source):
        f_s, _ = self.compute_map(source)

        if not hasattr(self, 'plan_path') or self.plan_path.expired or self.plan_path.get_distance(f_s).min() > 2:
            path_idx_list, path_f_s, _ = \
                self._map_graph.path2top_nodes(f_s.cpu().numpy(), undirect=False)
            if path_idx_list is not None:
                self.plan_path = PlanPath(path_idx_list, torch.tensor(path_f_s, device=self.device),
                                          wait_step=10, min_future=self._min_future_step, max_future=6)
            else:
                pass

        goal_f_s = self.plan_path.get_goal(f_s)

        if self.f_goal:
            goal = goal_f_s[None]
        else:
            goal = self._obtain_orig_state(goal_f_s).view(*source.shape)
        action = self._goal_imitator(source, goal)

        return action

    @torch.no_grad()
    def action_to_best_neighbor(self, source):
        f_s, _ = self.compute_map(source)

        if 'door' in self._map_graph.env_name:
            handle = source[..., -7:-4]
            distance_handle = torch.norm(handle - torch.tensor(self.map_state[..., -7:-4], device=handle.device), dim=-1)
            mask = (distance_handle < torch.sort(distance_handle)[0][100]).cpu().numpy()
        else:
            mask = None

        if not hasattr(self, 'plan_path') or self.plan_path.expired or self.plan_path.get_distance(f_s).min() > 2:
            path_idx, path_f_s, already_on_graph = \
                self._map_graph.get_high_value_target(f_s.cpu().numpy(), step=self._neighbor_step, top=False, mask=mask)
            self.plan_path = PlanPath(path_idx, torch.tensor(path_f_s, device=self.device), wait_step=10, min_future=self._min_future_step, max_future=8)

        next_target = self.plan_path.get_goal(f_s)

        goal = self._obtain_orig_state(torch.tensor(next_target, device=self.device)).view(*source.shape)
        if not isinstance(source, torch.Tensor):
            source = torch.tensor(source, dtype=torch.float32, device=self.device)
        processed_goal = self.process_goal(goal)
        action = self._goal_imitator(source, processed_goal)
        return action

    def predict_value(
        self, x: np.ndarray, action: np.ndarray, with_std: bool
    ) -> np.ndarray:
        raise NotImplementedError("BC does not support value estimation")

    @eval_api
    @torch_api(scaler_targets=["obs", "goal"])
    def goal_conditioned_action(self, obs: torch.Tensor, goal: torch.Tensor) -> np.ndarray:
        processed_goal = self.process_goal(goal)
        return self._goal_imitator(obs, processed_goal).cpu().detach().numpy()

    def process_goal(self, goal):
        if self.f_goal:
            goal = self._enc_s(goal)
        return goal

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))

        def set_state_dict(impl: Any, chkpt: Dict[str, Any]) -> None:
            for key in _get_attributes(impl):
                obj = getattr(impl, key)
                if isinstance(obj, torch.nn.Module):
                    print(obj.load_state_dict(chkpt[key], strict=False))
                # elif isinstance(obj, torch.optim.Optimizer):
                #     obj.load_state_dict(chkpt[key])

        set_state_dict(self, chkpt)


class MapImpl(MapBaseImpl):

    _imitator: Optional[DeterministicRegressor]

    def _build_network(self) -> None:
        self._imitator = create_deterministic_regressor(
            self._observation_shape, self._action_size, self._encoder_factory
        )

        goal_size = self._map_size if self.f_goal else self._observation_shape[0]
        self._goal_imitator = create_mapgoal_imitator(
            [self._observation_shape[0] + goal_size], self._action_size, self._encoder_factory,
            normalized=self.f_goal,
            g_mean=torch.zeros(self._map_size) if self.f_goal else torch.zeros(self._observation_shape[0]),
            g_std=torch.ones(self._map_size) if self.f_goal else torch.ones(self._observation_shape[0]),
            additional_config=self._action_translator_config
        )

        self._enc_s = create_state_converter(
            self._observation_shape, self._map_size, self._encoder_factory)
        self._enc_a = create_action_converter(
            self._map_size, self._action_size, self._encoder_factory)
        self._map_policy = create_probablistic_regressor([self._map_size*2], self._action_size, self._encoder_factory)

    def rebuild_goal_imitator(self):
        self.f_goal = True
        goal_size = self._map_size
        g_mean = torch.mean(self.map_f_s, dim=0)
        g_std = torch.std(self.map_f_s, dim=0)

        self._goal_imitator = create_mapgoal_imitator(
            [self._observation_shape[0] + goal_size], self._action_size, self._encoder_factory,
            normalized=self.f_goal,
            g_mean=g_mean,
            g_std=g_std
        )

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        self._build_optim()


class PlanPath(object):
    def __init__(self, idx_list, path_f_s, wait_step=3, min_future=3, max_future=3):
        self.idx_list = idx_list
        self.path_f_s = path_f_s
        self.not_achieved = 0
        self.wait_step = wait_step
        self.current_step = 0
        self.min_future = min_future
        self.max_future = max_future

    def get_distance(self, f_s):
        dist_square = ((f_s - self.path_f_s) ** 2).sum(-1)
        return dist_square ** 0.5

    def check_close_path_node(self, f_s):
        distance = self.get_distance(f_s)
        close_path_node_idx = torch.argmin(distance)
        if close_path_node_idx > self.not_achieved:  # check how many steps we have stucked on this node
            self.not_achieved = close_path_node_idx
            self.current_step = 0
        else:
            self.current_step = self.current_step + 1
        return close_path_node_idx

    def get_goal(self, f_s):
        close_path_node_idx = self.check_close_path_node(f_s)
        goal_idx = min(close_path_node_idx + self.future_step, len(self.idx_list) - 1)
        return self.path_f_s[goal_idx]

    @property
    def future_step(self):
        """ this is like a relu centered at the center between max future and min future.
        If the step is before the center, the future step is the min. after the center,
        it grows linearly to the max"""
        pass_center = self.current_step - self.wait_step / 2
        if pass_center < 0:
            return self.min_future
        else:
            return self.min_future + \
                   int(pass_center / (self.wait_step / 2) * (self.max_future - self.min_future))

    @property
    def expired(self):
        return self.current_step > self.wait_step