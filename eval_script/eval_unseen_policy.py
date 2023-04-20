import os
import argparse
import json
from tqdm import tqdm
import numpy as np

import d3rlpy
from d3rlpy.dataset import MDPDataset

from maprl.algos.mapworld import MapWorld
from maprl.scorer import evaluate_on_environment_normalized


OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3


def evaluate_unencoraged(env, algo, task='bottom burner', n_trials=10, epsilon=0.0, render=False, bar=True):
    element_idx = OBS_ELEMENT_INDICES[task]

    episode_completations = []
    loop = range(n_trials)
    if bar:
        loop = tqdm(loop)
    for _ in loop:
        observation = env.reset()

        if hasattr(algo, 'reset'):
            algo.reset(observation=observation)
        episode_reward = 0.0
        n_step = 0

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = algo.predict([observation])[0]

            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            qp = observation[:30]
            distance = np.linalg.norm(qp[element_idx] - OBS_ELEMENT_GOALS[task])
            complete = distance < BONUS_THRESH

            n_step += 1
            if done or complete:
                episode_completations.append(float(complete))
                print('n step: {}'.format(n_step))
                break
        if bar:
            loop.set_description('score: {:2f}'.format(np.mean(episode_completations)))

    return float(np.mean(episode_completations))


def relable_reward(dataset, task):
    element_idx = OBS_ELEMENT_INDICES[task]
    qp = dataset.observations[:, :30]
    distance = np.linalg.norm(
        qp[..., element_idx] -
        OBS_ELEMENT_GOALS[task], axis=1)
    complete = distance < BONUS_THRESH
    new_rewards = complete.astype(float)

    output_dict = {
        'observations': dataset.observations,
        'actions': dataset.actions,
        'rewards': new_rewards,
        'terminals': dataset.terminals,
        'episode_terminals': dataset.episode_terminals
    }
    post_dataset = MDPDataset(**output_dict)

    return post_dataset


def init_everything(args):
    print('initializing')
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    # dataset = relable_reward(dataset, task=args.task)

    with open(os.path.join(args.model_path, 'params.json'), 'r') as f:
        params = json.load(f)
        if 'f_goal' not in params:
            params['f_goal'] = False

    map_model = MapWorld(map_size=params['map_size'], f_goal=params['f_goal'], use_gpu=args.gpu)
    action_size = dataset.get_action_size()
    observation_shape = tuple(dataset.get_observation_shape())
    map_model.create_impl(
        map_model._process_observation_shape(observation_shape),
        action_size,
        action_mode=args.action_mode,
        neighbor_step=args.neighbor_step,
        min_future_step=args.min_future_step
    )
    map_model.load_model(os.path.join(args.model_path, 'model_{}000.pt'.format(args.ckpt)))
    map_model.init_map_graph(dataset.observations, dataset.actions,
                             dataset.episode_terminals, dataset.rewards,
                             cluster_thresh=args.cluster_thresh,
                             value_discount=args.discount,
                             env_name=args.dataset,
                             model_path=args.model_path)
    map_model._map_graph.init_node_value()
    # map_model._impl.update_graph()
    print('initialization finish')

    return dataset, env, map_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-partial-v0')
    parser.add_argument(
        '--model_path', type=str,
        default='/home/zhud/project/mapworldmodel/d3rlpy_logs/kitchen-partial-v0/Map_1_20211214113131')
    parser.add_argument('--ckpt', type=str, default='500')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--cluster_thresh', type=float, default=1.0)
    parser.add_argument('--discount', type=float, default=0.8)
    parser.add_argument('--action_mode', type=str, default='top')
    parser.add_argument('--min_future_step', type=int, default=1)
    parser.add_argument('--neighbor_step', type=int, default=12)
    parser.add_argument('--filter_set', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='hinge cabinet')
    args = parser.parse_args()

    dataset, env, map_model = init_everything(args)
    print('start evaluation')
    result = evaluate_unencoraged(env, map_model, task=args.task, n_trials=args.n_trials, bar=True)
    print(result)


if __name__ == '__main__':
    main()
