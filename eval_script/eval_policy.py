import os
import argparse
import json
import numpy as np

import d3rlpy

from maprl.algos.mapworld import MapWorld
from maprl.scorer import evaluate_on_environment_normalized
from maprl.utils import postprocess_dataset


def init_everything(args):
    print('initializing')
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    if args.filter_set:
        dataset = postprocess_dataset(dataset, args.dataset)

    print('repeat state numbers: {}'.format(len(dataset.observations) - len(np.unique(dataset.observations, axis=-1))))

    with open(os.path.join(args.model_path, 'params.json'), 'r') as f:
        params = json.load(f)
        if 'f_goal' not in params:
            params['f_goal'] = False

        kargs = {}
        if 'action_translator_config' in params:
            kargs['action_translator_config'] = params['action_translator_config']

    map_model = MapWorld(map_size=params['map_size'], f_goal=params['f_goal'], use_gpu=args.gpu, **kargs)
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
                             model_path=args.model_path,
                             merge_reward_style=args.merge_reward_style)
    map_model._map_graph.init_node_value()
    # map_model._impl.update_graph()
    print('initialization finish')

    return dataset, env, map_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0')
    parser.add_argument(
        '--model_path', type=str,
        default='/home/zhud/project/mapworldmodel/d3rlpy_logs/kitchen-complete-v0/Map_1_20211124140834')
    parser.add_argument('--ckpt', type=str, default='500')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--cluster_thresh', type=float, default=1.0)
    parser.add_argument('--discount', type=float, default=0.8)
    parser.add_argument('--action_mode', type=str, default='neighbor')
    parser.add_argument('--min_future_step', type=int, default=1)
    parser.add_argument('--neighbor_step', type=int, default=12)
    parser.add_argument('--filter_set', default=False, action='store_true')
    parser.add_argument('--merge_reward_style', type=str, default='avg')

    args = parser.parse_args()

    dataset, env, map_model = init_everything(args)
    print('start evaluation')
    result = evaluate_on_environment_normalized(env, n_trials=args.n_trials, bar=True)(map_model)
    print(result)


if __name__ == '__main__':
    main()
