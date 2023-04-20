import os
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import d3rlpy

from maprl.algos.mapworld import MapWorld
from maprl.algos.mapgraph import MapGraph
from maprl.utils import postprocess_dataset


def plot(map_episode, color='r-'):
    plt.plot(map_episode[:, 0], map_episode[:, 1], color)
    plt.plot(map_episode[-1, 0], map_episode[-1, 1], 'go', markersize=10)


def init_everything(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    if args.filter_set:
        dataset = postprocess_dataset(dataset, args.dataset)

    with open(os.path.join(args.model_path, 'params.json'), 'r') as f:
        params = json.load(f)
        if 'f_goal' not in params:
            params['f_goal'] = False

    map = MapWorld(map_size=params['map_size'], f_goal=params['f_goal'], use_gpu=args.gpu)
    action_size = dataset.get_action_size()
    observation_shape = tuple(dataset.get_observation_shape())
    map.create_impl(
        map._process_observation_shape(observation_shape),
        action_size,
    )
    map.load_model(os.path.join(args.model_path, 'model_{}000.pt'.format(args.ckpt)))
    map.init_map_graph(dataset.observations, dataset.actions,
                       dataset.episode_terminals, dataset.rewards,
                       cluster_thresh=args.cluster_thresh,
                       value_discount=args.discount,
                       env_name=args.dataset,
                       model_path=args.model_path)
    map._map_graph.init_visual()
    map._map_graph.init_node_value()

    return dataset, env, map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='antmaze-large-diverse-v1')
    parser.add_argument(
        '--model_path', type=str,
        default='/home/zhud/ai/log/mapworld/antmaze-large-diverse-v1/total_1_20220306163143')
    parser.add_argument('--ckpt', type=str, default='800')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)

    parser.add_argument('--cluster_thresh', type=float, default=0.8)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--filter_set', default=False, action='store_true')

    args = parser.parse_args()


    dataset, env, map = init_everything(args)
    f_s, f_a = map._impl.compute_map(dataset.observations, dataset.actions)
    f_s = f_s.detach().cpu().numpy()
    f_a = f_a.detach().cpu().numpy()

    sns.kdeplot(data=(f_a**2).sum(-1)**0.5)
    plt.show()

    map.draw_map()
    plt.savefig('graph.png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()
    pos = {i: loc for i, loc in enumerate(map._impl.map_state[:, :2])}
    map.draw_map(pos)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('graph_maze.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # graph = MapGraph(f_s, episode_terminals=dataset.episode_terminals, edge_thresh=1.6, )
    # graph.init_visual()
    # f_s_visual = graph._visual_model.transform(f_s)
    #
    # start_i = 0
    # for end_i in np.where(dataset.episode_terminals)[0]:
    #     plt.figure(figsize=(8, 8))
    #     graph.draw()
    #     episode_visual = f_s_visual[start_i:end_i+1]
    #     episode = f_s[start_i:end_i + 1]
    #     plot(episode_visual, color='C1-')
    #     # planed_episode_idx, planed_episode, _ = graph.plan_path(episode[0], episode[-1])
    #     # plot(graph.reduced_f_s_visual[planed_episode_idx], color='r-')
    #     plt.axis("off")
    #     plt.show()
    #     start_i = end_i + 1

    print('hold out')



if __name__ == '__main__':
    main()



