import argparse
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import d3rlpy

from maprl import algos

matplotlib.use('TkAgg')


def init_everything(args):
    print('initializing')
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    map_model = algos.MapWorld()
    action_size = dataset.get_action_size()
    observation_shape = tuple(dataset.get_observation_shape())
    map_model.create_impl(
        map_model._process_observation_shape(observation_shape), action_size
    )
    map_model.load_model(
        '/home/zhud/project/mapworldmodel/d3rlpy_logs/Map_kitchen-complete-v0_1_20211012160613/model_500000.pt')
    map_model.init_map_graph(dataset.observations, dataset.actions, dataset.episode_terminals)
    map_model._map_graph.init_visual()
    print('initialization finish')

    return dataset, env, map_model


def draw_map_img(map_model, observation_list, color='r-'):
    f_s, _ = map_model._impl.compute_map(np.stack(observation_list))
    f_s_visual = map_model._map_graph._visual_model.transform(f_s)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    map_model.draw_map()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    map_img_list = []
    for t in range(len(observation_list)):
        fig.canvas.restore_region(bg)
        ax.plot(f_s_visual[:t+1, 0], f_s_visual[:t+1, 1], color)
        fig.canvas.draw()
        # fig.canvas.blit(ax.bbox)
        # fig.canvas.flush_events()
        plot_string = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        map_img_list.append(plot_string.reshape(fig.canvas.get_width_height()[::-1] + (3,)))

    return map_img_list


def run(
    env: gym.Env, map_model: algos.MapWorld, n_trials: int = 10, render: bool = False
) -> None:

    map_model.draw_map()
    plt.show()


    idx1, subgoal1, subgoal_visual1 = map_model._map_graph.get_close_node_visual([12.45, 5.03])
    idx2, subgoal2, subgoal_visual2 = map_model._map_graph.get_close_node_visual([7.72, 3.48])


    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-complete-render-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    dataset, env, map_model = init_everything(args)

    run(env, map_model, render=True)


if __name__ == '__main__':
    main()
