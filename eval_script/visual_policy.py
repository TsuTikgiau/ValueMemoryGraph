import os
import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import json

import d3rlpy
from d3rlpy.preprocessing.stack import StackedObservation

from maprl.algos.mapworld import MapWorld
import maprl.env
from maprl.utils import postprocess_dataset


def init_everything(args):
    print('initializing')
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    if args.filter_set:
        dataset = postprocess_dataset(dataset, args.dataset)

    with open(os.path.join(args.model_path, 'params.json'), 'r') as f:
        params = json.load(f)

    map_model = MapWorld(map_size=params['map_size'], use_gpu=args.gpu)
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
                             model_path=args.model_path
                             )
    map_model._map_graph.init_node_value()
    # map_model._impl.update_graph()
    map_model._map_graph.init_visual()
    print('initialization finish')

    return dataset, env, map_model


def draw_map_img(map_model, observation_list, on_map_list, path_f_s_list=[]):
    f_s, _ = map_model._impl.compute_map(np.stack(observation_list))
    f_s_visual = map_model._map_graph._visual_model.transform(f_s.cpu().numpy())

    if len(path_f_s_list):
        path_visual_list = np.split(map_model._map_graph._visual_model.transform(torch.cat(path_f_s_list).cpu().numpy()),
                                    np.cumsum([len(path_f_s) for path_f_s in path_f_s_list])[:-1])

    # fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(dpi=300)
    ax = plt.gca()
    map_model.draw_map()
    plt.tight_layout()
    fig.canvas.draw()
    graph_bg = get_canvas(fig)
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    plt.close()

    line_mask_list = []
    map_img_list = []
    # fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(dpi=300)
    ax = plt.gca()

    tmp_draw = []

    for t in range(1, len(observation_list)):
        [line.remove() for line in tmp_draw]
        tmp_draw = []
        if len(path_f_s_list):
            path = ax.plot(path_visual_list[t-1][:, 0], path_visual_list[t-1][:, 1], 'm')
            tmp_draw.extend(path)
        current_marker = ax.plot(f_s_visual[t, 0], f_s_visual[t, 1], 'k+')
        tmp_draw.extend(current_marker)

        color = 'g-' if on_map_list[t] else 'r-'
        ax.plot(f_s_visual[t-1:t+1, 0], f_s_visual[t-1:t+1, 1], color, linewidth=2)  # line width for debug
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        plt.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        img = get_canvas(fig)
        line_mask = (img.sum(-1, keepdims=True) != 255*3).astype(np.uint8)
        map_img = graph_bg * (1 - line_mask) + line_mask * img
        line_mask_list.append(line_mask)
        map_img_list.append(map_img)

    return map_img_list


def draw_map_img_maze(map_model, observation_list):
    f_s_visual = np.stack(observation_list)[:, :2]

    # fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(dpi=300)
    ax = plt.gca()
    pos = {i: loc for i, loc in enumerate(map_model._impl.map_state[:, :2])}
    map_model.draw_map(pos)
    plt.tight_layout()
    fig.canvas.draw()
    graph_bg = get_canvas(fig)
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    plt.close()

    line_mask_list = []
    map_img_list = []
    # fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(dpi=300)
    ax = plt.gca()

    tmp_draw = []

    for t in range(1, len(observation_list)):
        [line.remove() for line in tmp_draw]
        tmp_draw = []
        current_marker = ax.plot(f_s_visual[t, 0], f_s_visual[t, 1], 'k+')
        tmp_draw.extend(current_marker)

        color = 'g-'  # for debug if on_map_list[t] else 'r-'
        ax.plot(f_s_visual[t-1:t+1, 0], f_s_visual[t-1:t+1, 1], color, linewidth=2)  # line width for debug
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        plt.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        img = get_canvas(fig)
        line_mask = (img.sum(-1, keepdims=True) != 255*3).astype(np.uint8)
        map_img = graph_bg * (1 - line_mask) + line_mask * img
        line_mask_list.append(line_mask)
        map_img_list.append(map_img)

    return map_img_list


def get_canvas(fig):
    plot_string = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = plot_string.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return canvas


def create_video(frame_list, map_img_list, save_path):
    # v_w = cv2.VideoWriter(save_path,
    #                       cv2.VideoWriter_fourcc(*'DIVX'), 20, (2240, 960))
    v_w = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc(*'MP4V'), 20, (2240, 960))
    for frame, map_img in zip(frame_list, map_img_list):
        to_plot = np.concatenate(
            [cv2.resize(frame, (1280, 960)),
             cv2.resize(map_img, (960, 960))], axis=1)
        to_plot = cv2.cvtColor(to_plot, cv2.COLOR_BGR2RGB)

        v_w.write(to_plot)
    v_w.release()


def create_frames(frame_list, map_img_list, map_img_list_maze, save_path):
    os.makedirs(save_path, exist_ok=True)
    for t in range(len(frame_list)):
        frame, map_img, map_img_maze = frame_list[t], map_img_list[t], map_img_list_maze[t]

        frame = cv2.resize(frame, (1280, 960))[:, 160:1280-160]
        map_img = cv2.resize(map_img, (960, 960))
        to_plot = np.concatenate([frame, map_img], axis=0)
        to_plot = cv2.cvtColor(to_plot, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_path, '{:0>3d}.png'.format(t)), to_plot)

        # if t in [0, len(frame_list)-1]:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        #     map_img_maze = cv2.cvtColor(map_img_maze, cv2.COLOR_BGR2RGB)
        #
        #     cv2.imwrite(os.path.join(save_path, '{:0>3d}.png'.format(t)), frame)
        #     cv2.imwrite(os.path.join(save_path, '{:0>3d}_m.png'.format(t)), map_img)
        #     # cv2.imwrite(os.path.join(save_path, '{:0>3d}_mm.png'.format(t)), map_img_maze)


def run(
    env: gym.Env, map_model: MapWorld, save_path, n_trials: int = 3,
) -> float:

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    if is_image:
        stacked_observation = StackedObservation(
            observation_shape, map_model.n_frames
        )

    episode_rewards = []
    for i in range(n_trials):

        observation_list = []
        frame_list = []
        on_map_list = []
        path_f_s_list = []

        observation = env.reset()
        if hasattr(map_model, 'reset'):
            map_model.reset(observation=observation)
        episode_reward = 0.0

        # frame stacking
        observation_list.append(observation)
        distance = map_model.distance_to_node(observation)
        on_map_list.append(distance < 0.6)
        if is_image:
            stacked_observation.clear()
            stacked_observation.append(observation)

        n_step_after_done = 1
        has_done = False
        while True:
            # take action
            if is_image:
                policy_input = stacked_observation.eval()
            else:
                policy_input = observation

            # action = map_model.predict(policy_input)
            action = map_model.predict([policy_input])[0]

            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            observation_list.append(observation)
            if hasattr(map_model.impl, 'plan_path'):
                path_f_s_list.append(map_model.impl.plan_path.path_f_s.cpu())
            distance = map_model.distance_to_node(observation)
            on_map_list.append(distance<0.6)

            if is_image:
                stacked_observation.append(observation)

            frame_list.append(env.render('rgb_array'))

            if done:
                has_done = True

            if has_done:
                n_step_after_done = n_step_after_done - 1
                if n_step_after_done < 0:
                    break

        print('episode finished')
        os.makedirs(save_path, exist_ok=True)
        print('episode_reward: {}'.format(episode_reward))

        # for debug
        if episode_reward < 2:
            continue
        # for debug end

        video_save_path = '{}/video_{}.mp4'.format(save_path, i)
        # # for debug
        # map_img_list = draw_map_img(map_model, observation_list, on_map_list)
        # map_img_list_maze = draw_map_img_maze(map_model, observation_list)
        # create_frames(frame_list, map_img_list, map_img_list_maze, video_save_path)
        # # for debug end

        # map_img_list = draw_map_img(map_model, observation_list, on_map_list)
        map_img_list = draw_map_img(map_model, observation_list, on_map_list, path_f_s_list=path_f_s_list)
        create_video(frame_list, map_img_list, video_save_path)

        episode_rewards.append(env.get_normalized_score(episode_reward))
        print('Video Created {}'.format(video_save_path))
    return float(np.mean(episode_rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-complete-render-v0')
    parser.add_argument(
        '--model_path', type=str,
        default='/home/zhud/project/mapworldmodel/d3rlpy_logs/kitchen-complete-v0/Map_1_20211124140834')
    parser.add_argument('--ckpt', type=str, default='500')
    parser.add_argument('--save_root', type=str, default='/home/zhud/project/mapworldmodel/plot')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--cluster_thresh', type=float, default=1.0)
    parser.add_argument('--discount', type=float, default=0.8)
    parser.add_argument('--action_mode', type=str, default='neighbor')
    parser.add_argument('--min_future_step', type=int, default=1)
    parser.add_argument('--neighbor_step', type=int, default=12)
    parser.add_argument('--filter_set', default=False, action='store_true')
    args = parser.parse_args()

    save_path = os.path.join(args.save_root, args.dataset, os.path.basename(os.path.normpath(args.model_path)))

    dataset, env, map_model = init_everything(args)
    run(env, map_model, save_path, n_trials=args.n_trials)


if __name__ == '__main__':
    main()
