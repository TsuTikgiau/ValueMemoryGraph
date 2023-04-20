import numpy as np
import torch


def get_reward_label(env_name: str, state: np.ndarray, episode_terminals: np.ndarray=None):
    if state is None:
        return [0]

    # if 'hammer' in env_name:
    #     nail_board_height = state[:, -2]
    #     height_grid = np.array([0.115, 0.145, 0.175, 0.205, 0.235])
    #     min_idx = np.argmin(np.abs(nail_board_height - height_grid[:, None]), axis=0)
    #     reward_label = height_grid[min_idx]
    # elif 'door' in env_name:
    #     handle_pos = state[:, -7:-4]
    #     pos_grid = np.meshgrid([0.025, 0.09, 0.155],
    #                            [0.08, 0.145, 0.21],
    #                            [0.265, 0.335])
    #     pos_grid = np.stack(pos_grid, axis=-1).reshape(-1, 3)
    #     min_idx = np.argmin(np.linalg.norm(handle_pos - pos_grid[:, None], axis=-1),
    #                         axis=0)
    #
    #     if episode_terminals is not None:
    #         # we use only the initial handle position to distinguish different setting.
    #         episode_start = np.concatenate([np.ones(1), episode_terminals[:-1]])
    #         start_min_idx = min_idx[episode_start == 1]
    #
    #         episode_idx = (np.cumsum(episode_start) - 1).astype(int)
    #         min_idx = start_min_idx[episode_idx]
    #     reward_label = min_idx

    else:
        reward_label = np.zeros_like(state[:, 0])
    return reward_label


def add_reward(reward_dict, reward, reward_label=None):
    if not reward_label:
        reward_label = 0

    if reward_label in reward_dict:
        reward_dict[reward_label].append(reward)
    else:
        reward_dict[reward_label] = [reward]


def combine_reward_dict(reward_dict_1: dict, reward_dict_2: dict):
    total_keys = set(list(reward_dict_1.keys()) + list(reward_dict_2.keys()))
    out = {}
    for key in total_keys:
        a = 0 if key not in reward_dict_1 else reward_dict_1[key]
        b = 0 if key not in reward_dict_2 else reward_dict_2[key]
        out[key] = a + b
    return out


def construct_sparse_reward(graph, device, reward_label=0, end_loop=True):
    n_node = len(graph.nodes)

    x, y, r = [], [], []
    for node_from, node_to, edge_p in graph.edges(data=True):
        x.append(node_from)
        y.append(node_to)
        r.append(-2 if reward_label not in edge_p['reward']
                 else edge_p['reward'][reward_label])
    x_tmp = torch.tensor(x, dtype=torch.int, device=device)
    y_tmp = torch.tensor(y, dtype=torch.int, device=device)
    r_tmp = torch.tensor(r, dtype=torch.float16, device=device)

    if end_loop == True:
        # if a node is the end of an episode, add a self connection with the reward same as the max reward of the in-edge
        for i in range(n_node):
            if not (x_tmp==i).sum():  # no out-edge
                in_reward = r_tmp[y_tmp==i]
                if in_reward.nelement():
                    self_pseudo_reward = in_reward.max()
                    x.append(i)
                    y.append(i)
                    r.append(self_pseudo_reward)

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float16, device=device)

    return x, y, r