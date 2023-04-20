import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from d3rlpy.dataset import MDPDataset


def bluecmap():
    min_val, max_val = 0.3, 1.0
    n = 10
    orig_cmap = plt.cm.Blues
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    BlueCMAP = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)
    return BlueCMAP


def postprocess_dataset(dataset, env_name='', return_tresh=0):
    # remove the bad episode
    output_dict = {'observations':[], 'actions':[], 'rewards':[], 'terminals':[], 'episode_terminals':[]}
    episode_terminals = dataset.episode_terminals
    start_idx = 0
    terminals_idx = np.where(episode_terminals)
    for last_idx in terminals_idx[0]:
        rewards = dataset.rewards[start_idx:last_idx+1]

        episode_return = rewards.sum()
        if episode_return.sum() > return_tresh:
            output_dict['observations'].append(dataset.observations[start_idx:last_idx+1])
            output_dict['actions'].append(dataset.actions[start_idx:last_idx + 1])
            output_dict['rewards'].append(rewards)
            output_dict['terminals'].append(dataset.terminals[start_idx:last_idx + 1])
            output_dict['episode_terminals'].append(dataset.episode_terminals[start_idx:last_idx + 1])
        start_idx = last_idx + 1

    output_dict = {k: np.concatenate(v) for k, v in output_dict.items()}
    post_dataset = MDPDataset(**output_dict)
    return post_dataset
