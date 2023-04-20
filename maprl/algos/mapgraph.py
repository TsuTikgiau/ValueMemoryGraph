import os
from collections import deque
from tqdm import tqdm
import numpy as np
import networkx as nx
from umap import UMAP
import torch
import torch_scatter
import faiss
from matplotlib import pyplot as plt

from maprl.algos.utils import get_reward_label, add_reward, \
    combine_reward_dict, construct_sparse_reward
from maprl.utils import bluecmap


def sparse_node(f_s, k_means=False, *args, **kargs):
    if k_means:
        # ncentroids = int(f_s.shape[0] / 5)
        # ncentroids = 10000 # antmaze large
        # ncentroids = 1000  # antmaze medium
        # ncentroids = 6000  # antmaze small
        # ncentroids = 2000  # antmaze small
        # ncentroids = 25000  # kitchen partial
        # ncentroids = 3000  # kitchen complete
        ncentroids = 25000  # kitchen mixed
        niter = 50
        verbose = True
        d = f_s.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True, nredo=10)
        kmeans.train(f_s)
        D, I = kmeans.index.search(f_s, 1)
        reduced_f_s = kmeans.centroids
        labels = I[:, 0]
        return reduced_f_s, labels

    elif len(f_s) < 400000:
        return sparse_node_slow(f_s, *args, **kargs)
    else:
        return sparse_node_fast(f_s, *args, **kargs)


def sparse_node_slow(f_s, dist_thresh=1.0, device='cuda'):
    with torch.no_grad():
        f_s = torch.tensor(f_s, device=device)
        reduced_f_s = torch.ones_like(f_s) * np.inf
        labels = torch.ones_like(f_s[:, 0], dtype=torch.int) * -1
        reduced_f_s[0] = f_s[0]
        pointer = 1

        pbar = tqdm(desc='sparsing nodes', total=len(f_s))
        for i, f in enumerate(f_s):
            distance = torch.norm(f - reduced_f_s, dim=-1)
            value, index = torch.min(distance, dim=0)
            if value < dist_thresh:
                labels[i] = index
            else:
                reduced_f_s[pointer] = f
                labels[i] = pointer
                pointer = pointer + 1
            pbar.update(1)

        reduced_f_s = reduced_f_s[:pointer].cpu().numpy()
        labels = labels.cpu().numpy()
        pbar.close()

    return reduced_f_s, labels


def sparse_node_fast(f_s, dist_thresh=1.0, device='cuda'):
    with torch.no_grad():
        f_s = torch.tensor(f_s, dtype=torch.float16, device=device)
        reduced_f_s = torch.ones_like(f_s) * np.inf
        labels = torch.ones_like(f_s[:, 0], dtype=torch.int) * -1
        reduced_f_s[0] = f_s[0]
        pointer = 1

        pbar = tqdm(desc='sparsing nodes', total=len(f_s))
        batch_size = 512
        for i in range(0, len(f_s), batch_size):
            f = f_s[i:min(i + batch_size, len(f_s))]

            distance = torch.norm(f[:, None] - reduced_f_s[:pointer], dim=-1)
            values, indices = torch.min(distance, dim=1)

            remain_f = []
            remain_j = []
            for j, (value, index) in enumerate(zip(values, indices)):
                if value < dist_thresh:
                    labels[i + j] = index
                    pbar.update(1)
                else:
                    remain_j.append(j)
                    remain_f.append(f[j])

            if len(remain_j):
                new_reduced_f_s, new_labels = sub_sparse_node(torch.stack(remain_f), dist_thresh)
                new_labels = new_labels + pointer
                num_new = len(new_reduced_f_s)
                reduced_f_s[pointer:pointer + num_new] = new_reduced_f_s
                for j, label in zip(remain_j, new_labels):
                    labels[i + j] = label
                    pbar.update(1)
                pointer = pointer + num_new

        reduced_f_s = reduced_f_s[:pointer].cpu().numpy()
        labels = labels.cpu().numpy()
        pbar.close()

    return reduced_f_s, labels


def sub_sparse_node(f_s, dist_thresh=1.0):
    reduced_f_s = torch.ones_like(f_s) * np.inf
    labels = torch.ones_like(f_s[:, 0], dtype=torch.long) * -1
    reduced_f_s[0] = f_s[0]
    pointer = 1
    for i, f in enumerate(f_s):
        distance = torch.norm(f - reduced_f_s, dim=-1)
        value, index = torch.min(distance, dim=0)
        if value < dist_thresh:
            labels[i] = index
            continue
        else:
            reduced_f_s[pointer] = f
            labels[i] = pointer
            pointer = pointer + 1

    reduced_f_s = reduced_f_s[:pointer]
    labels = labels
    return reduced_f_s, labels


class MapGraph(object):
    def __init__(self, f_s, episode_terminals, rewards=None, reward_label=None,
                 cluster_thresh=0.6, edge_thresh=1.5, value_discount=0.95,
                 env_name='', device='cuda', reduced_f_s=None, labels=None, merge_reward_style='avg'):
        self.cluster_thresh = cluster_thresh
        self.edge_thresh = edge_thresh * cluster_thresh
        self.merge_reward_style = merge_reward_style

        if reduced_f_s is None or labels is None:
            reduced_f_s, labels = sparse_node(f_s, dist_thresh=cluster_thresh, device=device)

        print('Graph/Dataset: {}/{}'.format(len(reduced_f_s), len(f_s)))

        orig_edges = get_orig_edges(labels, episode_terminals, rewards, reward_label,
                                    merge_reward_style=self.merge_reward_style)
        # edges_from_distance = self._create_edge_by_distance(reduced_f_s)
        # edges_from_distance.update(orig_edges)
        # self._create_end_edge()

        self.transition_ratio = get_transition_ratio(labels, episode_terminals)
        print('Env/Graph Transition Ratio: {}'.format(self.transition_ratio))

        self.f_s = f_s
        self.episode_terminals = episode_terminals
        self.reduced_f_s = reduced_f_s

        self.graph = self._create_graph(reduced_f_s, orig_edges)
        self._visual_model = None
        self._cache_value = {}
        self.value_discount = value_discount
        self.env_name = env_name
        self.labels = labels

    def save(self, save_path):
        save_path = os.path.join(save_path, 'graph_{}.npz'.format(self.cluster_thresh))
        np.savez(save_path, reduced_f_s=self.reduced_f_s, labels=self.labels)

    def _create_graph(self, f_s: np.ndarray, edges: dict):
        graph = nx.DiGraph()
        graph.add_nodes_from([(i, {'pos': pos}) for i, pos in enumerate(f_s)])

        edges = [(*edge, {'reward': reward}) for edge, reward in edges.items()]
        graph.add_edges_from(edges)

        # process self connection
        for node in graph.nodes():
            if not (node, node) in graph.edges(node):
                continue

            # distribute the reward inside to in/out connections

            internal_reward_weight = 1
            if self.merge_reward_style == 'weighted':
                internal_reward_weight = self.transition_ratio - 1
            half_self_reward_dict = \
                {reward_label: reward / 2 * internal_reward_weight
                 for reward_label, reward in graph.edges[node, node]['reward'].items()}

            if self.merge_reward_style != 'rm_t':
                for edge in graph.in_edges(node):
                    graph.edges[edge]['reward'] = combine_reward_dict(graph.edges[edge]['reward'],
                                                                      half_self_reward_dict)
            if self.merge_reward_style != 'rm_h':
                for edge in graph.out_edges(node):
                    graph.edges[edge]['reward'] = combine_reward_dict(graph.edges[edge]['reward'],
                                                                      half_self_reward_dict)

            # remove self connection
            graph.remove_edge(node, node)

        return graph

    def _create_end_edge(self, end_node_f_s, end_node_idx, f_s):

        square_distance_matric = ((end_node_f_s[None, :] - f_s[:, None]) ** 2).sum(-1)
        adj_matrix = (square_distance_matric < self.edge_thresh ** 2) & (
                    square_distance_matric > 0)  # add edges between close nodes remove the self connection
        # edges = [edge for edge in zip(*np.where(np.tril(adj_matrix)))]  # undirect edge
        edges = {[end_node_idx[edge[0]], edge[1]]: {0: 0} for edge in zip(*np.where(np.tril(adj_matrix)))}

        return edges

    def _create_edge_by_distance(self, f_s):
        square_distance_matric = ((f_s[None, :] - f_s[:, None]) ** 2).sum(-1)
        adj_matrix = (square_distance_matric < self.edge_thresh ** 2) & (
                    square_distance_matric > 0)  # add edges between close nodes remove the self connection
        # edges = [edge for edge in zip(*np.where(np.tril(adj_matrix)))]  # undirect edge
        edges = {edge: {0: 0} for edge in zip(*np.where(adj_matrix))}
        return edges

    def plan_path(self, source, target, n_neighbor=5, undirect=False):
        source_idx_list, _, distance_start_list = self.get_close_node(source, n=n_neighbor, force_list=True)
        target_idx, _, distance_end = self.get_close_node(target)
        graph = self.graph.to_undirected(as_view=True) if undirect else self.graph
        path_idx = None
        for source_idx, distance_start in zip(source_idx_list, distance_start_list):
            try:
                path_idx = nx.shortest_path(graph, source=source_idx, target=target_idx, weight='value_weight')
                break
            except nx.NetworkXAlgorithmError:
                continue
        if path_idx is None:
            return None, None, None
        path_f_s = np.stack([self.graph.nodes[idx]['pos'] for idx in path_idx])
        already_on_graph = distance_start < self.cluster_thresh
        return path_idx, path_f_s, already_on_graph

    def path2top_nodes(self, source, n_neighbor=10, undirect=False):
        if not hasattr(self, 'current_top'):
            self.current_top = 0
        target = self.reduced_f_s[self.top_nodes[self.current_top]]
        results = self.plan_path(
            source, target, n_neighbor=n_neighbor, undirect=undirect)
        if results[0] is not None:
            return results

        for i in range(1, len(self.top_nodes)):
            target = self.reduced_f_s[self.top_nodes[i]]
            results = self.plan_path(
                source, target, n_neighbor=1, undirect=undirect)
            if results[0] is not None:
                self.current_top = i
                break

        return results

    def get_close_node(self, point, n=1, force_list=False, mask=None):
        square_distance = ((self.reduced_f_s - point) ** 2).sum(-1)
        if not mask is None:
            square_distance[~mask] = 999999999
        sort_idx = np.argsort(square_distance)
        if n == 1 and not force_list:
            idx = sort_idx[0]
        else:
            idx = sort_idx[:n]
        return idx, self.reduced_f_s[idx], square_distance[idx] ** 0.5


    def get_close_node_debug(self, point, n=1, force_list=False):
        square_distance = ((self.reduced_f_s - point) ** 2).sum(-1)
        sort_idx = np.argsort(square_distance)
        if n == 1 and not force_list:
            idx = sort_idx[0]
        else:
            idx = sort_idx[:n]
        return idx, self.reduced_f_s[idx], square_distance[idx] ** 0.5

    def get_close_node_visual(self, point_visual):
        square_distance = ((self.reduced_f_s_visual - point_visual) ** 2).sum(-1)
        idx = np.argmin(square_distance)
        return idx, self.reduced_f_s[idx], self.reduced_f_s_visual[idx]

    ######################
    # Value Related Part #
    ######################
    def init_node_value(self, theta=0.1, max_iter=1000, init_v=0,
                        state=None, device='cuda'):
        if state is not None:
            state = state[None]
        reward_label = get_reward_label(self.env_name, state)[0]

        if reward_label in self._cache_value:
            nodes_value = self._cache_value[reward_label]
        else:
            pbar = tqdm(total=1, desc='initializing node value')
            nodes_value = value_iteraction(self.graph, self.value_discount, theta, max_iter, init_v,
                                           reward_label=reward_label, device=device)
            self._cache_value[reward_label] = nodes_value
            pbar.update(1)
            pbar.close()

        max_value = nodes_value.max()
        for node_id, value in enumerate(nodes_value):
            self.graph.nodes[node_id]['value'] = value
            self.graph.nodes[node_id]['value_weight'] = max_value - value
        self.top_nodes = np.argsort(nodes_value)[-1:-30:-1]

    def get_high_value_target(self, source_f_s, step=5, top=False, n_near=10, mask=None):
        node_idx_list, node_f_s_list, dist2graph_list = self.get_close_node(source_f_s, n_near, mask=mask)

        # best_neighbor_idx, best_neighbor_value = self.get_best_neighbor(node_idx, step)
        # path_idx = nx.shortest_path(self.graph, source=node_idx, target=best_neighbor_idx, weight='value_weight')

        path_idx = None
        best_sum_value = -9999999
        shortest_len2top = 9999999
        for node_idx, node_f_s, dist2graph in zip(node_idx_list, node_f_s_list, dist2graph_list):
            path_idx_candidate, best_sum_value_candidate = generic_bfs_highest_path(self.graph, node_idx, depth_limit=step)
            if top:
                path2top, path2top_f_s, _ = \
                    self.path2top_nodes(self.graph.nodes[path_idx_candidate[-1]]['pos'])
                if path2top is None: continue
                path_length = self.path_length(path2top_f_s)
                if path_length < shortest_len2top:
                    shortest_len2top = path_length
                    path_idx = path_idx_candidate + path2top

            elif best_sum_value_candidate > best_sum_value:
                best_sum_value = best_sum_value_candidate
                path_idx = path_idx_candidate
        if path_idx is None:
            path_idx, best_sum_value = generic_bfs_highest_path(self.graph, node_idx_list[0], depth_limit=step)

        path_f_s = np.stack([self.graph.nodes[idx]['pos'] for idx in path_idx])
        already_on_graph = dist2graph < self.cluster_thresh

        return path_idx, path_f_s, already_on_graph

    def get_best_neighbor(self, node_idx, step=5):
        best_neighbor_idx = node_idx
        best_neighbor_value = -np.inf
        future_nodes = nx.ego_graph(self.graph, node_idx, center=False, radius=step).nodes
        for node_next in future_nodes:
            value_next = self.graph.nodes[node_next]['value']
            if value_next > best_neighbor_value:
                best_neighbor_idx = node_next
                best_neighbor_value = value_next
        return best_neighbor_idx, best_neighbor_value

    def path_length(self, path2top_f_s):
        total_length = np.linalg.norm(path2top_f_s[1:] - path2top_f_s[:-1], axis=-1).sum()
        return total_length

    ######################
    # Visualization Part #
    ######################
    def init_visual(self):
        if self._visual_model is not None:
            return
        pbar = tqdm(total=1, desc='visualizing the map')
        self._visual_model = UMAP(verbose=True, )
        self._visual_model.fit(self.reduced_f_s)
        self.reduced_f_s_visual = \
            self._visual_model.transform(self.reduced_f_s)
        pbar.update(1)
        pbar.close()

    def draw(self, pos=None):
        self.init_visual()
        if pos is None:
            pos = {i: loc for i, loc in enumerate(self.reduced_f_s_visual)}
        node_colors = [self.graph.nodes[i]['value'] for i in pos.keys()]
        plot_edges = nx.draw_networkx_edges(self.graph, pos, alpha=0.4, arrows=False)
        plot_nodes = nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=5,
            node_color=node_colors,
            cmap=bluecmap(),
        )
        # plt.colorbar(plot_nodes, ax=plt.gca)

    def draw_episode(self, f_s, color='r-'):
        f_s_visual = self._visual_model.transform(f_s)
        plt.plot(f_s_visual[:, 0], f_s_visual[:, 1], color)


@torch.no_grad()
def value_iteraction(graph, discount=0.9, theta=0.1, max_iter=1000,
                     init_v=0, reward_label=0, device='cuda'):
    n_node = len(graph.nodes)
    value_V = torch.ones([n_node], dtype=torch.float16, device=device) * init_v

    X, Y, R = construct_sparse_reward(graph, device=device, reward_label=reward_label)
    num_node_not_done = 999
    i = 0
    while num_node_not_done > 0 and i < max_iter:
        A = R + discount * value_V[Y]
        new_value_V = torch_scatter.scatter_max(A, X, dim_size=n_node,
                                                out=torch.ones_like(value_V) * init_v)[0]
        num_node_not_done = (torch.abs(new_value_V - value_V) > theta).sum()
        value_V = new_value_V
        i = i + 1
    return value_V.cpu().numpy()


def get_transition_ratio(labels, episode_terminals):
    ratio_list = []

    start_i = 0
    for end_i in np.where(episode_terminals)[0]:
        episode_labels = labels[start_i:end_i + 1]
        len_on_graph = len(np.unique(episode_labels)) - 1
        if len_on_graph > 0:
            ratio_list.append(len(episode_labels) / len_on_graph)
        start_i = end_i + 1
    ratio = sum(ratio_list) / len(ratio_list)
    return ratio


def get_orig_edges(labels, episode_terminals, rewards=None, reward_labels=None, merge_reward_style='avg'):
    edges_reward = {}
    start_i = 0
    for end_i in np.where(episode_terminals)[0]:
        episode_labels = labels[start_i:end_i + 1]
        if rewards is not None:
            episode_rewards = rewards[start_i:end_i + 1]
        if reward_labels is not None:
            episode_reward_labels = reward_labels[start_i:end_i + 1]
        for edge_idx, edge in enumerate(zip(episode_labels[:-1], episode_labels[1:])):
            if edge not in edges_reward.keys():
                edges_reward[edge] = {}
            if rewards is not None:
                reward = episode_rewards[edge_idx+1]
                reward_label = episode_reward_labels[edge_idx+1] if reward_labels is not None else None
                add_reward(edges_reward[edge], reward, reward_label)
            else:
                edges_reward[edge] = {0: [0]}
        start_i = end_i + 1

    edges_avg_reward = {}
    self_edges = {}

    if merge_reward_style in ['avg', 'rm', 'rm_h', 'rm_t', 'weighted']:
        merge_reward_fn = lambda reward_list: sum(reward_list) / len(reward_list)
    elif merge_reward_style == 'sum':
        merge_reward_fn = lambda reward_list: sum(reward_list)
    elif merge_reward_style == 'max':
        merge_reward_fn = lambda reward_list: max(reward_list)
    else:
        raise NotImplementedError

    for edge, reward_dict in edges_reward.items():
        if edge[0] != edge[1]:  # edge across (merged) nodes
            reward = {reward_label: merge_reward_fn(reward_list).item()
                      for reward_label, reward_list in reward_dict.items()}
            edges_avg_reward[edge] = reward
        else:  # edge inside a (merged) node
            if merge_reward_style == 'rm': continue # rm doesn't use rewards inside a node
            reward = {reward_label: merge_reward_fn(reward_list).item()
                      for reward_label, reward_list in reward_dict.items()}
            self_edges[edge] = reward
    edges_total = {**edges_avg_reward, **self_edges}

    return edges_total


def generic_bfs_highest_path(G, source, depth_limit, bi_direction=False):
    visited = {source: (depth_limit+1, G.nodes[source]['value'])}
    to_parent = {}
    levels = {}
    if bi_direction:
        next_one = lambda node: iter(list(G.successors(node)) + list(G.predecessors(node)))
    else:
        next_one = G.successors

    if depth_limit is None:
        depth_limit = len(G)
    queue = deque([(source, depth_limit, next_one(source))])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                to_parent[child] = parent
                sum_value = G.nodes[child]['value'] + visited[parent][1]
                visited[child] = (depth_now,
                                  sum_value)
                if depth_now in levels:
                    levels[depth_now].append((child, sum_value))
                else:
                    levels[depth_now] = [(child, sum_value)]

                if depth_now > 1:
                    queue.append((child, depth_now - 1, next_one(child)))

        except StopIteration:
            queue.popleft()

    if len(levels):
        leaves = levels[min(levels.keys())]
        leaves.sort(key=lambda x: x[1])
        best_leaf, best_sum_value = leaves[-1]
        path = [best_leaf]
        node = best_leaf
        while True:
            try:
                parent_node = to_parent[node]
                path.append(parent_node)
                node = parent_node
            except KeyError:
                break
        path = path[::-1]
    else:
        path = [source]
        best_sum_value = G.nodes[source]['value']
    return path, best_sum_value




