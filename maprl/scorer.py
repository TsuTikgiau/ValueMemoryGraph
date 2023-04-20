from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from typing_extensions import Protocol
from tqdm import tqdm

from d3rlpy.metrics.scorer import AlgoProtocol, _make_batches, WINDOW_SIZE
from d3rlpy.dataset import Episode
from d3rlpy.preprocessing.stack import StackedObservation


def evaluate_on_environment_normalized(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False, bar: bool = False
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        scoerer function.


    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        loop = range(n_trials)
        if bar:
            loop = tqdm(loop)
        for _ in loop:
            observation = env.reset()

            if hasattr(algo, 'reset'):
                algo.reset(observation=observation)
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(env.get_normalized_score(episode_reward))
            if bar:
                loop.set_description('score: {:2f}'.format(np.mean(episode_rewards)))

        return float(np.mean(episode_rewards))

    return scorer


def continuous_goal_action_diff_scorer(
    algo, episodes: List[Episode]
) -> float:
    r"""Returns squared difference of actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in continuous action-space.
    If the given episodes are near-optimal, the small action difference would
    be better.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D} [(a_t - \pi_\phi (s_t))^2]

    Args:
        algo: mapworld algorithm.
        episodes: list of episodes.

    Returns:
        negative squared action difference.

    """
    total_diffs = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.goal_conditioned_action(
                batch.observations, algo._get_goal_target(batch))
            diff = ((batch.actions - actions) ** 2).sum(axis=1).tolist()
            total_diffs += diff
    # smaller is better, sometimes?
    return -float(np.mean(total_diffs))