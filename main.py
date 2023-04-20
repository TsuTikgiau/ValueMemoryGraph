import os
import argparse
from datetime import datetime
import json
from omegaconf import OmegaConf
import d3rlpy
from sklearn.model_selection import train_test_split
from maprl.algos.mapworld import MapWorld
from maprl.scorer import evaluate_on_environment_normalized, continuous_goal_action_diff_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='d3rlpy_logs')
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--map_size', type=int, default=10)
    parser.add_argument('--action_step', type=float, default=1)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--exp_tag', type=str, default='Test')
    parser.add_argument('--config', type=str, default='default')

    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    # save path setting
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = "{}_{}_{}".format(args.exp_tag, args.seed, date)
    logdir = os.path.join(args.save_root, args.dataset)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    action_translator_config = OmegaConf.to_object(
        OmegaConf.load('configs/{}.yaml'.format(args.config))['action_translator'])

    map = MapWorld(map_size=args.map_size, use_gpu=args.gpu, action_step=args.action_step, K=args.K,
                   action_translator_config=action_translator_config)




    map.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=800000,
            n_steps_per_epoch=1000,
            save_interval=50,
            scorers={
                # 'environment_norm': evaluate_on_environment_normalized(env),
                # 'action_error': d3rlpy.metrics.continuous_action_diff_scorer,
                'goal_action_error': continuous_goal_action_diff_scorer
            },
            experiment_name=experiment_name,
            logdir=logdir,
            tensorboard_dir=logdir,
            with_timestamp=False)


if __name__ == '__main__':
    main()
