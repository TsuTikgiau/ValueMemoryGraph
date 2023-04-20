from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv
from maprl.env.hand_manipulation_suite.hand import \
    PenEnvV0Render, HammerEnvV0Render, DoorEnvV0Render, RelocateEnvV0Render
from d4rl import infos


# V1 envs
MAX_STEPS = {'hammer': 200, 'relocate': 200, 'door': 200, 'pen': 100}
ENV_MAPPING = {'hammer': 'HammerEnvV0Render', 'relocate': 'RelocateEnvV0Render',
               'door': 'DoorEnvV0Render', 'pen': 'PenEnvV0Render'}
for agent in ['hammer', 'pen', 'relocate', 'door']:
    for dataset in ['human', 'expert', 'cloned']:
        env_name = '%s-%s-render-v1' % (agent, dataset)
        env_name_no_render = '%s-%s-v1' % (agent, dataset)
        register(
            id=env_name,
            entry_point='maprl.env.hand_manipulation_suite:' + ENV_MAPPING[agent],
            max_episode_steps=MAX_STEPS[agent],
            kwargs={
                'ref_min_score': infos.REF_MIN_SCORE[env_name_no_render],
                'ref_max_score': infos.REF_MAX_SCORE[env_name_no_render],
                'dataset_url': infos.DATASET_URLS[env_name_no_render]
            }
        )


DOOR_RANDOM_SCORE = -56.512833
DOOR_EXPERT_SCORE = 2880.5693087298737

HAMMER_RANDOM_SCORE = -274.856578
HAMMER_EXPERT_SCORE = 12794.134825156867

PEN_RANDOM_SCORE = 96.262799
PEN_EXPERT_SCORE = 3076.8331017826877

RELOCATE_RANDOM_SCORE = -6.425911
RELOCATE_EXPERT_SCORE = 4233.877797728884


# Swing the door open
register(
    id='door-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:DoorEnvV0Render',
    max_episode_steps=200,
)

register(
    id='door-human-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:DoorEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_demos_clipped.hdf5'
    }
)

register(
    id='door-human-longhorizon-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:DoorEnvV0Render',
    max_episode_steps=300,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_demos_clipped.hdf5'
    }
)

register(
    id='door-cloned-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:DoorEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='door-expert-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:DoorEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_expert_clipped.hdf5'
    }
)

# Hammer a nail into the board
register(
    id='hammer-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:HammerEnvV0Render',
    max_episode_steps=200,
)

register(
    id='hammer-human-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:HammerEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_demos_clipped.hdf5'
    }
)

register(
    id='hammer-human-longhorizon-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:HammerEnvV0Render',
    max_episode_steps=600,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_demos_clipped.hdf5'
    }
)

register(
    id='hammer-cloned-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:HammerEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='hammer-expert-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:HammerEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_expert_clipped.hdf5'
    }
)

# Reposition a pen in hand
register(
    id='pen-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:PenEnvV0Render',
    max_episode_steps=100,
)

register(
    id='pen-human-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:PenEnvV0Render',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_demos_clipped.hdf5'
    }
)

register(
    id='pen-human-longhorizon-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:PenEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_demos_clipped.hdf5'
    }
)

register(
    id='pen-cloned-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:PenEnvV0Render',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='pen-expert-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:PenEnvV0Render',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_expert_clipped.hdf5'
    }
)

# Relcoate an object to the target
register(
    id='relocate-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:RelocateEnvV0Render',
    max_episode_steps=200,
)

register(
    id='relocate-human-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:RelocateEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_demos_clipped.hdf5'
    }
)

register(
    id='relocate-human-longhorizon-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:RelocateEnvV0Render',
    max_episode_steps=500,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_demos_clipped.hdf5'
    }
)

register(
    id='relocate-cloned-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:RelocateEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='relocate-expert-render-v0',
    entry_point='maprl.env.hand_manipulation_suite:RelocateEnvV0Render',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_expert_clipped.hdf5'
    }
)

