import gym
import gym_minigrid


def make_env(env_key, env_args=None, seed=None):
    if env_args is None:
        env = gym.make(env_key)
    else:
        env = gym.make(env_key, **env_args)
    env.seed(seed)
    return env
