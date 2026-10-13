"""Adapt TWISTER Atari observations to Gym's process/shared-memory vector API."""

from functools import partial

import gym
import numpy as np
import torch

from .reset_on_exception import ResetOnException
from .time_limit import TimeLimit
from seeding import seed_environment


class AtariEvalEnv(gym.Env):
    def __init__(self, env_class, env_params, time_limit, seed):
        # Each emulator has a dedicated process. Nested OpenCV/PyTorch pools
        # otherwise create hundreds of contending threads for small images.
        import cv2
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        self.env = ResetOnException(TimeLimit(env_class(**env_params), time_limit))
        if seed is not None:
            seed_environment(self.env, seed, "atari100k")
        channels = (1 if env_params.get("grayscale_obs", False) else 3) * env_params.get("history_frames", 1)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(channels, *env_params["img_size"]), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(self.env.num_actions)
        self.time_limit = time_limit
        self.frames = 0

    def reset(self):
        self.frames = 0
        return self.env.reset().state.numpy()

    def step(self, action):
        obs = self.env.step(torch.tensor(action))
        self.frames += self.env.action_repeat
        return (obs.state.numpy(), float(obs.reward),
                bool(obs.done) or self.frames >= self.time_limit,
                {"error": bool(obs.error)})

    def close(self):
        env = self.env
        while env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
                break
            env = getattr(env, "env", None)


def create_eval_vector(model, num_envs):
    params = dict(model.config.env_params, **model.config.eval_env_params)
    channels = (1 if params.get("grayscale_obs", False) else 3) * params.get("history_frames", 1)
    observation_space = gym.spaces.Box(
        0, 255, shape=(channels, *params["img_size"]), dtype=np.uint8)
    factories = [partial(
        AtariEvalEnv, model.config.env_class, params, model.config.time_limit_eval,
        None if model.config.seed is None else
        (model.config.seed + model.config.num_envs + index) % (2 ** 31 - 1))
        for index in range(num_envs)]
    # Supply spaces to avoid constructing a dummy emulator in the GPU process.
    # Spawn is safe even when training/evaluation has already initialized CUDA.
    return gym.vector.AsyncVectorEnv(
        factories, observation_space=observation_space,
        action_space=gym.spaces.Discrete(model.env.num_actions),
        shared_memory=True, copy=False, context="spawn")
