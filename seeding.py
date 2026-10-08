"""Run RNGs and simulator seeds, applied before network initialization and reset."""

import random

import numpy as np
import torch

from configuration import validate_seed


def seed_everything(seed):
    validate_seed(seed)
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    # Also seeds all CUDA devices, including devices initialized later.
    torch.manual_seed(seed)


def seed_environment(env, seed, env_type):
    if env_type == "atari100k":
        env.seed(seed)
    elif env_type == "dmc":
        env.set_seed(seed)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")
