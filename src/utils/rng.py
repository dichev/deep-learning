import numpy as np
import torch
import random


def seed_global(seed: int, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def get_rng_states():
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'deterministic': torch.are_deterministic_algorithms_enabled()
    }

def set_rng_states(states):
    random.setstate(states['random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])
    torch.use_deterministic_algorithms(states['deterministic'])


def pick_uniform(arr, n=1, exclude=None):
    if exclude is not None:
        mask = np.ones(len(arr), dtype=bool)
        mask[exclude] = False
        arr = arr[mask]
    return np.random.choice(arr, n)


def sample_from(probs, n=1, exclude=None):
    if exclude is not None:
        probs = probs.copy()
        probs[exclude] = 0.
        probs = probs / probs.sum()
    return np.random.choice(len(probs), n, p=probs)


def trunc_normal_sample(n_shape=(1, ), mu=0, sig=1, a=-2, b=2):
    U = torch.distributions.Uniform(0, 1)
    N = torch.distributions.Normal(0, 1)
    p = U.rsample(n_shape)
    Fa, Fb = N.cdf(torch.tensor([a, b]))
    p_trunk = N.icdf(Fa + p * (Fb - Fa)) * sig + mu
    return p_trunk

