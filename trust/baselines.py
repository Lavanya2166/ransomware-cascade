import numpy as np


def u_maxsoft(p: np.ndarray) -> np.ndarray:
    return 1.0 - p.max(axis=1)


def u_credibility(pv: np.ndarray) -> np.ndarray:
    return 1.0 - pv.max(axis=1)


def u_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def u_random(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def escalate_at_rate(u: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Escalates the exact top `rate` fraction by uncertainty score u."""
    n = len(u)
    jitter = np.random.default_rng(seed).random(n) * 1e-9
    order = np.argsort(-(u + jitter))
    esc = np.zeros(n, dtype=bool)
    esc[order[: int(round(rate * n))]] = True
    return esc