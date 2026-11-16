import numpy as np


def cal_scores(p_oof: np.ndarray, y_oof: np.ndarray) -> dict[int, np.ndarray]:
    """Sorted true-class nonconformity scores s_i = 1 - p(y_i | x_i) per class."""
    return {c: np.sort(1.0 - p_oof[y_oof == c, c]) for c in (0, 1)}


def pvalues(p: np.ndarray, cal: dict[int, np.ndarray]) -> np.ndarray:
    """Computes class-conditional (Mondrian) conformal p-values for classes 0 and 1."""
    out = np.zeros((len(p), 2), dtype=float)
    for c in (0, 1):
        s = cal[c]
        test_scores = 1.0 - p[:, c]
        n_ge = len(s) - np.searchsorted(s, test_scores, side="left")
        out[:, c] = (n_ge + 1.0) / (len(s) + 1.0)
    return out


def prediction_sets(pv: np.ndarray, alpha: tuple[float, float]) -> np.ndarray:
    """Returns boolean matrix [N, 2] where True means class c is included in Gamma(x)."""
    return np.stack([pv[:, 0] > alpha[0], pv[:, 1] > alpha[1]], axis=1)


def route(p: np.ndarray, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keeps single-label sets; escalates empty (0-label) or ambiguous (2-label) sets."""
    keep = sets.sum(axis=1) == 1
    pred = np.where(keep, sets[:, 1].astype(int), p.argmax(axis=1))
    return keep, pred