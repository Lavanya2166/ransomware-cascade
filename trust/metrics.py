import numpy as np
from sklearn.metrics import roc_auc_score


def router_metrics(y: np.ndarray, pred: np.ndarray, esc: np.ndarray) -> dict:
    keep = ~esc
    err = pred != y
    return {
        "esc_rate": float(esc.mean()),
        "kept_acc": float((~err[keep]).mean()) if keep.any() else np.nan,
        "silent_miss": float(((pred == 0) & (y == 1) & keep).sum() / max((y == 1).sum(), 1)),
        "false_alarm_kept": float(((pred == 1) & (y == 0) & keep).sum() / max((y == 0).sum(), 1)),
        "err_caught": float((err & esc).sum() / max(err.sum(), 1)),
        "esc_precision": float((err & esc).sum() / max(esc.sum(), 1)),
        "n_err": int(err.sum()),
    }


def coverage(y: np.ndarray, sets: np.ndarray) -> dict[int, float]:
    return {c: float(sets[y == c, c].mean()) for c in (0, 1)}


def err_auroc(y: np.ndarray, pred: np.ndarray, u: np.ndarray) -> float:
    err = (pred != y).astype(int)
    return float(roc_auc_score(err, u)) if 0 < err.sum() < len(err) else np.nan