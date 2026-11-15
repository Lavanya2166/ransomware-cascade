import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from common.config import LABEL, FAMILY, TEST_FRAC, N_INNER


def outer_split(df, seed: int):
    fam_labels = df.groupby(FAMILY)[LABEL].mean().round().astype(int)
    rng = np.random.default_rng(seed)
    test_fams = []

    for lab in (0, 1):
        fams = fam_labels.index[fam_labels == lab].to_numpy().copy()
        rng.shuffle(fams)
        n_test = max(1, round(TEST_FRAC * len(fams)))
        test_fams.extend(fams[:n_test])

    is_test = df[FAMILY].isin(test_fams).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def inner_folds(df, dev_idx, seed: int, mode: str = "group"):
    d = df.iloc[dev_idx]
    if mode == "group":
        splitter = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
        it = splitter.split(d, d[LABEL], groups=d[FAMILY])
    elif mode == "random":
        splitter = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
        it = splitter.split(d, d[LABEL])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [(dev_idx[tr], dev_idx[va]) for tr, va in it]


def check_splits(df, dev_idx, test_idx, group_folds):
    fams = df[FAMILY].to_numpy()
    assert set(fams[dev_idx]).isdisjoint(set(fams[test_idx])), "Family leak between dev and test!"
    for tr, va in group_folds:
        assert set(fams[tr]).isdisjoint(set(fams[va])), "Family leak inside inner fold!"
    for idx in (dev_idx, test_idx):
        assert set(df[LABEL].iloc[idx]) == {0, 1}, "A split is missing one of the classes!"