from common.config import SEEDS
from common.data import load_master
from common.splits import outer_split, inner_folds, check_splits


def test_all_seeds():
    df, feats = load_master()
    assert len(feats) > 0
    for s in SEEDS:
        dev, te = outer_split(df, s)
        folds = inner_folds(df, dev, s, mode="group")
        check_splits(df, dev, te, folds)