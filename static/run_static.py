# usage: python -m static.run_static --seed 0 [--model lr|rf] [--tune]
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common.config import ID, LABEL, FAMILY
from scipy import sparse
from common.splits import outer_split, inner_folds, check_splits
from static.model import make_static_model
from static.groups import regroup_benign


def fit_predict(X, y, tr, te, k, seed, kind):
    m = make_static_model(k, seed, kind).fit(X[tr], y[tr])
    assert list(m.classes_) == [0, 1]
    return m, m.predict_proba(X[te])


def oof(X, y, folds, k, seed, kind):
    p = np.full((X.shape[0], 2), np.nan)
    for tr, va in folds:
        p[va] = fit_predict(X, y, tr, va, k, seed, kind)[1]
    return p


def choose_k(X, y, dev, folds, seed, kind, grid=(100, 500, 2000, None)):
    scores = {k: roc_auc_score(y[dev], oof(X, y, folds, k, seed, kind)[dev, 1]) for k in grid}
    print('k search (OOF AUC):', scores)
    return max(scores, key=scores.get)


def rows(meta, idx, p, seed, mode, role):
    out = meta.iloc[idx][[ID, FAMILY, LABEL]].rename(columns={LABEL: 'y'}).copy()
    out['seed'], out['cv_mode'], out['role'] = seed, mode, role
    out['p_benign'], out['p_ransom'] = p[idx, 0], p[idx, 1]
    return out


def summarize(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=float((pred == y).mean()), recall=float(pred[y == 1].mean()),
                fpr=float(pred[y == 0].mean()), auc=float(roc_auc_score(y, p)))


def main(seed, tune, kind):
    meta = pd.read_parquet('data/master.parquet', columns=[ID, LABEL, FAMILY]).reset_index(drop=True)
    X = sparse.load_npz('data/X_sparse.npz').tocsr()
    assert X.shape[0] == len(meta), 'run python -m static.prepare_X first'
    meta = regroup_benign(meta, X)
    os.makedirs('artifacts', exist_ok=True)
    y = meta[LABEL].to_numpy()
    dev, test = outer_split(meta, seed)
    gf, rdf = inner_folds(meta, dev, seed, 'group'), inner_folds(meta, dev, seed, 'random')
    check_splits(meta, dev, test, gf)
    k = choose_k(X, y, dev, gf, seed, kind) if tune else None
    p_g, p_r = oof(X, y, gf, k, seed, kind), oof(X, y, rdf, k, seed, kind)
    model, p_te = fit_predict(X, y, dev, test, k, seed, kind)
    p_t = np.full((X.shape[0], 2), np.nan)
    p_t[test] = p_te

    if 'rf' in model.named_steps:
        model.set_params(rf__n_jobs=1)                   # single-thread timing
    n = min(100, len(test))
    t0 = time.perf_counter()
    for i in test[:n]:
        model.predict_proba(X[i:i + 1])
    ms = (time.perf_counter() - t0) / n * 1000

    out = pd.concat([rows(meta, dev, p_g, seed, 'group', 'oof'),
                     rows(meta, dev, p_r, seed, 'random', 'oof'),
                     rows(meta, test, p_t, seed, 'group', 'test')])
    stem = '' if kind == 'lr' else f'_{kind}'
    path = f'artifacts/preds_static{stem}_seed{seed}.parquet'
    out.to_parquet(path, index=False)
    summ = dict(seed=seed, model=kind, k=k, latency_ms=ms, n_dev=len(dev), n_test=len(test),
                test_families=int(meta[FAMILY].iloc[test].nunique()),
                random_oof=summarize(y[dev], p_r[dev, 1]), group_oof=summarize(y[dev], p_g[dev, 1]),
                test=summarize(y[test], p_te[:, 1]))
    with open(f'artifacts/static_summary{stem}_seed{seed}.json', 'w') as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    from common.validate import validate_preds
    validate_preds(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--model', choices=['lr', 'rf'], default='lr')
    a = ap.parse_args()
    main(a.seed, a.tune, a.model)
