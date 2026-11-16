# usage: python -m static.prepare_X
# Builds data/X_sparse.npz (sparse feature matrix) aligned row-for-row with data/master.parquet,
# reading the raw CSV directly so the huge wide parquet never has to be loaded.
import os
import numpy as np
import pandas as pd
from scipy import sparse
from common.config import ID, LABEL


def main():
    meta = pd.read_parquet('data/master.parquet', columns=[ID, LABEL]).reset_index(drop=True)
    raw = pd.read_csv(os.path.join('data', 'raw', 'RansomwareData.csv'), header=None)
    ids = raw.iloc[:, 0].astype(str).str.strip()
    y = raw.iloc[:, 1].astype(int).to_numpy()
    X = raw.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)
    del raw
    keep = ~ids.duplicated().to_numpy()                      # same rule used by build_master
    ids, y, X = ids[keep].reset_index(drop=True), y[keep], X[keep]
    assert len(ids) == len(meta), f'row count mismatch: {len(ids)} vs {len(meta)}'
    assert (ids.to_numpy() == meta[ID].to_numpy()).all(), 'sample ids are not aligned with master.parquet'
    assert (y == meta[LABEL].to_numpy()).all(), 'labels are not aligned with master.parquet'
    sparse.save_npz('data/X_sparse.npz', sparse.csr_matrix(X))
    print('Saved data/X_sparse.npz', X.shape, '| non-zero fraction:', round(float((X != 0).mean()), 5))


if __name__ == '__main__':
    main()
