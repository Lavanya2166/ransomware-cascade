import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from common.config import LABEL, FAMILY, N_BENIGN_GROUPS


def regroup_benign(meta, X, seed=0):
    """Replace the benign family labels with balanced KMeans groups (SVD-50 of the binary features).
    The original grouping put 804 of 942 benign files in one group, which breaks family-wise CV."""
    meta = meta.copy()
    benign = np.where(meta[LABEL].to_numpy() == 0)[0]
    Z = TruncatedSVD(50, random_state=seed).fit_transform(np.log1p(X[benign].toarray()))
    lab = KMeans(N_BENIGN_GROUPS, n_init=10, random_state=seed).fit_predict(Z)
    meta.loc[meta.index[benign], FAMILY] = [f'benign_g{c:02d}' for c in lab]
    return meta
