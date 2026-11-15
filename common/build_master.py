import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from common.config import ID, LABEL, FAMILY, N_BENIGN_GROUPS


def cluster_benign(X: np.ndarray, n_clusters: int) -> list[str]:
    X_norm = normalize(X, norm="l2")
    model = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(X)),
        metric="cosine",
        linkage="average",
    )
    cids = model.fit_predict(X_norm)
    return [f"benign_g{cid:02d}" for cid in cids]


def main():
    raw_dir = os.path.join("data", "raw")
    data_path = os.path.join(raw_dir, "RansomwareData.csv")
    vars_path = os.path.join(raw_dir, "VariableNames.txt")

    # Col 0 = sample ID, Col 1 = label (0/1), Col 2 = family ID (0 for benign, 1..11 for ransom)
    raw = pd.read_csv(data_path, header=None)
    sample_ids = raw.iloc[:, 0].astype(str).str.strip()
    labels = raw.iloc[:, 1].astype(int).to_numpy()
    fam_ids = raw.iloc[:, 2].astype(int).to_numpy()

    X_raw = raw.iloc[:, 3:].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    # Attach unique feature names
    feat_cols = [f"s_feat_{i:05d}" for i in range(X_raw.shape[1])]
    if os.path.exists(vars_path):
        with open(vars_path, "r", encoding="latin-1") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == X_raw.shape[1]:
            feat_cols = [f"s_{i:05d}_{name}" for i, name in enumerate(names)]
    X_raw.columns = feat_cols

    # Ground-truth family names for ransomware; cosine-clustered groups for benign
    families = np.empty(len(raw), dtype=object)
    ransom_mask = labels == 1
    benign_mask = labels == 0

    families[ransom_mask] = [f"ransom_f{fid:02d}" for fid in fam_ids[ransom_mask]]
    families[benign_mask] = cluster_benign(
        X_raw.loc[benign_mask].to_numpy(), n_clusters=N_BENIGN_GROUPS
    )

    master = pd.concat(
        [
            pd.DataFrame({ID: sample_ids, LABEL: labels, FAMILY: families}),
            X_raw,
        ],
        axis=1,
    ).drop_duplicates(subset=[ID]).reset_index(drop=True)

    out_path = os.path.join("data", "master.parquet")
    master.to_parquet(out_path, index=False)

    print("\n" + "=" * 55)
    print("MASTER DATASET BUILT SUCCESSFULLY")
    print("=" * 55)
    print(f"Saved to            : {out_path}")
    print(f"Total rows          : {len(master)}")
    print(f"Feature columns     : {X_raw.shape[1]}")
    print(f"Class counts        : {master[LABEL].value_counts().to_dict()}")
    print(f"Ransomware families : {master.loc[master[LABEL]==1, FAMILY].nunique()}")
    print(f"Benign groups       : {master.loc[master[LABEL]==0, FAMILY].nunique()}")
    print("=" * 55)


if __name__ == "__main__":
    main()