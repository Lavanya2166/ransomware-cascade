import numpy as np
import pandas as pd


def validate_preds(path: str):
    d = pd.read_parquet(path)
    required = {"seed", "cv_mode", "role", "sample_id", "family", "y", "p_benign", "p_ransom"}
    assert required <= set(d.columns), f"Missing columns: {required - set(d.columns)}"
    assert d[["p_benign", "p_ransom"]].notna().all().all(), "Found NaN probabilities!"
    assert np.allclose(d["p_benign"] + d["p_ransom"], 1.0, atol=1e-6), "Probabilities do not sum to 1!"

    g = d[d["cv_mode"] == "group"]
    assert set(g[g["role"] == "oof"]["family"]).isdisjoint(
        set(g[g["role"] == "test"]["family"])
    ), "Family leakage detected between OOF and test!"
    assert d.groupby(["cv_mode", "role"])["sample_id"].apply(lambda s: s.is_unique).all(), "Duplicate sample_ids!"
    print(f"[OK] {path} passed all validation checks.")