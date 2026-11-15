import pandas as pd


def load_master(path: str = "data/master.parquet"):
    df = pd.read_parquet(path).reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("s_")]
    return df, feats