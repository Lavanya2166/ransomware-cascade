import glob
import os
import numpy as np
import pandas as pd
from trust import baselines as B
from trust import conformal as C
from trust import metrics as M

ALPHA_B = 0.10
ALPHA_R_GRID = [0.01, 0.02, 0.05, 0.10]


def load_seed(seed: int):
    d = pd.read_parquet(f"artifacts/preds_static_seed{seed}.parquet")
    pick = lambda mode, role: d[(d["cv_mode"] == mode) & (d["role"] == role)]
    return pick("group", "oof"), pick("random", "oof"), pick("group", "test")


def extract_py(df: pd.DataFrame):
    return df[["p_benign", "p_ransom"]].to_numpy(), df["y"].to_numpy()


def run_seed(seed: int) -> pd.DataFrame:
    oof_g, oof_r, te = load_seed(seed)
    pt, yt = extract_py(te)
    rows = []

    for cal_name, cal_df in (("group", oof_g), ("random", oof_r)):
        pc, yc = extract_py(cal_df)
        pv = C.pvalues(pt, C.cal_scores(pc, yc))

        for ar in ALPHA_R_GRID:
            sets = C.prediction_sets(pv, (ALPHA_B, ar))
            keep, pred = C.route(pt, sets)
            cov = M.coverage(yt, sets)
            esc = ~keep

            rows.append(
                {
                    "seed": seed,
                    "calib": cal_name,
                    "method": "conformal",
                    "alpha_r": ar,
                    "cov_benign": cov[0],
                    "cov_ransom": cov[1],
                    **M.router_metrics(yt, pred, esc),
                }
            )

            if cal_name == "group":
                arg = pt.argmax(axis=1)
                target_rate = float(esc.mean())
                baselines = [
                    ("maxsoft", B.u_maxsoft(pt)),
                    ("entropy", B.u_entropy(pt)),
                    ("credibility", B.u_credibility(pv)),
                    ("random", B.u_random(len(pt), seed)),
                ]
                for name, u in baselines:
                    b_esc = B.escalate_at_rate(u, target_rate, seed)
                    rows.append(
                        {
                            "seed": seed,
                            "calib": "group",
                            "method": name,
                            "alpha_r": ar,
                            "cov_benign": np.nan,
                            "cov_ransom": np.nan,
                            **M.router_metrics(yt, arg, b_esc),
                        }
                    )

    return pd.DataFrame(rows)


def run_curves(seed: int) -> pd.DataFrame:
    oof_g, _, te = load_seed(seed)
    (po, yo), (pt, yt) = extract_py(oof_g), extract_py(te)
    pv = C.pvalues(pt, C.cal_scores(po, yo))
    arg = pt.argmax(axis=1)
    out = []

    baselines = [
        ("maxsoft", B.u_maxsoft(pt)),
        ("entropy", B.u_entropy(pt)),
        ("credibility", B.u_credibility(pv)),
        ("random", B.u_random(len(pt), seed)),
    ]
    for name, u in baselines:
        for rate in np.linspace(0.0, 0.6, 31):
            out.append(
                {
                    "seed": seed,
                    "method": name,
                    "rate": float(rate),
                    "err_auroc": np.nan,
                    **M.router_metrics(yt, arg, B.escalate_at_rate(u, rate, seed)),
                }
            )
        out.append(
            {
                "seed": seed,
                "method": name,
                "rate": -1.0,
                "err_auroc": M.err_auroc(yt, arg, u),
            }
        )

    return pd.DataFrame(out)


def main():
    os.makedirs("results", exist_ok=True)
    files = sorted(glob.glob("artifacts/preds_static_seed*.parquet"))
    all_seeds = [int(f.split("seed")[1].split(".")[0]) for f in files]
    real_seeds = [s for s in all_seeds if s != 99]
    seeds = real_seeds if real_seeds else all_seeds

    assert len(seeds) > 0, "No prediction files found in artifacts/."
    print(f"[*] Running trust evaluation on seeds: {seeds}")

    res = pd.concat([run_seed(s) for s in seeds], ignore_index=True)
    res.to_csv("results/all_runs.csv", index=False)
    res.drop(columns=["seed"]).groupby(["calib", "method", "alpha_r"]).agg(["mean", "std"]).to_csv(
        "results/summary.csv"
    )
    pd.concat([run_curves(s) for s in seeds], ignore_index=True).to_csv(
        "results/curves.csv", index=False
    )
    print("[OK] Saved results/all_runs.csv, results/summary.csv, and results/curves.csv")


if __name__ == "__main__":
    main()