import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_coverage_ablation(runs: pd.DataFrame, out_dir: str):
    """Fig 1: Empirical ransomware coverage under family-disjoint vs. random calibration."""
    conf = runs[runs["method"] == "conformal"].copy()
    agg = (
        conf.groupby(["calib", "alpha_r"])["cov_ransom"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    alphas = sorted(agg["alpha_r"].unique())
    nominals = [1.0 - a for a in alphas]
    ax.plot(alphas, nominals, "k--", linewidth=1.5, label="Nominal Target ($1 - \\alpha_r$)")

    styles = {
        "group": ("Family-Disjoint Calibration (Ours)", "#1f77b4", "o"),
        "random": ("Naive Random Calibration", "#d62728", "s"),
    }
    for calib, (label, color, marker) in styles.items():
        sub = agg[agg["calib"] == calib].sort_values("alpha_r")
        ax.errorbar(
            sub["alpha_r"],
            sub["mean"],
            yerr=sub["std"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=4,
        )

    ax.set_xlabel("Target Ransomware Error Budget ($\\alpha_r$)")
    ax.set_ylabel("Empirical Ransomware Coverage on Unseen Families")
    ax.set_title("Conformal Coverage Validity Under Family Shift")
    ax.set_xticks(alphas)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_coverage_shift.png"), dpi=300)
    plt.close(fig)


def plot_silent_miss_curves(runs: pd.DataFrame, curves: pd.DataFrame, out_dir: str):
    """Fig 2: Silent miss rate vs. dynamic escalation budget (Conformal vs. Baselines)."""
    c_curve = curves[curves["rate"] >= 0].copy()
    agg = (
        c_curve.groupby(["method", "rate"])[["esc_rate", "silent_miss"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = {
        "maxsoft": ("Max-Softmax", "#ff7f0e", "--"),
        "entropy": ("Shannon Entropy", "#2ca02c", "-."),
        "credibility": ("Conformal Credibility", "#9467bd", ":"),
        "random": ("Random Escalation", "#7f7f7f", "--"),
    }
    for method, (label, color, ls) in palette.items():
        sub = agg[agg["method"] == method].sort_values("esc_rate")
        ax.plot(
            sub["esc_rate"] * 100,
            sub["silent_miss"] * 100,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.8,
        )

    # Overlay Mondrian conformal operating points (group calibration)
    conf = runs[(runs["method"] == "conformal") & (runs["calib"] == "group")]
    c_pts = (
        conf.groupby("alpha_r", as_index=False)
        .agg(
            esc_mean=("esc_rate", "mean"),
            esc_std=("esc_rate", "std"),
            sm_mean=("silent_miss", "mean"),
            sm_std=("silent_miss", "std"),
        )
        .sort_values("alpha_r")
    )
    ax.errorbar(
        c_pts["esc_mean"] * 100,
        c_pts["sm_mean"] * 100,
        xerr=c_pts["esc_std"] * 100,
        yerr=c_pts["sm_std"] * 100,
        fmt="o-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=7,
        capsize=4,
        label="Mondrian Conformal Router (Ours)",
    )
    for _, row in c_pts.iterrows():
        ax.annotate(
            f"$\\alpha_r={row['alpha_r']:.2f}$",
            (row["esc_mean"] * 100 + 0.8, row["sm_mean"] * 100 + 0.4),
            fontsize=8,
            color="#1f77b4",
        )

    ax.set_xlabel("Dynamic Analysis Escalation Rate (%)")
    ax.set_ylabel("Silent Miss Rate — Uncaught Ransomware (%)")
    ax.set_title("Security Risk vs. Dynamic Analysis Cost")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_silent_miss_vs_cost.png"), dpi=300)
    plt.close(fig)


def plot_kept_accuracy(runs: pd.DataFrame, out_dir: str):
    """Fig 3: Auto-decided (kept) accuracy vs. escalation rate across risk budgets."""
    conf = runs[(runs["method"] == "conformal") & (runs["calib"] == "group")]
    agg = (
        conf.groupby("alpha_r", as_index=False)
        .agg(
            kept_mean=("kept_acc", "mean"),
            kept_std=("kept_acc", "std"),
            caught_mean=("err_caught", "mean"),
            caught_std=("err_caught", "std"),
        )
        .sort_values("alpha_r")
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(agg))
    width = 0.36

    ax.bar(
        x - width / 2,
        agg["kept_mean"] * 100,
        width,
        yerr=agg["kept_std"] * 100,
        capsize=4,
        label="Kept Accuracy (%)",
        color="#1f77b4",
    )
    ax.bar(
        x + width / 2,
        agg["caught_mean"] * 100,
        width,
        yerr=agg["caught_std"] * 100,
        capsize=4,
        label="Stage-1 Errors Caught by Escalation (%)",
        color="#2ca02c",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\alpha_r={a:.2f}$" for a in agg["alpha_r"]])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Selective Automation Quality Across Risk Budgets")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_kept_accuracy.png"), dpi=300)
    plt.close(fig)


def main():
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    runs = pd.read_csv(os.path.join(out_dir, "all_runs.csv"))
    curves = pd.read_csv(os.path.join(out_dir, "curves.csv"))

    plot_coverage_ablation(runs, out_dir)
    plot_silent_miss_curves(runs, curves, out_dir)
    plot_kept_accuracy(runs, out_dir)
    print("[OK] Saved fig1_coverage_shift.png, fig2_silent_miss_vs_cost.png, and fig3_kept_accuracy.png in results/")


if __name__ == "__main__":
    main()