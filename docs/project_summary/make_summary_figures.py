import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trust import conformal as C, baselines as B, metrics as M
from trust.run_trust import ALPHA_B, ALPHA_R_GRID, extract_py

OUT = 'docs/project_summary/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

def eval_trust(folder):
    rows, curves = [], []
    for seed in range(5):
        d = pd.read_parquet(f"{folder}/preds_static_seed{seed}.parquet")
        pick = lambda mode, role: d[(d["cv_mode"] == mode) & (d["role"] == role)]
        oof_g, oof_r, te = pick("group", "oof"), pick("random", "oof"), pick("group", "test")
        pt, yt = extract_py(te)
        for cal_name, cal_df in (("group", oof_g), ("random", oof_r)):
            pc, yc = extract_py(cal_df)
            pv = C.pvalues(pt, C.cal_scores(pc, yc))
            for ar in ALPHA_R_GRID:
                sets = C.prediction_sets(pv, (ALPHA_B, ar))
                keep, pred = C.route(pt, sets)
                cov = M.coverage(yt, sets)
                esc = ~keep
                rows.append(dict(seed=seed, calib=cal_name, method="conformal", alpha_r=ar,
                                 cov_benign=cov[0], cov_ransom=cov[1], **M.router_metrics(yt, pred, esc)))
                if cal_name == "group":
                    arg = pt.argmax(axis=1)
                    target_rate = float(esc.mean())
                    for name, u in [("maxsoft", B.u_maxsoft(pt)), ("entropy", B.u_entropy(pt)),
                                    ("credibility", B.u_credibility(pv)), ("random", B.u_random(len(pt), seed))]:
                        b_esc = B.escalate_at_rate(u, target_rate, seed)
                        rows.append(dict(seed=seed, calib="group", method=name, alpha_r=ar,
                                         cov_benign=np.nan, cov_ransom=np.nan, **M.router_metrics(yt, arg, b_esc)))
        (po, yo), (pt, yt) = extract_py(oof_g), extract_py(te)
        pv = C.pvalues(pt, C.cal_scores(po, yo))
        arg = pt.argmax(axis=1)
        for name, u in [("maxsoft", B.u_maxsoft(pt)), ("entropy", B.u_entropy(pt)),
                        ("credibility", B.u_credibility(pv)), ("random", B.u_random(len(pt), seed))]:
            for rate in np.linspace(0.0, 0.6, 31):
                curves.append(dict(seed=seed, method=name, rate=float(rate), err_auroc=np.nan,
                                   **M.router_metrics(yt, arg, B.escalate_at_rate(u, rate, seed))))
            curves.append(dict(seed=seed, method=name, rate=-1.0, err_auroc=M.err_auroc(yt, arg, u)))
    return pd.DataFrame(rows), pd.DataFrame(curves)

rf_runs, rf_curves = eval_trust('old random forest results')
lr_runs, lr_curves = eval_trust('artifacts')

rf_runs.to_csv('docs/project_summary/trust_runs_rf.csv', index=False)
lr_runs.to_csv('docs/project_summary/trust_runs_lr.csv', index=False)

# 1. Coverage validity under family shift (Group vs Random calibration, for LR and RF)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
for ax, (runs, title) in zip(axes, [(lr_runs, 'Stage 1 = Logistic Regression (Primary)'),
                                    (rf_runs, 'Stage 1 = Random Forest (Old Baseline)')]):
    conf = runs[runs['method'] == 'conformal']
    agg = conf.groupby(['calib', 'alpha_r'])['cov_ransom'].agg(['mean', 'std']).reset_index()
    alphas = sorted(agg['alpha_r'].unique())
    ax.plot(alphas, [(1.0 - a) * 100 for a in alphas], 'k--', lw=1.6, label='Target coverage (1 - alpha_r)')
    for calib, (lbl, col, mk) in [('group', ('Family-disjoint OOF calib (Ours)', '#2563eb', 'o')),
                                  ('random', ('Naive random-split OOF calib', '#dc2626', 's'))]:
        sub = agg[agg['calib'] == calib].sort_values('alpha_r')
        ax.errorbar(sub['alpha_r'], sub['mean'] * 100, yerr=sub['std'] * 100, label=lbl,
                    color=col, marker=mk, lw=2, capsize=4)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Ransomware risk budget (alpha_r)')
    ax.set_xticks(alphas)
    ax.set_ylim(75, 102)
    ax.grid(True, ls=':', alpha=0.6)
axes[0].set_ylabel('Empirical ransomware coverage on unseen families (%)')
axes[0].legend(loc='lower left', frameon=False, fontsize=9)
fig.suptitle('Ablation: Family-Disjoint Calibration Preserves Coverage; Random Calibration Undercovers', fontsize=11.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_trust_coverage_both.png', dpi=180)
plt.close(fig)

# 2. Conformal router performance across alpha_r: LR vs RF
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
metrics_spec = [
    ('esc_rate', 'Files escalated to dynamic sandbox (%)\n(Lower = faster / cheaper)'),
    ('silent_miss', 'Silent-miss rate on ransomware (%)\n(Lower = safer, target <= alpha_r)'),
    ('kept_acc', 'Accuracy on kept (auto-decided) files (%)\n(Higher = cleaner auto-decisions)')
]
for idx_ax, (ax, (col_name, title)) in enumerate(zip(axes, metrics_spec)):
    for runs, lbl, col, mk in [(lr_runs, 'LR + Conformal (New)', '#2563eb', 'o'),
                               (rf_runs, 'RF + Conformal (Old)', '#f59e0b', 's')]:
        sub = runs[(runs['method'] == 'conformal') & (runs['calib'] == 'group')].groupby('alpha_r')[col_name].agg(['mean', 'std']).reset_index()
        x = np.arange(len(sub))
        vals = sub['mean'] * 100
        ax.plot(x, vals, marker=mk, color=col, lw=2.2, label=lbl)
        for xi, val in zip(x, vals):
            if idx_ax == 0:
                dy = 2.5 if col == '#2563eb' else -5.2
            elif idx_ax == 1:
                dy = 0.55 if (col == '#2563eb' and xi == 3) else (-0.95 if col == '#2563eb' else 0.55)
            else:
                dy = 1.8 if col == '#2563eb' else -4.2
            ax.annotate(f'{val:.1f}%', (xi, val + dy),
                        ha='center', fontsize=8.5, color='#b45309' if col == '#f59e0b' else '#1d4ed8', fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['0.01', '0.02', '0.05', '0.10'])
    ax.set_xlabel('alpha_r (ransomware risk budget)')
    ax.set_title(title, fontsize=10)
    ax.grid(True, ls=':', alpha=0.5)
axes[0].set_ylim(0, 72)
axes[1].set_ylim(-1.6, 10.5)
axes[2].set_ylim(45, 96)
axes[0].legend(loc='upper right', frameon=False, fontsize=9)
fig.suptitle('Mondrian Conformal Router Across Risk Budgets: Logistic Regression vs. Random Forest', fontsize=11.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_trust_escalation_both.png', dpi=180)
plt.close(fig)

# 3. Error-Detection AUROC comparison (how well uncertainty flags Stage-1 mistakes)
fig, ax = plt.subplots(figsize=(8.2, 4.0))
methods = ['credibility', 'maxsoft', 'entropy', 'random']
m_labels = ['Conformal\nCredibility', 'Max-Softmax\n(1 - max p)', 'Shannon\nEntropy', 'Random\nEscalation']
w = 0.36
x = np.arange(len(methods))
for i, (curves, lbl, col) in enumerate([(rf_curves, 'On Random Forest (Old)', '#f59e0b'),
                                        (lr_curves, 'On Logistic Regression (New)', '#2563eb')]):
    sub = curves[curves['rate'] == -1].groupby('method')['err_auroc'].agg(['mean', 'std']).reindex(methods)
    xpos = x + (i - 0.5) * w
    ax.bar(xpos, sub['mean'], w * 0.92, yerr=sub['std'], capsize=4, color=col, alpha=0.9, label=lbl)
    for xi, val in zip(xpos, sub['mean']):
        ax.text(xi, 0.06, f'{val:.3f}', ha='center', fontsize=9.5, fontweight='bold', color='white')
ax.axhline(0.5, color='grey', ls=':', lw=1.2, label='Coin flip (0.50)')
ax.set_xticks(x)
ax.set_xticklabels(m_labels)
ax.set_ylim(0, 1.18)
ax.set_ylabel('Error-Detection AUROC (higher is better)')
ax.set_title('How Well Uncertainty Scores Catch Stage-1 Mistakes on Unseen Families')
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_trust_auroc_both.png', dpi=180)
plt.close(fig)

# 4. Cost Proxy: Expected Latency per File (s) vs Escalation Budget
fig, ax = plt.subplots(figsize=(8.4, 3.9))
t_dyn = 9.1
t_lr = 0.00195
t_rf = 0.02155
alphas_str = ['alpha_r=0.01\n(strictest)', 'alpha_r=0.02', 'alpha_r=0.05\n(operating pt)', 'alpha_r=0.10\n(loosest)']
x = np.arange(4)
w = 0.36
rf_esc = rf_runs[(rf_runs['method'] == 'conformal') & (rf_runs['calib'] == 'group')].groupby('alpha_r')['esc_rate'].mean().to_numpy()
lr_esc = lr_runs[(lr_runs['method'] == 'conformal') & (lr_runs['calib'] == 'group')].groupby('alpha_r')['esc_rate'].mean().to_numpy()
rf_cost = t_rf + rf_esc * t_dyn
lr_cost = t_lr + lr_esc * t_dyn

ax.axhline(t_dyn, color='#dc2626', ls='--', lw=1.6, label='100% Dynamic Sandbox Analysis (9.10 s / file)')
ax.bar(x - w/2, rf_cost, w * 0.92, color='#f59e0b', alpha=0.9, label='RF + Conformal Cascade')
ax.bar(x + w/2, lr_cost, w * 0.92, color='#2563eb', alpha=0.9, label='LR + Conformal Cascade (Primary)')
for xi, v in zip(x - w/2, rf_cost):
    ax.text(xi, v + 0.18, f'{v:.2f}s', ha='center', fontsize=9, fontweight='bold', color='#b45309')
for xi, v in zip(x + w/2, lr_cost):
    ax.text(xi, v + 0.18, f'{v:.2f}s', ha='center', fontsize=9, fontweight='bold', color='#1d4ed8')
ax.set_xticks(x)
ax.set_xticklabels(alphas_str)
ax.set_ylim(0, 11.8)
ax.set_ylabel('Expected Time per File (seconds)')
ax.set_title('Cost Proxy: Expected Analysis Time per File (t_static + esc_rate * 9.1s)')
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_cost_proxy.png', dpi=180)
plt.close(fig)
print("Saved all 4 project summary figures successfully.")
