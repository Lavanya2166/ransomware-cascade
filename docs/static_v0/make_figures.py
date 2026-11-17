# usage (from repo root): python docs/static_v0/make_figures.py
# Rebuilds every figure and number used in the Static Stage v0 explainer, straight from the
# saved prediction files in artifacts/. Nothing here retrains the RF or LR runs; it only reads them.
# The one exception is the "top behaviours" chart, which fits the same LR pipeline on seed 0's dev set.
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

OUT = 'docs/static_v0/figures'
os.makedirs(OUT, exist_ok=True)
C = {'rf': '#f59e0b', 'lr': '#2563eb'}
NAME = {'rf': 'Random Forest (old)', 'lr': 'Logistic Regression (new)'}
SETTINGS = [(('random', 'oof'), 'random_oof', 'Random split\n(families seen)'),
            (('group', 'oof'), 'group_oof', 'Practice exam\n(unseen families)'),
            (('group', 'test'), 'test', 'Final exam\n(unseen families)')]
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})


def load(m):
    stem = '' if m == 'lr' else '_rf'
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'artifacts/preds_static{stem}_seed*.parquet'))])


P = {m: load(m) for m in ('rf', 'lr')}


def part(m, cv, role):
    d = P[m]
    return d[(d.cv_mode == cv) & (d.role == role)]


def mets(d):
    y, p = d.y.to_numpy(), d.p_ransom.to_numpy()
    pred = (p >= 0.5).astype(int)
    return dict(acc=(pred == y).mean(), recall=pred[y == 1].mean(), fpr=pred[y == 0].mean(),
                auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                tp=int(((pred == 1) & (y == 1)).sum()), fn=int(((pred == 0) & (y == 1)).sum()),
                fp=int(((pred == 1) & (y == 0)).sum()), tn=int(((pred == 0) & (y == 0)).sum()), n=len(y))


rows = []
for m in P:
    for (cv, role), key, _ in SETTINGS:
        for s, d in part(m, cv, role).groupby('seed'):
            rows.append(dict(model=m, setting=key, seed=int(s), **mets(d)))
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/../per_seed_metrics.csv', index=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{OUT}/{name}.png', dpi=170)
    plt.close(fig)
    print('saved', name)


# 1. Accuracy by setting, both models, with each seed as a dot
fig, ax = plt.subplots(figsize=(9, 4.6))
w = 0.38
for i, m in enumerate(('rf', 'lr')):
    for j, (_, key, label) in enumerate(SETTINGS):
        v = R[(R.model == m) & (R.setting == key)].sort_values('seed').acc.to_numpy() * 100
        x = j + (i - 0.5) * w
        ax.bar(x, v.mean(), w * 0.92, color=C[m], alpha=0.9, label=NAME[m] if j == 0 else None)
        ax.scatter(np.full(5, x) + np.linspace(-0.08, 0.08, 5), v, color='black', s=13, zorder=3)
        ax.text(x, v.mean() + 1.2, f'{v.mean():.1f}%', ha='center', fontsize=10.5, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels([s[2] for s in SETTINGS])
ax.set_ylim(50, 104); ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy: bars = average of 5 seeds, dots = each seed')
ax.legend(loc='lower left', frameon=False)
save(fig, 'fig_accuracy_by_setting')

# 2. Per-seed practice-exam (family-wise OOF) comparison: accuracy and false alarms
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
for ax, (k, title, fmt) in zip(axes, [('acc', 'Accuracy (higher is better)', '%'),
                                      ('fpr', 'False-alarm rate (lower is better)', '%'),
                                      ('auc', 'AUC (higher is better)', '')]):
    for m in ('rf', 'lr'):
        v = R[(R.model == m) & (R.setting == 'group_oof')].sort_values('seed')[k].to_numpy()
        v = v * 100 if fmt == '%' else v
        ax.plot(range(5), v, 'o-', color=C[m], lw=2, label=NAME[m])
    ax.set_xticks(range(5)); ax.set_xticklabels([f'seed {s}' for s in range(5)], fontsize=9)
    ax.set_title(title, fontsize=11)
axes[0].legend(frameon=False, fontsize=9, loc='lower left')
fig.suptitle('Practice exam on unseen families, seed by seed: Logistic Regression wins on all 5 seeds', fontsize=12)
save(fig, 'fig_per_seed_practice')

# 3. Confusion matrices pooled over 5 seeds (practice exam and final exam)
fig, axes = plt.subplots(2, 2, figsize=(9, 7.6))
for r, ((cv, role), key, label) in enumerate([SETTINGS[1], SETTINGS[2]]):
    for c, m in enumerate(('rf', 'lr')):
        ax = axes[r, c]
        g = R[(R.model == m) & (R.setting == key)][['tp', 'fn', 'fp', 'tn']].sum()
        M = np.array([[g.tn, g.fp], [g.fn, g.tp]])
        ax.imshow([[1, 0], [0, 1]], cmap='Greens', vmin=-1.5, vmax=2.2)
        names = [['correct\nbenign', 'FALSE\nALARM'], ['MISSED\nransomware', 'caught\nransomware']]
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f'{M[a, b]:,}\n{names[a][b]}', ha='center', va='center', fontsize=10,
                        fontweight='bold' if a != b else 'normal', color='#b91c1c' if a != b else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['said benign', 'said ransomware'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['really benign', 'really\nransomware'])
        ax.set_title(f"{NAME[m]}\n{label.replace(chr(10), ' ')}", fontsize=10.5, color=C[m], fontweight='bold')
        for s in ax.spines.values(): s.set_visible(False)
fig.suptitle('All 5 seeds added together', fontsize=12)
save(fig, 'fig_confusion')

# 4. ROC curves pooled, practice exam
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, ((cv, role), key, label) in zip(axes, [SETTINGS[1], SETTINGS[2]]):
    for m in ('rf', 'lr'):
        d = part(m, cv, role)
        f, t, _ = roc_curve(d.y, d.p_ransom)
        mean_auc = R[(R.model == m) & (R.setting == key)].auc.mean()
        ax.plot(f, t, color=C[m], lw=2.2, label=f'{NAME[m]}  (mean AUC {mean_auc:.3f})')
    ax.plot([0, 1], [0, 1], ':', color='grey')
    ax.set_xlabel('False-alarm rate'); ax.set_ylabel('Ransomware caught (recall)')
    ax.set_title(label.replace('\n', ' '), fontsize=11)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
save(fig, 'fig_roc')

# 5. Reliability (calibration) curves, practice exam
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for m in ('rf', 'lr'):
    d = part(m, 'group', 'oof')
    fr, mp = calibration_curve(d.y, d.p_ransom, n_bins=10, strategy='quantile')
    b = R[(R.model == m) & (R.setting == 'group_oof')].brier.mean()
    ax.plot(mp, fr, 'o-', color=C[m], lw=2, label=f'{NAME[m]}  (Brier {b:.3f})')
ax.plot([0, 1], [0, 1], ':', color='grey', label='perfectly honest scores')
ax.set_xlabel('What the model said (probability of ransomware)')
ax.set_ylabel('What was actually true (share ransomware)')
ax.set_title('Are the confidence scores honest?\n(practice exam, unseen families)', fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
save(fig, 'fig_calibration')

# 6. Score histograms on the practice exam: how well each model separates the two classes
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, m in zip(axes, ('rf', 'lr')):
    d = part(m, 'group', 'oof')
    bins = np.linspace(0, 1, 21)
    ax.hist(d[d.y == 0].p_ransom, bins, color='#10b981', alpha=0.7, label='really benign')
    ax.hist(d[d.y == 1].p_ransom, bins, color='#ef4444', alpha=0.6, label='really ransomware')
    ax.axvline(0.5, color='black', ls='--', lw=1)
    ax.set_title(NAME[m], color=C[m], fontweight='bold')
    ax.set_xlabel('Score given (probability of ransomware)')
axes[0].set_ylabel('Number of files (5 seeds)'); axes[0].legend(frameon=False)
fig.suptitle('Practice exam: green to the right of the dashed line = false alarms', fontsize=11.5)
save(fig, 'fig_score_hist')

# 7. Latency
lat = {}
for f in glob.glob('artifacts/static_summary*_seed*.json'):
    j = json.load(open(f)); lat.setdefault(j.get('model', 'rf'), []).append(j['latency_ms'])
fig, ax = plt.subplots(figsize=(6.5, 2.4))
for i, m in enumerate(('rf', 'lr')):
    v = np.mean(lat[m]); ax.barh(i, v, color=C[m]); ax.text(v + 0.3, i, f'{v:.1f} ms', va='center', fontweight='bold')
ax.set_yticks([0, 1]); ax.set_yticklabels([NAME['rf'], NAME['lr']]); ax.set_xlim(0, 26)
ax.set_xlabel('Time to score one file (milliseconds, model only, average of 5 seeds)')
save(fig, 'fig_latency')

# 8. Top behaviours the LR model uses (seed 0 dev set, same pipeline as static/model.py)
from common.config import ID, LABEL, FAMILY
from common.splits import outer_split
from static.groups import regroup_benign
from static.model import make_static_model
names = [ln.rstrip('\n').split(';', 1)[1] for ln in open('data/raw/VariableNames.txt', encoding='latin-1')
         if ln.strip()][3:]
meta = pd.read_parquet('data/master.parquet', columns=[ID, LABEL, FAMILY]).reset_index(drop=True)
X = sparse.load_npz('data/X_sparse.npz').tocsr()
assert len(names) == X.shape[1]
meta = regroup_benign(meta, X)
dev, _ = outer_split(meta, 0)
mdl = make_static_model(None, 0, 'lr').fit(X[dev], meta[LABEL].to_numpy()[dev])
kept = np.array(names)[mdl.named_steps['var'].get_support()]
coef = mdl.named_steps['lr'].coef_[0]
order = np.argsort(coef)
top = list(order[:8]) + list(order[-10:])
fig, ax = plt.subplots(figsize=(9, 6))
lab = [(kept[i][:42] + '...') if len(kept[i]) > 45 else kept[i] for i in top]
ax.barh(range(len(top)), coef[top], color=['#10b981' if coef[i] < 0 else '#ef4444' for i in top])
ax.set_yticks(range(len(top))); ax.set_yticklabels(lab, fontsize=8.5)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Weight\n(red = pushes towards ransomware, green = pushes towards benign)')
ax.set_title('Biggest weights in the Logistic Regression model (seed 0)', fontsize=11)
save(fig, 'fig_top_features')

cats = pd.Series([n.split(':', 1)[0] for n in names]).value_counts()
extra = dict(latency_ms={m: float(np.mean(v)) for m, v in lat.items()},
             n_features_after_variance_filter_seed0=int(len(kept)),
             feature_categories=cats.to_dict(),
             top_ransom=[kept[i] for i in order[::-1][:10]], top_benign=[kept[i] for i in order[:8]])
json.dump(extra, open(f'{OUT}/../extra_stats.json', 'w'), indent=2)
print(json.dumps(extra, indent=2))

