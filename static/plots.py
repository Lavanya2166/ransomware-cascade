# usage: python -m static.plots [--model lr] [--dpi 200]
# Reads artifacts/ (summaries and predictions) and writes figures to results/static_*.png
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

SETTINGS = [('random_oof', 'Random split\n(families seen)'),
            ('group_oof', 'Family-wise OOF\n(unseen families)'),
            ('test', 'Held-out test\nfamilies')]
COLOR = {'lr': '#2563eb', 'rf': '#f59e0b'}
NAME = {'lr': 'Logistic regression', 'rf': 'Random Forest'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})


def load_summaries():
    rows = []
    for f in sorted(glob.glob('artifacts/static_summary*_seed*.json')):
        d = json.load(open(f))
        for key, _ in SETTINGS:
            rows.append(dict(model=d.get('model', 'rf'), seed=d['seed'], setting=key, **d[key]))
    return pd.DataFrame(rows)


def load_preds(model):
    stem = '' if model == 'lr' else f'_{model}'
    files = sorted(glob.glob(f'artifacts/preds_static{stem}_seed*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in files]) if files else None


def save(fig, name, dpi):
    os.makedirs('results', exist_ok=True)
    path = f'results/static_{name}.png'
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print('saved', path)


def plot_accuracy(S, dpi):
    models = [m for m in ('lr', 'rf') if m in S.model.unique()]
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, m in enumerate(models):
        for j, (key, _) in enumerate(SETTINGS):
            v = S[(S.model == m) & (S.setting == key)].acc.to_numpy() * 100
            x = j + (i - (len(models) - 1) / 2) * w
            sd = v.std(ddof=1) if len(v) > 1 else 0
            ax.bar(x, v.mean(), w * 0.95, yerr=sd, capsize=4, color=COLOR[m], alpha=0.85,
                   label=NAME[m] if j == 0 else None)
            ax.scatter([x] * len(v), v, color='black', s=14, zorder=3)
            ax.text(x, 5, f'{v.mean():.1f}', ha='center', fontsize=11, color='white', fontweight='bold')
    ax.set_xticks(range(len(SETTINGS)))
    ax.set_xticklabels([label for _, label in SETTINGS])
    ax.set_ylim(0, 112)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy drops on families the model has never seen\n(bars = mean over seeds, dots = single seeds, whiskers = std)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, 'accuracy_by_setting', dpi)


def plot_per_seed(S, model, dpi):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, label in SETTINGS:
        d = S[(S.model == model) & (S.setting == key)].sort_values('seed')
        ax.plot(d.seed, d.acc * 100, marker='o', lw=2, label=label.replace('\n', ' '))
    ax.set_xticks(sorted(S.seed.unique()))
    ax.set_xlabel('Seed (each seed hides different families)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 102)
    ax.set_title(f'{NAME[model]}: results change a lot with the hidden families')
    ax.legend(loc='lower left', fontsize=9)
    save(fig, 'accuracy_per_seed', dpi)


def plot_roc(P, model, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, (role, title) in zip(axes, (('oof', 'Family-wise OOF (unseen families)'),
                                        ('test', 'Held-out test families'))):
        sub = P[(P.cv_mode == 'group') & (P.role == role)]
        for s, g in sub.groupby('seed'):
            fpr, tpr, _ = roc_curve(g.y, g.p_ransom)
            ax.plot(fpr, tpr, lw=1.6, alpha=0.85, label=f'seed {s} (AUC {roc_auc_score(g.y, g.p_ransom):.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('False-positive rate')
        ax.legend(loc='lower right', fontsize=8)
    axes[0].set_ylabel('True-positive rate (ransomware recall)')
    fig.suptitle(f'{NAME[model]}: ROC curves per seed')
    save(fig, 'roc', dpi)


def plot_prob_hist(P, model, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, (role, title) in zip(axes, (('oof', 'Family-wise OOF'), ('test', 'Held-out test families'))):
        sub = P[(P.cv_mode == 'group') & (P.role == role)]
        ax.hist(sub[sub.y == 0].p_ransom, bins=20, range=(0, 1), alpha=0.65, density=True, color='#16a34a', label='Benign files')
        ax.hist(sub[sub.y == 1].p_ransom, bins=20, range=(0, 1), alpha=0.65, density=True, color='#dc2626', label='Ransomware files')
        ax.axvline(0.5, color='black', ls='--', lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('Predicted probability of ransomware')
        ax.legend()
    axes[0].set_ylabel('Density')
    fig.suptitle(f'{NAME[model]}: where the scores fall (all seeds pooled)')
    save(fig, 'probability_histograms', dpi)


def plot_reliability(P, model, dpi):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    bins = np.linspace(0, 1, 11)
    for role, label, color in (('oof', 'Family-wise OOF', '#2563eb'), ('test', 'Held-out test families', '#dc2626')):
        sub = P[(P.cv_mode == 'group') & (P.role == role)]
        idx = np.clip(np.digitize(sub.p_ransom, bins) - 1, 0, 9)
        g = sub.assign(b=idx).groupby('b').agg(pred=('p_ransom', 'mean'), obs=('y', 'mean'), n=('y', 'size'))
        g = g[g.n >= 10]
        ax.plot(g.pred, g.obs, marker='o', color=color, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfectly calibrated')
    ax.set_xlabel('Predicted probability of ransomware')
    ax.set_ylabel('Observed fraction that is ransomware')
    ax.set_title(f'{NAME[model]}: are the probabilities trustworthy?')
    ax.legend(loc='upper left', fontsize=9)
    save(fig, 'reliability', dpi)


def plot_family_accuracy(P, model, dpi):
    sub = P[(P.cv_mode == 'group') & (P.role == 'test')].copy()
    sub['correct'] = ((sub.p_ransom >= 0.5).astype(int) == sub.y)
    g = sub.groupby('family').agg(acc=('correct', 'mean'), n=('correct', 'size')).sort_values('acc')
    colors = ['#dc2626' if f.startswith('ransom') else '#16a34a' for f in g.index]
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.28 * len(g) + 1.5)))
    ax.barh([f'{f} (n={n})' for f, n in zip(g.index, g.n)], g.acc * 100, color=colors)
    ax.axvline(50, color='black', ls='--', lw=0.8)
    ax.set_xlabel('Accuracy on held-out test files (%)')
    ax.set_xlim(0, 100)
    ax.set_title(f'{NAME[model]}: which families are hard?\n(red = ransomware family, green = benign group)', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    save(fig, 'accuracy_per_family', dpi)


def plot_confusion(P, model, dpi):
    sub = P[(P.cv_mode == 'group') & (P.role == 'test')]
    cm = confusion_matrix(sub.y, (sub.p_ransom >= 0.5).astype(int))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}\n({pct[i, j]:.1f}%)', ha='center', va='center',
                    color='white' if pct[i, j] > 55 else 'black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Predicted\nbenign', 'Predicted\nransomware'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actually\nbenign', 'Actually\nransomware'])
    ax.set_title(f'{NAME[model]}: held-out test files (all seeds pooled)')
    save(fig, 'confusion_matrix', dpi)


def main(model, dpi):
    S = load_summaries()
    if S.empty:
        raise SystemExit('No summaries found. Run python -m static.run_static first.')
    plot_accuracy(S, dpi)
    if model in S.model.unique():
        plot_per_seed(S, model, dpi)
    P = load_preds(model)
    if P is None:
        print('No predictions for', model, '- skipping the prediction-based plots')
        return
    plot_roc(P, model, dpi)
    plot_prob_hist(P, model, dpi)
    plot_reliability(P, model, dpi)
    plot_family_accuracy(P, model, dpi)
    plot_confusion(P, model, dpi)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['lr', 'rf'], default='lr')
    ap.add_argument('--dpi', type=int, default=200)
    a = ap.parse_args()
    main(a.model, a.dpi)
