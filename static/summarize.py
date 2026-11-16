# usage: python -m static.summarize
# Reads artifacts/static_summary*_seed*.json and writes results/static_runs.csv and results/static_summary.md
import glob
import json
import os
import pandas as pd

SETTINGS = [('random_oof', 'Random-split CV (families seen)'),
            ('group_oof', 'Family-wise OOF (unseen families)'),
            ('test', 'Held-out test families')]
MODELS = [('lr', 'Logistic regression (primary)'), ('rf', 'Random Forest (comparison)')]


def main():
    rows = []
    for f in sorted(glob.glob('artifacts/static_summary*_seed*.json')):
        d = json.load(open(f))
        for key, _ in SETTINGS:
            rows.append(dict(model=d.get('model', 'rf'), seed=d['seed'], setting=key, n_test=d['n_test'],
                             latency_ms=d['latency_ms'], **d[key]))
    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/static_runs.csv', index=False)

    lines = []
    for mk, mlabel in MODELS:
        sub = df[df.model == mk]
        if sub.empty:
            continue
        lines += [f'**{mlabel}**, mean +/- std over {sub.seed.nunique()} seeds', '',
                  '| Setting | Accuracy | Recall | False-positive rate | AUC |',
                  '| --- | --- | --- | --- | --- |']
        for key, label in SETTINGS:
            g = sub[sub.setting == key]
            cells = [f'{g[m].mean():.3f} +/- {g[m].std():.3f}' for m in ('acc', 'recall', 'fpr', 'auc')]
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        t = sub[sub.setting == 'test']
        lines += ['', f'Test files per seed: {t.n_test.min()} to {t.n_test.max()}; '
                      f'inference latency {sub.latency_ms.mean():.2f} ms per file (excludes feature extraction)', '']
    text = '\n'.join(lines)
    open('results/static_summary.md', 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
