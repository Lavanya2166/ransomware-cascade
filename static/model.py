from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def make_static_model(k=None, seed=0):
    # Features are binary 0/1, so no log or scaling step is needed.
    steps = [('var', VarianceThreshold(0.0))]
    if k is not None:
        score = partial(mutual_info_classif, discrete_features=True, random_state=seed)
        steps.append(('mi', SelectKBest(score, k=k)))
    steps.append(('rf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                               class_weight='balanced_subsample')))
    return Pipeline(steps)
