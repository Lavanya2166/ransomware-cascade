import numpy as np
from trust import conformal as C


def test_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)

    def draw(n):
        y = rng.integers(0, 2, n)
        conf = np.clip(rng.normal(0.75, 0.15, n), 0.01, 0.99)
        p = np.where(y[:, None] == np.arange(2), conf[:, None], (1.0 - conf)[:, None])
        return p, y

    pc, yc = draw(4000)
    pt, yt = draw(4000)
    sets = C.prediction_sets(C.pvalues(pt, C.cal_scores(pc, yc)), (0.10, 0.10))
    for c in (0, 1):
        assert abs(sets[yt == c, c].mean() - 0.90) < 0.03