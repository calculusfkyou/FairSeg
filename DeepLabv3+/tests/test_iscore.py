import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from iscore import bda_once, binarize, iscore, prefix_unions, rank_unique, run_bda

# I-score values checked against the senior's I_Score_function6.py on 2026-11-23.
X = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]])
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
NOISE = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
A = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1]); B = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0])


def test_binarize_uses_per_channel_median():
    f = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 7.0], [4.0, 5.0]])
    Xb, med = binarize(f)
    assert np.allclose(med, [2.5, 5.0])
    assert (Xb == np.array([[0, 0], [0, 0], [1, 1], [1, 0]])).all()   # equal to the median -> 0


def test_iscore_matches_senior_values():
    assert abs(iscore(X, y, [0, 1]) - 1.8) < 1e-12
    assert abs(iscore(X, y, [0]) - 0.2) < 1e-12 and abs(iscore(X, y, [1]) - 0.2) < 1e-12
    assert iscore(X, y, []) == 0 and iscore(X, np.zeros(10, int), [0]) == 0


def test_bda_drops_noise():
    X3 = np.column_stack([X, NOISE])                       # I({0,1,2}) = 1.0
    final, score, dropped = bda_once(X3, y, [0, 1, 2])
    assert final == (0, 1) and dropped == [2] and abs(score - 1.8) < 1e-12


def test_bda_zero_gain_does_not_drop():
    X4 = np.column_stack([X, NOISE, NOISE])                # dropping 2 or 3 both give gain exactly 0
    final, score, dropped = bda_once(X4, y, [0, 1, 2, 3])
    assert final == (0, 1, 2, 3) and dropped == [] and abs(score - 1.0) < 1e-12


def test_bda_tie_drops_smaller_index():
    X5 = np.column_stack([X, A, B])                        # dropping 2 or 3 both give 1.4
    final, score, dropped = bda_once(X5, y, [0, 1, 2, 3])
    assert dropped == [2, 3] and final == (0, 1) and abs(score - 1.8) < 1e-12


def test_run_bda_deterministic():
    rng = np.random.default_rng(0)
    Xr, yr = rng.integers(0, 2, (50, 20)), rng.integers(0, 2, 50)
    a, b = run_bda(Xr, yr, n_runs=30, seed=42), run_bda(Xr, yr, n_runs=30, seed=42)
    assert a == b and len(a) == 30
    assert all(len(set(r['start'])) == 3 and all(0 <= c < 20 for c in r['start']) for r in a)


def test_rank_and_unions():
    runs = [{'run': 0, 'final': (1, 2), 'score': 0.5}, {'run': 1, 'final': (3,), 'score': 0.9},
            {'run': 2, 'final': (1, 2), 'score': 0.5}, {'run': 3, 'final': (4,), 'score': 0.5}]
    assert rank_unique(runs) == [((3,), 0.9), ((1, 2), 0.5), ((4,), 0.5)]
    ranked = [((3,), .9), ((1, 3), .8), ((1,), .7), ((5,), .6)]
    assert prefix_unions(ranked) == [(1, (3,)), (2, (1, 3)), (4, (1, 3, 5))]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
