"""I-score and the backward dropping algorithm (spec §10.3-10.6). Inputs are NumPy:
X is 0/1 [n, d] (one column per layer4 channel), y is 0/1 [n]."""
import numpy as np

TOL = 1e-12


def binarize(features):
    """1 where a channel is above its own median over the samples, else 0."""
    medians = np.median(features, axis=0)
    return features > medians, medians


def iscore(X, y, subset):
    """I(S) = sum_j n_j^2 (Ybar_j - Ybar)^2 / (n s^2), cells j = 0/1 patterns of X on S."""
    subset = list(subset)
    y = np.asarray(y, dtype=np.float64)
    var = y.var()
    if not subset or var == 0:
        return 0.0
    # shortcut: cells are binary codes, so S is limited to 62 channels; BDA subsets start at 3.
    codes = X[:, subset].astype(np.int64) @ (1 << np.arange(len(subset), dtype=np.int64))
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=y)
    # n_j^2 (Ybar_j - Ybar)^2 = (sum_j - n_j Ybar)^2, which is 0 for empty cells
    return float(((sums - counts * y.mean()) ** 2).sum() / (len(y) * var))


def bda_once(X, y, start):
    """Drop the channel with the largest gain while that gain is > 0; ties drop the smallest index."""
    current = sorted(int(c) for c in start)
    score = iscore(X, y, current)
    dropped = []
    while current:
        scores = [iscore(X, y, [c for c in current if c != d]) for d in current]
        best = max(scores)
        if best - score <= TOL:
            break
        i = next(i for i, s in enumerate(scores) if s >= best - TOL)
        dropped.append(current.pop(i))
        score = scores[i]
    return tuple(current), score, dropped


def run_bda(X, y, n_runs=10000, start_size=3, seed=42):
    rng = np.random.default_rng(seed)
    runs = []
    for run in range(n_runs):
        start = [int(c) for c in rng.choice(X.shape[1], start_size, replace=False)]
        final, score, dropped = bda_once(X, y, start)
        runs.append({'run': run, 'start': start, 'dropped': dropped, 'final': final, 'score': score})
    return runs


def rank_unique(runs):
    """Distinct final subsets, highest score first; equal scores keep run order."""
    seen, ranked = set(), []
    for r in sorted(runs, key=lambda r: (-r['score'], r['run'])):
        final = tuple(r['final'])
        if final not in seen:
            seen.add(final)
            ranked.append((final, r['score']))
    return ranked


def prefix_unions(ranked):
    """(K, U_K) for K = 1..M, only where the union of the top-K subsets grows."""
    union, out = set(), []
    for k, (subset, _) in enumerate(ranked, 1):
        if not union.issuperset(subset):
            union.update(subset)
            out.append((k, tuple(sorted(union))))
    return out
