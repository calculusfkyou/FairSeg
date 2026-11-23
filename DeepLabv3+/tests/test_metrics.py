import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import warnings

import numpy as np

from utils import case_metrics, summarize


def test_case_metrics_edges():
    gt = np.zeros((4, 4), int); gt[0, :4] = 1                     # rim 4 px，沒有 cup
    pred = np.zeros((4, 4), int); pred[0, :2] = 1; pred[1, :2] = 1  # rim 4 px，重疊 2 px
    pred[3, 3] = 2                                                  # 預測有 cup、GT 沒有
    m = case_metrics(pred, gt)
    assert abs(m['rim_dice'] - 0.5) < 1e-9 and abs(m['rim_iou'] - 1 / 3) < 1e-9
    assert m['cup_dice'] == 1 and m['cup_iou'] == 1                  # 官方規則：預測非空、GT 空 → 1
    empty = case_metrics(np.zeros((4, 4), int), np.zeros((4, 4), int))
    assert empty['cup_dice'] == 0 and empty['rim_iou'] == 0          # 預測為空 → 0（含 GT 也空）


def _rec(cup, eth):
    return {'cup_dice': cup, 'rim_dice': 0.5, 'cup_iou': cup, 'rim_iou': 0.5,
            'attrs': {'ethnicity': eth, 'gender': 0}}


def test_summarize_es():
    s = summarize([_rec(0.8, 0), _rec(0.6, 0), _rec(0.4, 1), _rec(1.0, -1)], 'ethnicity')
    assert abs(s['cup_dice'] - 0.7) < 1e-9                           # overall 含 Unknown
    assert s['groups']['Hispanic']['n'] == 1 and s['groups']['Non-Hispanic']['n'] == 2
    assert abs(s['cup_es_dice'] - 0.7 / 1.3) < 1e-9                  # |0.7−0.7| + |0.4−0.7| = 0.3
    assert abs(s['rim_es_dice'] - 0.5) < 1e-9
    assert abs(s['mean_dice'] - 0.6) < 1e-9
    assert abs(s['mean_es_dice'] - (0.7 / 1.3 + 0.5) / 2) < 1e-9


def test_summarize_skips_empty_group():
    s = summarize([_rec(0.8, 0), _rec(0.6, 0)], 'gender')            # 沒有 Male
    assert s['groups']['Male']['n'] == 0
    assert math.isfinite(s['cup_es_dice']) and abs(s['cup_es_dice'] - 0.7) < 1e-9


def test_summarize_warns_on_empty_group():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        summarize([_rec(0.8, 0)], 'gender')                          # no Male
    assert any('gender' in str(w.message) and 'Male' in str(w.message) for w in caught)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
