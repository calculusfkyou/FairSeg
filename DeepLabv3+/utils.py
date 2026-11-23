import numpy as np
from medpy import metric

from datasets.dataset_fairseg import GROUP_NAMES

METRICS = ('cup_dice', 'rim_dice', 'cup_iou', 'rim_iou')


def _dice_iou(pred, gt):
    """Official TransUNet calculate_metric_percase without HD95: empty GT with a non-empty prediction scores 1."""
    if pred.sum() > 0 and gt.sum() > 0:
        return float(metric.binary.dc(pred, gt)), float(metric.binary.jc(pred, gt))
    if pred.sum() > 0:
        return 1.0, 1.0
    return 0.0, 0.0


def case_metrics(pred, gt):
    """Dice and IoU of rim (label 1) and cup (label 2) for one 0/1/2 label map."""
    m = {}
    for part, c in (('rim', 1), ('cup', 2)):
        m[f'{part}_dice'], m[f'{part}_iou'] = _dice_iou(pred == c, gt == c)
    return m


def equity_scaled(overall, group_means):
    """FairSeg equity_scaled_perf with alpha = 1."""
    return overall / (1 + sum(abs(g - overall) for g in group_means))


def summarize(records, attr):
    """Overall (Unknown included), per-group (Unknown excluded) and ES metrics; empty groups are left out of the ES sum."""
    out = {k: float(np.mean([r[k] for r in records])) for k in METRICS}
    groups = {}
    for code, name in enumerate(GROUP_NAMES[attr]):
        members = [r for r in records if r['attrs'][attr] == code]
        groups[name] = {'n': len(members),
                        **{k: float(np.mean([r[k] for r in members])) if members else None for k in METRICS}}
    for k in METRICS:
        part, kind = k.split('_')
        out[f'{part}_es_{kind}'] = equity_scaled(out[k], [g[k] for g in groups.values() if g['n']])
    out['mean_dice'] = (out['cup_dice'] + out['rim_dice']) / 2
    out['mean_es_dice'] = (out['cup_es_dice'] + out['rim_es_dice']) / 2
    out['groups'] = groups
    return out
