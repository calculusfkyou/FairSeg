import hashlib
import io
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric

from datasets.dataset_fairseg import ATTRS, GROUP_NAMES

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
        if not members:   # spec §6: shown once per attribute and group
            warnings.warn(f'{attr}: no {name} images, so {name} is left out of the ES sum')
        groups[name] = {'n': len(members),
                        **{k: float(np.mean([r[k] for r in members])) if members else None for k in METRICS}}
    for k in METRICS:
        part, kind = k.split('_')
        out[f'{part}_es_{kind}'] = equity_scaled(out[k], [g[k] for g in groups.values() if g['n']])
    out['mean_dice'] = (out['cup_dice'] + out['rim_dice']) / 2
    out['mean_es_dice'] = (out['cup_es_dice'] + out['rim_es_dice']) / 2
    out['groups'] = groups
    return out


class CEDiceLoss(nn.Module):
    """0.5 · CE (3 classes, unweighted) + 0.5 · soft Dice loss over rim and cup (spec §8).

    Dice sums over the whole batch, the denominator is not squared, and background is left out.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)
        prob = logits.softmax(1)[:, 1:]
        onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].to(prob.dtype)
        dims = (0, 2, 3)
        per_class = (2 * (prob * onehot).sum(dims) + self.eps) / (prob.sum(dims) + onehot.sum(dims) + self.eps)
        dice = 1 - per_class.mean()
        return 0.5 * ce + 0.5 * dice, ce, dice


def predict_records(model, loader, mask_channels=None):
    """Per-image metrics (argmax prediction) with each image's 'attrs' and 'name'."""
    # Unwrap DataParallel: it would re-replicate the model for every batch-1 val image.
    net = model.module if isinstance(model, nn.DataParallel) else model
    net.eval()
    device = next(net.parameters()).device
    records = []
    with torch.no_grad():
        for batch in loader:
            preds = net(batch['image'].to(device), mask_channels=mask_channels).argmax(1).cpu().numpy()
            for i, pred in enumerate(preds):
                rec = case_metrics(pred, batch['label'][i].numpy())
                rec['attrs'] = dict(zip(ATTRS, batch['attrs'][i].tolist()))
                rec['name'] = batch['name'][i]
                records.append(rec)
    return records


def ensure_new(path):
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists; results are never overwritten, choose another output')


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_checkpoint(obj, path, retries=3):
    """Save obj to path without torch.save's own file writer (copied from TransUNet/trainer.py).

    On this Windows setup, torch.save(obj, filename) corrupted checkpoints at a fixed byte offset.
    So: serialize into memory, write the bytes with plain file I/O, check the file loads back,
    then rename it into place. Later attempts fall back to the legacy (non-zip) format."""
    tmp = path + '.tmp'
    for attempt in range(1, retries + 1):
        try:
            buf = io.BytesIO()
            torch.save(obj, buf, _use_new_zipfile_serialization=(attempt == 1))
            with open(tmp, 'wb') as f:
                f.write(buf.getbuffer())
                f.flush()
                os.fsync(f.fileno())
            torch.load(tmp, map_location='cpu', weights_only=True)
            os.replace(tmp, path)
            return
        except Exception as e:
            logging.warning('saving %s failed (attempt %d/%d): %s', path, attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(30)


# Module level so DataLoader workers can pickle it: Windows starts workers with spawn.
def seed_worker(seed, worker_id):
    random.seed(seed + worker_id)


def state_dict_of(model):
    """state_dict without DataParallel's 'module.' prefix."""
    return (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
