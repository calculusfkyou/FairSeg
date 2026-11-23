"""I-score + BDA on one checkpoint (spec §10): cache the val layer4 (Z) and layer1 outputs, run BDA on
the GAP of Z, evaluate every growing prefix union U_K on val with those channels of Z set to 0, choose K.

  python bda.py --root_path <DATA> --checkpoint <ckpt> --attribute ethnicity --output_dir <dir>
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_fairseg import ATTRS, FairSegDataset, TestGenerator
from iscore import binarize, prefix_unions, rank_unique, run_bda
from networks.deeplabv3plus import DeepLabV3Plus, apply_channel_mask
from utils import case_metrics, predict_records, summarize

# Bit-for-bit reproducible inference: the cache check below compares per-image Dice exactly.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

COLUMNS = ('K', 'n_channels', 'cup_dice', 'rim_dice', 'cup_es_dice', 'rim_es_dice', 'mean_dice', 'mean_es_dice')

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='HF Dataset folder (contains Training/ and Test/)')
parser.add_argument('--list_dir', default='lists/FairSeg_split', help='folder with val.txt')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--attribute', required=True, choices=ATTRS, help='binary attribute that defines Y')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--n_runs', type=int, default=10000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=4)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def iscore_inputs(features, codes):
    """Drop Unknown (-1), then binarize at the per-channel median of the remaining samples."""
    codes = np.asarray(codes)
    if not np.isin(codes, (-1, 0, 1)).all():
        raise ValueError(f'I-score needs a binary attribute (codes -1/0/1), got {sorted(set(codes.tolist()))}')
    keep = np.flatnonzero(codes != -1)
    X, medians = binarize(features[keep])
    return X, codes[keep], medians, keep


def choose_k(rows, metric):
    """Row with the highest metric; ties go to the smaller union (so a tie with K = 0 keeps no mask)."""
    return max(rows, key=lambda r: (r[metric], -r['n_channels']))


def extract_val(model, loader):
    """Z and layer1 output of every image in loader order, kept on CPU."""
    model.eval()
    device = next(model.parameters()).device
    n, i = len(loader.dataset), 0
    with torch.no_grad():
        for batch in loader:
            z, low = model.encode(batch['image'].to(device))
            if i == 0:   # preallocate: collecting and concatenating would double the ~5 GB peak
                cache = {'z': torch.empty(n, *z.shape[1:]), 'low': torch.empty(n, *low.shape[1:]),
                         'labels': np.empty((n, *batch['label'].shape[1:]), np.uint8), 'attrs': [], 'names': []}
            b = len(z)
            cache['z'][i:i + b], cache['low'][i:i + b] = z.cpu(), low.cpu()
            cache['labels'][i:i + b] = batch['label'].numpy()
            cache['attrs'] += [dict(zip(ATTRS, a.tolist())) for a in batch['attrs']]
            cache['names'] += list(batch['name'])
            i += b
    return cache


def records_from_cache(model, cache, channels, batch_size=50):
    """predict_records from the cache: rerun only ASPP + decoder, with `channels` of Z set to 0."""
    model.eval()
    device = next(model.parameters()).device
    labels, records = cache['labels'], []
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            z = apply_channel_mask(cache['z'][i:i + batch_size].to(device), channels)
            logits = model.decode(z, cache['low'][i:i + batch_size].to(device), labels.shape[-2:])
            for j, pred in enumerate(logits.argmax(1).cpu().numpy(), i):
                records.append({**case_metrics(pred, labels[j]), 'attrs': cache['attrs'][j], 'name': cache['names'][j]})
    return records


def main(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = DeepLabV3Plus(pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    db = FairSegDataset(args.root_path, os.path.join(args.list_dir, 'val.txt'), TestGenerator())
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    cache = extract_val(model, loader)
    log(f'cached Z and layer1 of {len(db)} val images ({args.checkpoint}, epoch {ckpt["epoch"]})')

    # spec §13: with no mask and the same batch size (1), the cache path must match the full model exactly.
    full, cached = predict_records(model, loader), records_from_cache(model, cache, [], batch_size=1)
    bad = [f['name'] for f, c in zip(full, cached)
           if (f['name'], f['cup_dice'], f['rim_dice']) != (c['name'], c['cup_dice'], c['rim_dice'])]
    if bad or len(full) != len(cached):
        raise RuntimeError(f'cache path differs from the full model on {len(bad)} images, e.g. {bad[:3]}')
    log('cache check passed: per-image cup/rim Dice identical to the full model')
    # 4.8 GB on GPU 0 saves copying the whole cache from RAM for every union; values are unchanged.
    cache['z'], cache['low'] = cache['z'].cuda(), cache['low'].cuda()

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)
    with open(out('args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    codes = np.array([a[args.attribute] for a in cache['attrs']])
    X, y, medians, keep = iscore_inputs(cache['z'].mean(dim=(2, 3)).cpu().numpy(), codes)
    np.save(out('medians.npy'), medians)
    log(f'I-score samples: {len(y)} ({int(y.sum())} with {args.attribute} = 1); {len(codes) - len(y)} Unknown excluded')

    t = time.time()
    runs = run_bda(X, y, args.n_runs, seed=args.seed)
    ranked = rank_unique(runs)
    unions = dict(prefix_unions(ranked))
    with open(out('bda_runs.json'), 'w') as f:
        json.dump(runs, f)
    with open(out('ranked.json'), 'w') as f:
        json.dump([{'channels': list(s), 'score': score} for s, score in ranked], f)
    log(f'BDA: {args.n_runs} runs in {time.time() - t:.0f} s, {len(ranked)} distinct subsets, '
        f'{len(unions)} unions to evaluate (largest {max(map(len, unions.values()), default=0)} channels)')

    candidates, rows, t = {0: (), **unions}, [], time.time()
    with open(out('union_val.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        for n, (k, union) in enumerate(candidates.items(), 1):
            s = summarize(records_from_cache(model, cache, list(union)), args.attribute)
            row = {'K': k, 'n_channels': len(union), **{c: s[c] for c in COLUMNS[2:]}}
            writer.writerow(row)
            f.flush()
            rows.append(row)
            if n == 1 or n % 100 == 0 or n == len(candidates):
                log(f'{n}/{len(candidates)} evaluated, K {k}, {len(union)} channels: mean Dice {row["mean_dice"]:.4f}, '
                    f'mean ES-Dice {row["mean_es_dice"]:.4f} ({time.time() - t:.0f} s)')

    selection = {}
    for key, metric in (('dice', 'mean_dice'), ('es', 'mean_es_dice')):
        best = choose_k(rows, metric)
        selection[key] = {'K': best['K'], 'channels': list(candidates[best['K']])}
        log(f'K_{key}: K {best["K"]}, {best["n_channels"]} channels, val mean Dice {best["mean_dice"]:.4f}, '
            f'mean ES-Dice {best["mean_es_dice"]:.4f}')
    with open(out('selection.json'), 'w') as f:
        json.dump(selection, f, indent=2)


# Windows DataLoader workers re-import this file (spawn), so everything that builds
# models or starts workers must stay behind this guard.
if __name__ == '__main__':
    main(parser.parse_args())
