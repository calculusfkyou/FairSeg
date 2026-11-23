"""Stratified 7000/1000 train/val split of the official FairSeg train list (spec §5.3)."""
import argparse
import csv
import os
import random
import shutil

import numpy as np

from datasets.dataset_fairseg import ATTRS, GROUP_NAMES, index_npz_files, read_attrs


def stratified_split(names, keys, n_val, seed):
    """Largest-remainder quotas per stratum, random ties; both lists keep the order of names."""
    rng = random.Random(seed)
    strata = {}
    for name, key in zip(names, keys):
        strata.setdefault(key, []).append(name)
    total = len(names)
    take = {k: len(v) * n_val // total for k, v in strata.items()}
    remainder = {k: len(v) * n_val % total for k, v in strata.items()}   # integers, so ties are exact
    order = list(strata)
    rng.shuffle(order)
    order.sort(key=lambda k: remainder[k], reverse=True)   # stable: ties stay in the shuffled order
    for k in order[:n_val - sum(take.values())]:
        take[k] += 1
    val = set()
    for k, members in strata.items():
        val.update(rng.sample(members, take[k]))
    return [n for n in names if n not in val], [n for n in names if n in val]


def split_report(attrs, train, val, attr_names=ATTRS):
    """Group counts and percentages per attribute; percentages exclude Unknown (-1)."""
    rows = []
    for a in attr_names:
        n_groups = len(GROUP_NAMES[a])
        counts = {}
        for split, members in (('train', train), ('val', val)):
            codes = [attrs[n][a] for n in members]
            counts[split] = {c: codes.count(c) for c in list(range(n_groups)) + [-1]}
            counts[split]['known'] = len(codes) - counts[split][-1]
        for c in list(range(n_groups)) + [-1]:
            row = {'attr': a, 'group': GROUP_NAMES[a][c] if c >= 0 else 'Unknown',
                   'n_train': counts['train'][c], 'n_val': counts['val'][c],
                   'pct_train': None, 'pct_val': None, 'diff': None}
            if c >= 0 and counts['train']['known'] and counts['val']['known']:
                pt = 100 * counts['train'][c] / counts['train']['known']
                pv = 100 * counts['val'][c] / counts['val']['known']
                row.update(pct_train=round(pt, 2), pct_val=round(pv, 2), diff=round(pv - pt, 2))
            rows.append(row)
    return rows


def read_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True, help='HF Dataset folder (contains Training/ and Test/)')
    parser.add_argument('--official_dir', default='../TransUNet/lists/FairSeg_final')
    parser.add_argument('--out_dir', default='lists/FairSeg_split')
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    official_train = read_list(os.path.join(args.official_dir, 'train.txt'))
    official_test = read_list(os.path.join(args.official_dir, 'test.txt'))
    paths = index_npz_files(args.root_path)
    attrs = {}
    for n in official_train:
        with np.load(paths[n]) as data:
            attrs[n] = read_attrs(data)

    keys = [tuple(attrs[n][a] for a in ATTRS) for n in official_train]
    train, val = stratified_split(official_train, keys, args.n_val, args.seed)
    assert len(train) == len(official_train) - args.n_val and len(val) == args.n_val
    assert not set(train) & set(val) and set(train) | set(val) == set(official_train)
    assert not set(val) & set(official_test)

    os.makedirs(args.out_dir, exist_ok=True)
    for split, members in (('train', train), ('val', val)):
        with open(os.path.join(args.out_dir, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(members) + '\n')
    shutil.copyfile(os.path.join(args.official_dir, 'test.txt'), os.path.join(args.out_dir, 'test.txt'))

    rows = split_report(attrs, train, val)
    with open(os.path.join(args.out_dir, 'split_report.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f'strata: {len(set(keys))}, train {len(train)}, val {len(val)}, test {len(official_test)}')
    for a in ATTRS:
        worst = max(abs(r['diff']) for r in rows if r['attr'] == a and r['diff'] is not None)
        print(f'{a:14s} max |pct_val - pct_train| = {worst:.2f} pp')
    hisp = next(r for r in rows if r['attr'] == 'ethnicity' and r['group'] == 'Hispanic')
    print(f"val Hispanic: {hisp['n_val']}")


if __name__ == '__main__':
    main()
