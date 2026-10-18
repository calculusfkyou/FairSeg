"""Collect SAMed / TransUNet test logs into one table and compare with the paper.

Reads the logs written by scripts/test_baselines.sh and scripts/paper_baselines.csv
(baseline rows of Tables 2-5 in the FairSeg paper), and writes
  outputs/baseline_test/results.csv      every number we measured
  outputs/baseline_test/comparison.md    ours vs paper, per attribute
"""
import csv, glob, os, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'outputs', 'baseline_test')
ATTRS = ['race', 'gender', 'language', 'ethnicity']
# Group index -> name, from the HF dataset card encodings (read_attr in dataset_fairseg.py)
GROUPS = {
    'race': ['Asian', 'Black', 'White'],
    'gender': ['Female', 'Male'],
    'language': ['English', 'Spanish', 'Others'],
    'ethnicity': ['Non-Hispanic', 'Hispanic'],
}


def find_log(model, attr):
    if model == 'SAMed':
        pattern = os.path.join(OUT, 'samed', f'*_{attr}', 'test_log', 'log.txt')
    else:
        pattern = os.path.join(OUT, 'transunet', '*', f'{attr}.txt')
    logs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return logs[-1] if logs else None


def parse(path, attr):
    text = open(path, encoding='utf-8', errors='replace').read()
    rows = []
    for region in ('Cup', 'Rim'):
        for metric, label in (('Dice', 'Overall Dice'), ('IoU', 'Overall IoU'),
                              ('ES-Dice', 'Es-Dice'), ('ES-IoU', 'Es-IoU')):
            m = re.findall(rf'{region} {label}: ([\d.]+)', text)
            if m:
                rows.append((region, metric, '', float(m[-1])))
        for i, group in enumerate(GROUPS[attr]):
            for metric, label in (('Dice', 'dice'), ('IoU', 'Jaccard/IoU')):
                m = re.findall(rf'{i}-attr {re.escape(label)} for {region}: ([\d.]+)', text)
                if m:
                    rows.append((region, metric, group, float(m[-1])))
    return rows


def main():
    paper = {}
    with open(os.path.join(REPO, 'scripts', 'paper_baselines.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            paper[(r['attribute'], r['model'], r['region'], r['metric'], r['group'])] = float(r['value'])

    ours, missing = {}, []
    for attr in ATTRS:
        for model in ('SAMed', 'TransUNet'):
            log = find_log(model, attr)
            if not log:
                missing.append(f'{model}/{attr}')
                continue
            for region, metric, group, value in parse(log, attr):
                ours[(attr, model, region, metric, group)] = value

    with open(os.path.join(OUT, 'results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['attribute', 'model', 'region', 'metric', 'group', 'ours', 'paper', 'diff'])
        for k in sorted(ours):
            p = paper.get(k)
            w.writerow([*k, f'{ours[k]:.4f}', '' if p is None else f'{p:.4f}', '' if p is None else f'{ours[k] - p:+.4f}'])

    lines = ['# Baseline reproduction vs FairSeg paper (Tables 2-5)', '']
    if missing:
        lines += [f'Missing logs: {", ".join(missing)}', '']
    lines += ['Ours / paper / difference. Groups are named from the HF dataset encodings.', '']
    for attr in ATTRS:
        lines += [f'## {attr}', '', '| Model | Region | Metric | Group | Ours | Paper | Diff |', '|---|---|---|---|---|---|---|']
        for model in ('SAMed', 'TransUNet'):
            for region in ('Cup', 'Rim'):
                for metric in ('ES-Dice', 'Dice', 'ES-IoU', 'IoU'):
                    for group in [''] + (GROUPS[attr] if metric in ('Dice', 'IoU') else []):
                        k = (attr, model, region, metric, group)
                        if k not in ours and k not in paper:
                            continue
                        o, p = ours.get(k), paper.get(k)
                        fmt = lambda v: '' if v is None else f'{v:.4f}'
                        d = '' if o is None or p is None else f'{o - p:+.4f}'
                        lines.append(f'| {model} | {region} | {metric} | {group or "overall"} | {fmt(o)} | {fmt(p)} | {d} |')
        lines.append('')
    with open(os.path.join(OUT, 'comparison.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'parsed {len(ours)} values; missing: {missing or "none"}')
    print('wrote', os.path.join(OUT, 'results.csv'), 'and comparison.md')


if __name__ == '__main__':
    main()
