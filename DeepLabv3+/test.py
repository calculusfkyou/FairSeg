"""Evaluate one checkpoint on a split (spec §6, §9, §10.7), optionally with the X* channels of Z set to 0.

  python test.py --root_path <DATA> --split test --checkpoint <ckpt> [--selection <selection.json> --key es] --output <json>
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from datasets.dataset_fairseg import ATTRS, FairSegDataset, TestGenerator
from networks.deeplabv3plus import DeepLabV3Plus
from utils import predict_records, summarize

# Same inference settings as bda.py, so val (BDA) and test numbers come from identical arithmetic.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='HF Dataset folder (contains Training/ and Test/)')
parser.add_argument('--list_dir', default='lists/FairSeg_split')
parser.add_argument('--split', required=True, choices=('val', 'test'), help='reads <list_dir>/<split>.txt')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--selection', help='selection.json from bda.py; its [key][channels] are masked')
parser.add_argument('--key', choices=('dice', 'es'), help='which K of --selection to use')
parser.add_argument('--attributes', nargs='+', choices=ATTRS, default=list(ATTRS))
parser.add_argument('--output', required=True, help='result JSON; never overwritten')
parser.add_argument('--num_workers', type=int, default=4)


def ensure_new(path):
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists; results are never overwritten, choose another --output')


def print_summary(attr, s):
    groups = s['groups']
    print(f'\n{attr}: ' + ', '.join(f'{name} n={g["n"]}' for name, g in groups.items()))
    print(f'{"":5}{"Dice":>8}{"IoU":>8}{"ES-Dice":>9}{"ES-IoU":>8}' + ''.join(f'{name + " Dice":>20}' for name in groups))
    for part in ('cup', 'rim'):
        cells = ''.join(f'{g[f"{part}_dice"]:>20.4f}' if g['n'] else f'{"-":>20}' for g in groups.values())
        print(f'{part:5}{s[f"{part}_dice"]:>8.4f}{s[f"{part}_iou"]:>8.4f}{s[f"{part}_es_dice"]:>9.4f}'
              f'{s[f"{part}_es_iou"]:>8.4f}{cells}')


def main(args):
    ensure_new(args.output)
    if (args.selection is None) != (args.key is None):
        parser.error('--selection and --key must be given together')
    mask = []
    if args.selection:
        with open(args.selection) as f:
            chosen = json.load(f)[args.key]
        mask = chosen['channels']
        print(f'mask: K_{args.key} = {chosen["K"]} from {args.selection}, {len(mask)} channels', flush=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = DeepLabV3Plus(pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    db = FairSegDataset(args.root_path, os.path.join(args.list_dir, f'{args.split}.txt'), TestGenerator())
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    records = predict_records(model, loader, mask_channels=mask)

    result = {'args': vars(args), 'checkpoint': {'path': args.checkpoint, 'epoch': ckpt['epoch']},
              'split': args.split, 'mask_channels': mask,
              'attributes': {a: summarize(records, a) for a in args.attributes}}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'x') as f:   # 'x': fail rather than overwrite if it appeared meanwhile
        json.dump(result, f, indent=2)
    print(f'{args.split}: {len(records)} images, checkpoint epoch {ckpt["epoch"]} -> {args.output}')
    for a in args.attributes:
        print_summary(a, result['attributes'][a])


# Windows DataLoader workers re-import this file (spawn), so everything that builds
# models or starts workers must stay behind this guard.
if __name__ == '__main__':
    main(parser.parse_args())
