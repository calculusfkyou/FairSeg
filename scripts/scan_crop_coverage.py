"""Check, for every FairSeg npz, how much of the optic disc survives the 512x512
center crop used by both SAMed and TransUNet (crop_center in dataset_fairseg.py)."""
import argparse, csv, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='folder containing the npz files (searched recursively)')
parser.add_argument('--crop', type=int, default=512)
parser.add_argument('--out', default='outputs/data_check/crop_coverage.csv')
args = parser.parse_args()

files = sorted((f, os.path.join(d, f)) for d, _, fs in os.walk(args.root_path) for f in fs if f.endswith('.npz'))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['file', 'height', 'width', 'disc_px', 'cup_px', 'disc_px_in_crop', 'cup_px_in_crop',
                'disc_ymin', 'disc_ymax', 'disc_xmin', 'disc_xmax', 'mask_values'])
    for i, (name, path) in enumerate(files):
        m = np.load(path)['disc_cup_mask']
        h, wd = m.shape
        sy, sx = h // 2 - args.crop // 2, wd // 2 - args.crop // 2
        crop = m[sy:sy + args.crop, sx:sx + args.crop]
        disc = m < 0
        ys, xs = np.where(disc)
        bbox = [ys.min(), ys.max(), xs.min(), xs.max()] if len(ys) else [-1] * 4
        w.writerow([name, h, wd, int(disc.sum()), int((m == -2).sum()), int((crop < 0).sum()),
                    int((crop == -2).sum()), *bbox, ' '.join(str(v) for v in np.unique(m))])
        if (i + 1) % 1000 == 0:
            print(f'{i + 1}/{len(files)}', flush=True)
print('wrote', args.out)
