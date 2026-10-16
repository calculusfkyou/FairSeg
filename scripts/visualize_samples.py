"""Overlay disc/cup masks on FairSeg images at each preprocessing stage, using the
actual SAMed and TransUNet transforms, so crop / resize / labels can be checked by eye.

Colors: green = rim (label 1), red = cup (label 2), yellow box = 512x512 center crop.
"""
import argparse, copy, importlib.util, os, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


samed = load_module('samed_ds', os.path.join(REPO, 'SAMed', 'datasets', 'dataset_fairseg.py'))
transunet = load_module('transunet_ds', os.path.join(REPO, 'TransUNet', 'datasets', 'dataset_fairseg.py'))


def overlay(ax, image, label, title):
    """image: HxW in any range; label: HxW with 0 bg, 1 rim, 2 cup."""
    image = np.asarray(image, dtype=np.float32)
    image = (image - image.min()) / max(image.max() - image.min(), 1e-6)
    rgb = np.stack([image] * 3, axis=-1)
    label = np.asarray(label)
    for value, color in ((1, (0, 1, 0)), (2, (1, 0, 0))):
        m = label == value
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(color)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def raw_to_label(mask):
    label = np.zeros(mask.shape, dtype=np.int64)
    label[mask == -1] = 1
    label[mask == -2] = 2
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--list_dir', default=os.path.join(REPO, 'SAMed', 'lists', 'FairSeg_final'))
    parser.add_argument('--files', nargs='*', help='npz names to show; default: random picks from train and test lists')
    parser.add_argument('--n', type=int, default=3, help='random picks per split when --files is not given')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=os.path.join(REPO, 'outputs', 'data_check', 'overlay.png'))
    args = parser.parse_args()

    ds_args = type('A', (), {'list_dir': args.list_dir})()
    datasets = {s: samed.FairSeg_dataset(args.root_path, s, ds_args) for s in ('train', 'test')}
    rng = random.Random(args.seed)
    if args.files:
        picks = [(('test' if f in set(datasets['test'].data_files) else 'train'), f) for f in args.files]
    else:
        picks = [(s, f) for s in ('train', 'test') for f in rng.sample(datasets[s].data_files, args.n)]

    stages = [
        ('SAMed input 512', samed.TestGenerator([512, 512], [128, 128], 512, use_normalize=True), 'label'),
        ('SAMed loss target 128', samed.TestGenerator([512, 512], [128, 128], 512, use_normalize=True), 'low_res_label'),
        ('TransUNet input 224', transunet.TestGenerator([224, 224], [56, 56], 512, use_normalize=True), 'label'),
        ('TransUNet train aug', transunet.RandomGenerator([224, 224], 512, use_normalize=True), 'label'),
    ]
    fig, axes = plt.subplots(len(picks), 2 + len(stages), figsize=(3 * (2 + len(stages)), 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, (split, name) in enumerate(picks):
        ds = datasets[split]
        data = np.load(ds.file_index[name])
        image, mask = data['slo_fundus'], data['disc_cup_mask']
        h, w = image.shape
        sy, sx = h // 2 - 256, w // 2 - 256

        overlay(axes[row, 0], image, raw_to_label(mask), f'{split} {name}\nraw {h}x{w}')
        axes[row, 0].add_patch(patches.Rectangle((sx, sy), 512, 512, fill=False, edgecolor='yellow', lw=1))
        overlay(axes[row, 1], image[sy:sy + 512, sx:sx + 512], raw_to_label(mask)[sy:sy + 512, sx:sx + 512], 'center crop 512')

        raw = ds[ds.data_files.index(name)]
        for col, (title, transform, key) in enumerate(stages, start=2):
            random.seed(args.seed + row)
            np.random.seed(args.seed + row)
            s = transform(copy.deepcopy(raw))
            lab = s[key].numpy()
            img = s['image'][0].numpy()
            if lab.shape != img.shape:  # low-res target: show next to a downsized image
                img = img[::img.shape[0] // lab.shape[0], ::img.shape[1] // lab.shape[1]]
            overlay(axes[row, col], img, lab, f'{title}\nvalues {sorted(np.unique(lab).tolist())}')

    fig.suptitle('green = rim (1), red = cup (2), yellow = 512 center crop', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
