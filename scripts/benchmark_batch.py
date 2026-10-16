"""Measure peak GPU memory and training-step time per batch size for the SAMed / TransUNet
baselines, and how fast the official DataLoader can feed images.

Run from the repo root, e.g.
  python scripts/benchmark_batch.py --model samed --batch_sizes 4 8 12 16
  python scripts/benchmark_batch.py --model transunet --batch_sizes 8 16 24 32 --loader --root_path ...
"""
import argparse, os, sys, time, types
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['samed', 'transunet'], required=True)
parser.add_argument('--batch_sizes', type=int, nargs='+', default=[4, 8])
parser.add_argument('--steps', type=int, default=10, help='timed steps per batch size (after 3 warm-up steps)')
parser.add_argument('--n_gpu', type=int, default=1, help='>1 wraps the model in nn.DataParallel, as the trainers do')
parser.add_argument('--loader', action='store_true', help='also time the real DataLoader (needs --root_path)')
parser.add_argument('--root_path')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

model_dir = os.path.join(REPO, 'SAMed' if args.model == 'samed' else 'TransUNet')
os.chdir(model_dir)
sys.path.insert(0, model_dir)
from utils import DiceLoss


def build():
    if args.model == 'samed':
        from segment_anything import sam_model_registry
        from sam_lora_image_encoder import LoRA_Sam
        sam, emb = sam_model_registry['vit_b'](image_size=512, num_classes=2, checkpoint='checkpoints/sam_vit_b_01ec64.pth',
                                               pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
        net = LoRA_Sam(sam, 4).cuda()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.1)
        return net, opt, 512, emb * 4
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer
    cfg = CONFIGS['ViT-B_16']; cfg.n_classes = 3; cfg.n_skip = 0
    net = VisionTransformer(cfg, img_size=224, num_classes=3).cuda()
    net.load_from(weights=np.load(cfg.pretrained_path))
    opt = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    return net, opt, 224, 224


def step(net, opt, x, y, ce, dice):
    if args.model == 'samed':
        logits = net(x, True, 512)['low_res_logits']
        loss = 0.2 * ce(logits, y) + 0.8 * dice(logits, y, softmax=True)
    else:
        logits = net(x)
        loss = 0.5 * ce(logits, y) + 0.5 * dice(logits, y, softmax=True)
    opt.zero_grad()
    loss.backward()
    opt.step()


def main():
    net, opt, img, lab = build()
    if args.n_gpu > 1:
        net = nn.DataParallel(net)
    net.train()
    ce, dice = CrossEntropyLoss(), DiceLoss(3)
    print(f'{args.model}  input {img}  GPUs {args.n_gpu} ({torch.cuda.get_device_name(0)})')
    print(f'{"batch":>5} {"peak mem GB":>12} {"s/step":>8} {"img/s":>7}')
    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        for d in range(args.n_gpu):
            torch.cuda.reset_peak_memory_stats(d)
        x = torch.rand(bs, 3, img, img, device='cuda')
        y = torch.randint(0, 3, (bs, lab, lab), device='cuda')
        try:
            for _ in range(3):
                step(net, opt, x, y, ce, dice)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(args.steps):
                step(net, opt, x, y, ce, dice)
            torch.cuda.synchronize()
            dt = (time.time() - t) / args.steps
            mem = max(torch.cuda.max_memory_allocated(d) for d in range(args.n_gpu)) / 1e9
            reserved = max(torch.cuda.max_memory_reserved(d) for d in range(args.n_gpu))
            # On Windows the driver may spill into system RAM instead of raising OOM, which
            # only shows up as a big slowdown; flag batches whose reservation is near the card size.
            spill = ' <- near VRAM limit, may spill to system RAM' if reserved > 0.9 * torch.cuda.get_device_properties(0).total_memory else ''
            print(f'{bs:>5} {mem:>12.2f} {dt:>8.3f} {bs / dt:>7.1f}{spill}', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'{bs:>5} {"OOM":>12}', flush=True)
            opt.zero_grad(set_to_none=True)
            del x, y
            torch.cuda.empty_cache()
            break

    if args.loader:
        from torch.utils.data import DataLoader
        if args.model == 'samed':
            from datasets.dataset_fairseg import FairSeg_dataset, RandomGenerator
            tf = RandomGenerator([512, 512], [128, 128], 512, use_normalize=True)
        else:
            from datasets.dataset_fairseg import FairSeg_dataset, RandomGenerator
            tf = RandomGenerator([224, 224], 512, use_normalize=True)
        ds = FairSeg_dataset(args.root_path, 'train', types.SimpleNamespace(list_dir='lists/FairSeg_final'), transform=tf)
        bs = args.batch_sizes[0]
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        it = iter(loader)
        next(it)  # worker start-up
        n, t = 0, time.time()
        for _ in range(max(1, 200 // bs)):
            n += next(it)['image'].shape[0]
        dt = time.time() - t
        print(f'DataLoader ({args.num_workers} workers, no GPU work): {n / dt:.1f} img/s')


# Windows DataLoader workers re-import this file (spawn), so everything that builds
# models or starts workers must stay behind this guard.
if __name__ == '__main__':
    main()
