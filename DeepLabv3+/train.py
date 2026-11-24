import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required=True, help='HF Dataset folder (contains Training/ and Test/)')
parser.add_argument('--list_dir', default='lists/FairSeg_split', help='folder with train.txt and val.txt')
parser.add_argument('--output_dir', default='../outputs/deeplabv3plus/train_seed42')
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=24,
                    help='total batch over all GPUs (unlike TransUNet, not per GPU); lr is not rescaled')
parser.add_argument('--n_gpu', type=int, default=1,
                    help='more than 1 uses nn.DataParallel, which leaks ~0.43 GB of host memory per epoch on this '
                         'Windows setup (spec section 8)')
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=8)

# Windows DataLoader workers re-import this file (spawn), so everything that builds
# models or starts workers must stay behind this guard.
if __name__ == '__main__':
    args = parser.parse_args()
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from networks.deeplabv3plus import DeepLabV3Plus
    from trainer import trainer_fairseg
    trainer_fairseg(args, DeepLabV3Plus(pretrained=True))
