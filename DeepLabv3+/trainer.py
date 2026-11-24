import csv
import functools
import json
import logging
import os
import sys

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataset_fairseg import ATTRS, FairSegDataset, RandomGenerator, TestGenerator
from utils import CEDiceLoss, ensure_new, predict_records, save_checkpoint, seed_worker, state_dict_of, summarize


def update_best(best, epoch, summaries):
    """Update the 7 selection scores (mean Dice, then mean ES-Dice per attribute); ties keep the earlier epoch."""
    scores = {'dice': next(iter(summaries.values()))['mean_dice'],   # same for every attribute
              **{f'es_{a}': s['mean_es_dice'] for a, s in summaries.items()}}
    updated = [k for k, v in scores.items() if v > best.get(k, float('-inf'))]
    for k in updated:
        best[k] = scores[k]
        logging.info('epoch %d: new best %s = %.4f', epoch, k, scores[k])
    return updated


def trainer_fairseg(args, model):
    ensure_new(os.path.join(args.output_dir, 'args.json'))   # an earlier run's checkpoints would be overwritten
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.output_dir, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    db_train = FairSegDataset(args.root_path, os.path.join(args.list_dir, 'train.txt'), RandomGenerator())
    db_val = FairSegDataset(args.root_path, os.path.join(args.list_dir, 'val.txt'), TestGenerator())
    # persistent_workers: Windows spawns workers, and restarting them every epoch costs ~50 s.
    persistent = args.num_workers > 0
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, persistent_workers=persistent,
                             worker_init_fn=functools.partial(seed_worker, args.seed))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
                           pin_memory=True, persistent_workers=persistent)

    model = model.cuda()
    if args.n_gpu > 1:   # leaks ~0.43 GB of host memory per epoch on this Windows setup; formal runs use 1 GPU
        model = nn.DataParallel(model, device_ids=list(range(args.n_gpu)))
    criterion = CEDiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    max_iterations = args.max_epochs * len(trainloader)
    logging.info('%d train / %d val images, %d iterations per epoch, %d max iterations',
                 len(db_train), len(db_val), len(trainloader), max_iterations)

    csv_path = os.path.join(args.output_dir, 'epochs.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'lr', 'loss', 'ce', 'dice_loss', 'val_mean_dice', 'val_cup_dice',
                                'val_rim_dice'] + [f'val_mean_es_dice_{a}' for a in ATTRS])
    best = {}
    iter_num = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        sums = [0.0, 0.0, 0.0]
        for batch in trainloader:
            lr = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for group in optimizer.param_groups:
                group['lr'] = lr
            image, label = batch['image'].cuda(non_blocking=True), batch['label'].cuda(non_blocking=True)
            loss, ce, dice = criterion(model(image), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            values = (loss.item(), ce.item(), dice.item())
            sums = [s + v for s, v in zip(sums, values)]
            if iter_num % 20 == 0:   # scripts/watch_training.ps1 parses this line
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, *values))

        records = predict_records(model, valloader)
        summaries = {a: summarize(records, a) for a in ATTRS}
        overall = summaries[ATTRS[0]]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, lr] + [s / len(trainloader) for s in sums]
                                   + [overall['mean_dice'], overall['cup_dice'], overall['rim_dice']]
                                   + [summaries[a]['mean_es_dice'] for a in ATTRS])
        logging.info('epoch %d: loss %.4f, val mean Dice %.4f, mean ES-Dice %s', epoch, sums[0] / len(trainloader),
                     overall['mean_dice'], ', '.join(f"{a} {summaries[a]['mean_es_dice']:.4f}" for a in ATTRS))
        for name in update_best(best, epoch, summaries):
            save_checkpoint({'state_dict': state_dict_of(model), 'epoch': epoch, 'val': summaries},
                            os.path.join(args.output_dir, f'best_{name}.pth'))
    logging.info('best: %s', json.dumps(best))
