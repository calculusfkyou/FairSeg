import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.dataset_fairseg import ATTRS
from networks.deeplabv3plus import DeepLabV3Plus
from trainer import trainer_fairseg, update_best
from utils import predict_records, save_checkpoint, state_dict_of


def _summ(mean_dice, es):
    return {a: {'mean_dice': mean_dice, 'mean_es_dice': es} for a in ATTRS}


def test_update_best_keeps_earlier_on_tie():
    best = {}
    assert set(update_best(best, 1, _summ(0.80, 0.70))) == {'dice'} | {f'es_{a}' for a in ATTRS}
    assert update_best(best, 2, _summ(0.80, 0.70)) == []
    assert update_best(best, 3, _summ(0.81, 0.70)) == ['dice']


def test_state_dict_has_no_module_prefix():
    net = DeepLabV3Plus(pretrained=False)
    sd = state_dict_of(nn.DataParallel(net))
    assert not any(k.startswith('module.') for k in sd)
    DeepLabV3Plus(pretrained=False).load_state_dict(sd, strict=True)


class _Echo(nn.Module):
    """Predicts the label stored in image channel 0; records the mask it was given."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, x, mask_channels=None):
        self.seen.append(mask_channels)
        return F.one_hot(x[:, 0].long(), 3).permute(0, 3, 1, 2).float()


def test_predict_records():
    label = torch.zeros(2, 4, 4, dtype=torch.long); label[:, 0] = 1; label[:, 1, :2] = 2
    image = label.float().unsqueeze(1).repeat(1, 3, 1, 1)
    attrs = torch.tensor([[0, 1, -1, 0, 1, 0], [2, 0, 1, 2, 0, 1]])
    loader = [{'image': image, 'label': label, 'attrs': attrs, 'name': ['a.npz', 'b.npz']}]
    net = _Echo()
    recs = predict_records(net, loader, mask_channels=[3])
    assert [r['name'] for r in recs] == ['a.npz', 'b.npz'] and net.seen == [[3]]
    assert recs[1]['attrs'] == dict(zip(ATTRS, [2, 0, 1, 2, 0, 1]))
    assert all(r['cup_dice'] == 1 and r['rim_dice'] == 1 for r in recs)
    assert not net.training


def test_checkpoint_roundtrip_with_summaries():
    obj = {'state_dict': {'w': torch.ones(2)}, 'epoch': 3,
           'val': {'race': {'mean_dice': 0.5, 'groups': {'Asian': {'n': 0, 'cup_dice': None}}}}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'best_dice.pth')
        save_checkpoint(obj, path)
        back = torch.load(path, map_location='cpu', weights_only=True)
    assert back['epoch'] == 3 and back['val'] == obj['val'] and torch.equal(back['state_dict']['w'], obj['state_dict']['w'])


def test_trainer_refuses_existing_output():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'args.json'), 'w').close()          # an earlier run's folder
        try:
            trainer_fairseg(types.SimpleNamespace(output_dir=d), None)
        except FileExistsError:
            return
        raise AssertionError('expected FileExistsError')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
