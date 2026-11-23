import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bda import choose_k, extract_val, iscore_inputs, records_from_cache
from networks.deeplabv3plus import DeepLabV3Plus
from utils import predict_records


def test_iscore_inputs_excludes_unknown():
    feats = np.array([[1., 5.], [2., 6.], [100., 100.], [3., 7.], [4., 8.]])
    X, y, med, keep = iscore_inputs(feats, np.array([0, 1, -1, 0, 1]))
    assert keep.tolist() == [0, 1, 3, 4] and y.tolist() == [0, 1, 0, 1]
    assert np.allclose(med, [2.5, 6.5])                    # the Unknown row is not in the median
    assert X.shape == (4, 2)


def test_iscore_inputs_rejects_multiclass():
    try:
        iscore_inputs(np.zeros((3, 2)), np.array([0, 1, 2]))
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_choose_k_ties_prefer_fewer_channels():
    rows = [{'K': 0, 'n_channels': 0, 'mean_dice': 0.80, 'mean_es_dice': 0.70},
            {'K': 1, 'n_channels': 3, 'mean_dice': 0.80, 'mean_es_dice': 0.72},
            {'K': 4, 'n_channels': 9, 'mean_dice': 0.79, 'mean_es_dice': 0.72}]
    assert choose_k(rows, 'mean_dice')['K'] == 0
    assert choose_k(rows, 'mean_es_dice')['K'] == 1


def test_cache_path_matches_full_model():
    torch.manual_seed(0)
    net = DeepLabV3Plus(pretrained=False).eval()
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        labels = net(images).argmax(1)                     # GT = unmasked prediction, so a mask changes the metrics
    samples = [{'image': images[i], 'label': labels[i], 'attrs': torch.tensor([0, 1, i, 0, -1, 1]), 'name': f'n{i}'}
               for i in range(2)]
    loader = DataLoader(samples, batch_size=1)
    cache = extract_val(net, loader)
    assert cache['z'].shape == (2, 2048, 14, 14) and cache['low'].shape == (2, 256, 56, 56)
    assert cache['z'].device.type == 'cpu' and cache['labels'].dtype == np.uint8 and cache['labels'].shape == (2, 224, 224)
    assert cache['names'] == ['n0', 'n1'] and cache['attrs'][1]['ethnicity'] == 1
    half = list(range(0, 2048, 2))
    for channels in ([], half):
        assert records_from_cache(net, cache, channels, batch_size=1) == \
            predict_records(net, loader, mask_channels=channels or None)
    assert records_from_cache(net, cache, half) != records_from_cache(net, cache, [])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
