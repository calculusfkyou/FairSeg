import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import numpy as np
import torch

from datasets.dataset_fairseg import RandomGenerator, TestGenerator, read_attrs


def test_read_attrs_grouping():
    d = {'race': np.array(1), 'gender': np.array(0), 'ethnicity': np.array(-1), 'language': np.array(2),
         'maritalstatus': np.array(3), 'age': np.array(59.99)}
    assert read_attrs(d) == {'race': 1, 'gender': 0, 'ethnicity': -1, 'language': 2, 'maritalstatus': 1, 'age': 0}
    d.update(maritalstatus=np.array(0), age=np.array(60.0))
    assert read_attrs(d)['maritalstatus'] == 0 and read_attrs(d)['age'] == 1
    d.update(maritalstatus=np.array(-1))
    assert read_attrs(d)['maritalstatus'] == -1


def test_test_generator():
    image = np.full((664, 798), 255, np.uint8)
    mask = np.zeros((664, 798)); mask[300:364, 360:440] = -1; mask[320:344, 390:410] = -2
    out = TestGenerator()({'image': image, 'label': mask})
    assert out['image'].shape == (3, 224, 224) and out['image'].dtype == torch.float32
    assert abs(out['image'].max().item() - 1.0) < 1e-6
    assert out['label'].shape == (224, 224) and out['label'].dtype == torch.int64
    assert set(out['label'].unique().tolist()) == {0, 1, 2}
    assert out['label'][112, 112] == 2   # 中央裁切 512 後，影像中心落在 cup 內


def test_random_generator_keeps_label_values():
    random.seed(0); np.random.seed(0)
    image = np.random.randint(0, 256, (664, 798)).astype(np.uint8)
    mask = np.zeros((664, 798)); mask[300:364, 360:440] = -1; mask[320:344, 390:410] = -2
    for _ in range(10):
        out = RandomGenerator()({'image': image.copy(), 'label': mask.copy()})
        assert out['image'].shape == (3, 224, 224)
        assert set(out['label'].unique().tolist()) <= {0, 1, 2}


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
