import os
import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

ATTRS = ('race', 'gender', 'ethnicity', 'language', 'maritalstatus', 'age')

# Index = grouped code from read_attrs; Unknown is -1 and belongs to no group.
GROUP_NAMES = {
    'race': ('Asian', 'Black', 'White'),
    'gender': ('Female', 'Male'),
    'ethnicity': ('Non-Hispanic', 'Hispanic'),
    'language': ('English', 'Spanish', 'Others'),
    'maritalstatus': ('Married', 'Not married'),
    'age': ('<60', '>=60'),
}


def read_attrs(data):
    """Grouped codes of the six attributes, as in the official TransUNet read_attr."""
    attrs = {a: int(data[a].item()) for a in ('race', 'gender', 'ethnicity', 'language')}
    marital = int(data['maritalstatus'].item())
    attrs['maritalstatus'] = marital if marital in (0, -1) else 1
    attrs['age'] = 0 if data['age'].item() < 60 else 1
    return attrs


def index_npz_files(base_dir):
    """Map every npz filename under base_dir to its path; the list files decide the split."""
    index = {}
    for dirpath, _, filenames in os.walk(base_dir):
        for f in filenames:
            if f.endswith('.npz'):
                if f in index:
                    raise ValueError(f'{f} found twice under {base_dir}')
                index[f] = os.path.join(dirpath, f)
    return index


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def crop_center(img, size):
    y, x = img.shape
    startx = x // 2 - size // 2
    starty = y // 2 - size // 2
    return img[starty:starty + size, startx:startx + size]


def _prepare(image, label, crop):
    image = np.clip(image, 0, 255) / 255.0
    label = label.copy()
    label[label == -1] = 1   # rim
    label[label == -2] = 2   # cup
    return crop_center(image, crop), crop_center(label, crop)


def _to_tensors(image, label, output_size):
    x, y = image.shape
    if (x, y) != (output_size, output_size):
        image = zoom(image, (output_size / x, output_size / y), order=3)
        label = zoom(label, (output_size / x, output_size / y), order=0)
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
    label = torch.from_numpy(label.astype(np.int64))
    return image, label


class RandomGenerator:
    """Training transform of the official TransUNet (use_normalize=True)."""

    def __init__(self, output_size=224, crop=512):
        self.output_size, self.crop = output_size, crop

    def __call__(self, sample):
        image, label = _prepare(sample['image'], sample['label'], self.crop)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        image, label = _to_tensors(image, label, self.output_size)
        return {**sample, 'image': image, 'label': label}


class TestGenerator:
    """Validation / test transform of the official TransUNet, without augmentation."""

    def __init__(self, output_size=224, crop=512):
        self.output_size, self.crop = output_size, crop

    def __call__(self, sample):
        image, label = _prepare(sample['image'], sample['label'], self.crop)
        image, label = _to_tensors(image, label, self.output_size)
        return {**sample, 'image': image, 'label': label}


class FairSegDataset(Dataset):
    def __init__(self, base_dir, list_file, transform=None):
        with open(list_file) as f:
            self.names = [line.strip() for line in f if line.strip()]
        self.paths = index_npz_files(base_dir)
        missing = [n for n in self.names if n not in self.paths]
        if missing:
            raise FileNotFoundError(f'{len(missing)} files in {list_file} not found under {base_dir}, e.g. {missing[:3]}')
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data = np.load(self.paths[name])
        attrs = read_attrs(data)
        sample = {'image': data['slo_fundus'], 'label': data['disc_cup_mask'],
                  'attrs': torch.tensor([attrs[a] for a in ATTRS], dtype=torch.long), 'name': name}
        return self.transform(sample) if self.transform else sample
