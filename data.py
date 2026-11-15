import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F


def read_rgb(path, size):
    with Image.open(path) as image:
        original = image.size
        array = np.array(image.convert('RGB').resize((size, size), Image.BILINEAR))
    return array, original


def read_prior(path, size):
    prior = np.load(str(path), allow_pickle=False)
    if prior.shape != (size, size) or not np.isfinite(prior).all() or prior.min() < 0 or prior.max() > 1:
        raise ValueError('Prior must be a finite [image_size,image_size] array in [0,1]: '+str(path))
    return torch.from_numpy(prior.astype(np.float32))[None]


class Roads(Dataset):
    def __init__(self, manifest, config, augment=False):
        self.manifest, self.config, self.augment = Path(manifest).resolve(), config, augment
        self.rows = json.loads(self.manifest.read_text(encoding='utf-8'))
        if not self.rows:
            raise ValueError('Empty manifest')
        paths = [str(self.resolve(row['image'])) for row in self.rows]
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate images in manifest')
        for row in self.rows:
            if row.get('prior_kind') != config['prior_kind']:
                raise ValueError('Manifest prior_kind must match config (sam or synthetic_demo)')

    def resolve(self, name):
        return (self.manifest.parent/name).resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, size = self.rows[index], self.config['image_size']
        array, original = read_rgb(self.resolve(row['image']), size)
        image = torch.from_numpy(array.copy()).permute(2, 0, 1).float()/255
        with Image.open(self.resolve(row['mask'])) as mask:
            # Reject colored class masks and ambiguous anti-aliased labels.
            raw = np.asarray(mask)
            if raw.ndim == 3 and np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 1], raw[..., 2]):
                raw = raw[..., 0]
            if raw.ndim != 2 or mask.size != original or not np.isin(raw, [0, 1, 255]).all():
                raise ValueError('Masks must be aligned binary 0/1 or 0/255 grayscale images')
            target = torch.from_numpy((raw > 0).astype(np.float32))[None, None]
        target = F.interpolate(target, (size, size), mode='nearest')[0]
        prior = read_prior(self.resolve(row['prior']), size)
        if self.augment:
            for axis in (1, 2):
                if random.random() < .5:
                    image, target, prior = image.flip(axis), target.flip(axis), prior.flip(axis)
            rotation = random.randrange(4)
            image, target, prior = [v.rot90(rotation, (1, 2)) for v in (image, target, prior)]
            # Photometric transforms apply only to image; SAM cache stays geometric.
            image = (image*random.uniform(.8, 1.2)).clamp(0, 1)
            mean = image.mean((1, 2), keepdim=True)
            image = ((image-mean)*random.uniform(.8, 1.2)+mean).clamp(0, 1)
            gray = image.mean(0, keepdim=True)
            image = ((image-gray)*random.uniform(.8, 1.2)+gray).clamp(0, 1)
        return image, target, prior


def check_splits(train, val):
    if {train.resolve(r['image']) for r in train.rows} & {val.resolve(r['image']) for r in val.rows}:
        raise ValueError('Train/validation image overlap')
    groups_a = {r['group'] for r in train.rows if r.get('group')}
    groups_b = {r['group'] for r in val.rows if r.get('group')}
    if groups_a & groups_b:
        raise ValueError('Train/validation geographic group overlap')
