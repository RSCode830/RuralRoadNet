"""Generate label-free frozen SAM priors and a portable manifest with provenance."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from data import read_rgb
from priors import FrozenSAM, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='JSON list of image/mask paths; mask is never opened here')
    p.add_argument('--output', required=True, help='New manifest JSON path')
    p.add_argument('--config', default='config.json')
    p.add_argument('--sam-checkpoint', required=True)
    p.add_argument('--sam-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    source, output = Path(args.manifest).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output manifest; existing priors are not overwritten')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    rows = json.loads(source.read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent/(output.stem+'_priors')
    cache.mkdir(exist_ok=True)
    sam = FrozenSAM(args.sam_checkpoint, args.sam_type, args.device, cfg['sam']['points_per_side'])
    checkpoint_hash = sha256(args.sam_checkpoint)
    converted, provenance = [], []
    for i, row in enumerate(rows):
        path = (source.parent/row['image']).resolve()
        rgb, _ = read_rgb(path, cfg['image_size'])
        prior, count = sam(rgb, **cfg['sam']['scoring'])
        target = cache/('{:07d}.npy'.format(i))
        if target.exists():
            raise FileExistsError('Prior cache collision: '+str(target))
        np.save(str(target), prior)
        record = dict(row, image=os.path.relpath(str(path), str(output.parent)),
                      prior=os.path.relpath(str(target), str(output.parent)), prior_kind='sam')
        if 'mask' in row:
            record['mask'] = os.path.relpath(str((source.parent/row['mask']).resolve()), str(output.parent))
        converted.append(record)
        provenance.append(dict(image_sha256=sha256(path), proposals=count, prior_sha256=sha256(target)))
        print('{}/{}: {} proposals'.format(i+1, len(rows), count), flush=True)
    output.write_text(json.dumps(converted, indent=2), encoding='utf-8')
    output.with_suffix('.provenance.json').write_text(json.dumps(dict(sam_checkpoint_sha256=checkpoint_hash,
        sam_type=args.sam_type, size=cfg['image_size'], options=cfg['sam'], images=provenance), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
