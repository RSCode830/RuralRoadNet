"""Synthetic data for software validation ONLY; does not emulate a real SAM model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/demo')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    for split, count in [('train', 4), ('val', 2), ('test', 2)]:
        rows = []
        for i in range(count):
            stem = '{}_{}'.format(split, i)
            road = np.zeros((16, 16), dtype=np.uint8)
            road[:, rng.randint(2, 13):][:, :2] = 255
            road[rng.randint(2, 13), :] = 255
            rgb = rng.randint(0, 90, (16, 16, 3)).astype(np.uint8)
            rgb[road > 0] = 200
            # Independent random prior, not generated from labels or presented as SAM.
            prior = (.5+.23*rng.rand(16, 16)).astype(np.float32)
            Image.fromarray(rgb).save(out/(stem+'.png'))
            Image.fromarray(road).save(out/(stem+'_mask.png'))
            np.save(str(out/(stem+'_prior.npy')), prior)
            rows.append(dict(image=stem+'.png', mask=stem+'_mask.png', prior=stem+'_prior.npy',
                             prior_kind='synthetic_demo', group=split))
        (out/(split+'.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
    cfg = json.loads((Path(__file__).parent/'config.json').read_text(encoding='utf-8'))
    cfg.update(image_size=16, prior_kind='synthetic_demo', batch_size=2, epochs=2)
    cfg['model'].update(widths=[4, 8, 12], depths=[2, 2], decoder_depth=1, state_dim=2, stripe_width=4)
    cfg['loss']['skeleton_iterations'] = 3
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('Synthetic test data:', out)


if __name__ == '__main__':
    main()
