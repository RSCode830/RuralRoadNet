import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from taspnet import TaSPNet
from data import read_rgb, read_prior
from train import load_checkpoint


def main():
    p = argparse.ArgumentParser(description='Predict roads with a cached prior or frozen SAM checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prior', help='Aligned resized H,W .npy prior')
    group.add_argument('--sam-checkpoint')
    p.add_argument('--sam-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    saved = load_checkpoint(args.checkpoint, args.device)
    cfg = saved['config']
    torch.set_num_threads(cfg.get('cpu_threads', 2))
    rgb, original = read_rgb(args.image, cfg['image_size'])
    if args.prior:
        prior = read_prior(args.prior, cfg['image_size'])
    else:
        from priors import FrozenSAM
        generator = FrozenSAM(args.sam_checkpoint, args.sam_type, args.device, cfg['sam']['points_per_side'])
        prior, _ = generator(rgb, **cfg['sam']['scoring'])
        prior = torch.from_numpy(prior)[None]
    model = TaSPNet(**cfg['model']).to(args.device).eval()
    model.load_state_dict(saved['model'])
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None]/255
    with torch.no_grad():
        logits = model(image.to(args.device), prior[None].to(args.device))['logits']
        probabilities = F.interpolate(logits, (original[1], original[0]), mode='bilinear', align_corners=False).sigmoid()[0, 0].cpu().numpy()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    np.save(str(out/(stem+'_probability.npy')), probabilities)
    Image.fromarray((probabilities >= cfg['threshold']).astype(np.uint8)*255).save(out/(stem+'_mask.png'))
    print('Saved original-size probability and binary mask to', out)


if __name__ == '__main__':
    main()
