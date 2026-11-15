import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from taspnet import TaSPNet
from data import Roads
from train import load_checkpoint, epoch


def main():
    p = argparse.ArgumentParser(description='Evaluate binary road masks; reports macro and micro percentages')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    saved = load_checkpoint(args.checkpoint, args.device)
    cfg = saved['config']
    torch.set_num_threads(cfg.get('cpu_threads', 2))
    model = TaSPNet(**cfg['model']).to(args.device)
    model.load_state_dict(saved['model'])
    dataset = Roads(args.manifest, cfg)
    result = epoch(model, DataLoader(dataset, batch_size=cfg['batch_size']), cfg, args.device)
    result['prior_kind'] = cfg['prior_kind']
    result['resolution'] = cfg['image_size']
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
