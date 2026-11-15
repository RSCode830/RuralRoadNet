"""Train/validate TaSPNet. Frozen SAM priors are cached before training."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from taspnet import TaSPNet
from data import Roads, check_splits
from losses import objective
from metrics import Metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, device='cpu'):
    # Optimizer/RNG state contains Python objects. Only load your own trusted files.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def epoch(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    metrics, total, count = Metrics(cfg['threshold']), 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for image, target, prior in loader:
            image, target, prior = image.to(device), target.to(device), prior.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            result = model(image, prior)
            loss = objective(result, target, **cfg['loss'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
            metrics.update(result['logits'], target)
            total += float(loss.detach())*len(image)
            count += len(image)
    result = metrics.compute()
    result['loss'] = total/count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='End after this many completed epochs; preserves planned scheduler for resume')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed_all(cfg['seed'])
    if cfg.get('cpu_threads'):
        torch.set_num_threads(cfg['cpu_threads'])
    train, val = Roads(args.train, cfg, True), Roads(args.val, cfg)
    check_splits(train, val)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out/'last.pt').exists():
        raise FileExistsError('Existing run: choose new --output or --resume')
    loaders = [DataLoader(d, batch_size=cfg['batch_size'], shuffle=i == 0, num_workers=0)
               for i, d in enumerate((train, val))]
    model = TaSPNet(**cfg['model']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    start, best = 0, -1.
    if args.resume:
        saved = load_checkpoint(args.resume, args.device)
        if saved['config'] != cfg or saved['train_rows'] != train.rows or saved['val_rows'] != val.rows:
            raise ValueError('Resume config and manifests must match checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best = saved['epoch']+1, saved['best']
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'].cpu())
        if torch.cuda.is_available() and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('Parameters (excludes frozen SAM):', sum(p.numel() for p in model.parameters()), flush=True)
    if cfg['prior_kind'] == 'synthetic_demo':
        print('SYNTHETIC SOFTWARE TEST: not SAM priors or research evidence.', flush=True)
    end = min(cfg['epochs'], args.stop_after) if args.stop_after is not None else cfg['epochs']
    for index in range(start, end):
        training = epoch(model, loaders[0], cfg, args.device, optimizer)
        validation = epoch(model, loaders[1], cfg, args.device)
        score = validation['macro_percent']['road_iou']
        improved = score > best
        best = max(best, score)
        scheduler.step()
        record = {'epoch':index+1, 'train':training, 'val':validation, 'next_lr':optimizer.param_groups[0]['lr']}
        with (out/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print('epoch={} train_loss={:.5f} val_IoU={:.3f}%'.format(index+1, training['loss'], score), flush=True)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     epoch=index, best=best, config=cfg, train_rows=train.rows, val_rows=val.rows,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state, out/'last.pt')
        if improved:
            torch.save(state, out/'best.pt')


if __name__ == '__main__':
    main()
