"""Geometry-routed stripe scanning and an input-selective diagonal SSM.

This is a transparent paper-derived implementation, not the original authors'
kernel. See IMPLEMENTATION.md for hard-routing and COSA alignment choices.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def affine_scan(a, b, sequential=False):
    """h[t] = a[t]*h[t-1] + b[t], zero initial state. Shapes B,L,C,N."""
    if sequential:
        state = torch.zeros_like(b[:, 0])
        states = []
        for t in range(b.shape[1]):
            state = a[:, t]*state+b[:, t]
            states.append(state)
        return torch.stack(states, 1)
    # Associative prefix composition; no divisions by small cumulative products.
    step = 1
    while step < b.shape[1]:
        b = torch.cat([b[:, :step], b[:, step:]+a[:, step:]*b[:, :-step]], 1)
        a = torch.cat([a[:, :step], a[:, step:]*a[:, :-step]], 1)
        step *= 2
    return b


class SelectiveSSM(nn.Module):
    def __init__(self, channels, state_dim=8, backend='parallel'):
        super().__init__()
        if backend not in ('parallel', 'reference', 'cuda'):
            raise ValueError('Unknown SSM backend')
        self.backend, self.state_dim = backend, state_dim
        self.input = nn.Linear(channels, 2*channels)
        self.local = nn.Conv1d(channels, channels, 3, groups=channels, padding=2)
        self.params = nn.Linear(channels, channels+2*state_dim)
        self.log_a = nn.Parameter(torch.arange(1, state_dim+1).float().log().repeat(channels, 1))
        self.skip = nn.Parameter(torch.ones(channels))
        self.output = nn.Linear(channels, channels)

    def forward(self, sequence):
        u, gate = self.input(sequence).chunk(2, -1)
        u = F.silu(self.local(u.transpose(1, 2))[..., :u.shape[1]].transpose(1, 2))
        dt, b, c = torch.split(self.params(u), [u.shape[-1], self.state_dim, self.state_dim], -1)
        dt = F.softplus(dt.float()-2.)
        a = -self.log_a.float().exp()
        if self.backend == 'cuda':
            if not u.is_cuda:
                raise ValueError('CUDA backend requires CUDA tensors and mamba-ssm')
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            y = selective_scan_fn(u.float().transpose(1, 2).contiguous(),
                                  dt.transpose(1, 2).contiguous(), a,
                                  b.float().transpose(1, 2).contiguous(),
                                  c.float().transpose(1, 2).contiguous(), self.skip.float())
            y = y.transpose(1, 2)
        else:
            decay = torch.exp(dt[..., None]*a)
            drive = dt[..., None]*b.float()[:, :, None, :]*u.float()[..., None]
            hidden = affine_scan(decay, drive, self.backend == 'reference')
            y = (hidden*c.float()[:, :, None, :]).sum(-1)+self.skip*u.float()
        return self.output((y*F.silu(gate.float())).to(sequence.dtype))


def stripe_groups(labels, angles, width, shift=0):
    """Partition ALL pixels exactly once into direction-specific S-shaped stripes.

    Geometry sorting is discrete and done on CPU. Coordinates are projected onto
    each canonical direction; alternate rasterized transverse lanes reverse.
    Returned indices address a flattened H*W tensor; no interpolation or pixels lost.
    """
    if width <= 0:
        raise ValueError('stripe_width must be positive')
    h, w = labels.shape
    yy, xx = np.mgrid[:h, :w]
    groups = []
    for direction, angle in enumerate(angles):
        s = xx*np.cos(angle)+yy*np.sin(angle)
        t = -xx*np.sin(angle)+yy*np.cos(angle)
        band = np.floor((t+shift)/width).astype(np.int64)
        lane = np.floor(t+shift+1e-6).astype(np.int64)
        selected = labels == direction
        for band_id in np.unique(band[selected]):
            ids = np.flatnonzero(selected & (band == band_id))
            lanes = lane.ravel()[ids]
            along = s.ravel()[ids]*np.where(lanes % 2 == 0, 1., -1.)
            order = np.lexsort((ids, along, lanes))
            groups.append((direction, ids[order].copy()))
    return groups


class TopologyMamba(nn.Module):
    def __init__(self, channels, state_dim=8, stripe_width=8, shifted=False,
                 directions=4, backend='parallel', route_temperature=.2):
        super().__init__()
        if directions < 2 or route_temperature <= 0:
            raise ValueError('Need at least two directions and positive temperature')
        self.channels, self.directions = channels, directions
        self.stripe_width, self.shift = stripe_width, stripe_width/2 if shifted else 0
        self.temperature = route_temperature
        self.angles = [-math.pi/2+i*math.pi/directions for i in range(directions)]
        self.orientation = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                         nn.SiLU(), nn.Conv2d(channels, 1, 1))
        self.ssm = SelectiveSSM(channels, state_dim, backend)
        # Transpose orientation/channel axes before grouped 1x1 mixing (fixed shuffle).
        self.cosa = nn.Conv1d(channels*directions, channels*directions, 1, groups=channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x, prior_bias=None):
        batch, channels, h, w = x.shape
        theta = self.orientation(x).tanh()[:, 0]*(math.pi/2)
        angles = x.new_tensor(self.angles)
        difference = theta[:, None]-angles[None, :, None, None]
        # Undirected road orientation: circular distance modulo pi.
        distance = torch.abs(torch.atan2(torch.sin(2*difference), torch.cos(2*difference)))/2
        probabilities = (-distance/self.temperature).softmax(1)
        labels = distance.argmin(1).detach().cpu().numpy()
        flat = x.permute(0, 2, 3, 1).reshape(batch, h*w, channels)
        packed, metadata = [], []
        for n in range(batch):
            for direction, indices in stripe_groups(labels[n], self.angles, self.stripe_width, self.shift):
                ids = torch.as_tensor(indices, device=x.device, dtype=torch.long)
                # Straight-through selected-direction gate: exact hard forward routing,
                # a surrogate gradient to the orientation estimator, never through argsort.
                probability = probabilities[n, direction].flatten()[ids]
                gate = 1+probability-probability.detach()
                packed.append(flat[n, ids]*gate[:, None])
                metadata.append((n, direction, ids))
        lengths = [len(seq) for seq in packed]
        updated = self.ssm(pad_sequence(packed, batch_first=True))
        # Aggregate valid tokens only; padding cannot contaminate descriptors.
        sums, counts, maps = [], [], []
        for n in range(batch):
            orientation_maps, descriptors, support = [], [], []
            for direction in range(self.directions):
                items = [(i, ids) for i, (owner, d, ids) in enumerate(metadata) if owner == n and d == direction]
                if items:
                    ids = torch.cat([ids for _, ids in items])
                    values = torch.cat([updated[i, :lengths[i]] for i, _ in items])
                    restored = x.new_zeros(h*w, channels).index_copy(0, ids, values)
                    descriptors.append(values.mean(0))
                    support.append(True)
                else:
                    restored = x.new_zeros(h*w, channels)
                    descriptors.append(x.new_zeros(channels))
                    support.append(False)
                orientation_maps.append(restored)
            maps.append(torch.stack(orientation_maps))
            sums.append(torch.stack(descriptors))
            counts.append(support)
        descriptor = torch.stack(sums)  # N,O,C
        shuffled = descriptor.transpose(1, 2).reshape(batch, channels*self.directions, 1)
        scores = self.cosa(shuffled).reshape(batch, channels, self.directions).transpose(1, 2)
        if prior_bias is not None:
            scores = scores+prior_bias[:, :, None]
        # Independent sigmoid gates keep prior modulation trainable even when a
        # sample routes every pixel to one direction. Empty directions have zero maps.
        weights = scores.sigmoid()
        # Unequal-length stripe outputs are inverse-mapped BEFORE aligned spatial fusion.
        result = (torch.stack(maps)*weights[:, :, None, :]).sum(1)
        return self.project(result.transpose(1, 2).reshape(batch, channels, h, w))
