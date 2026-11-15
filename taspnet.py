"""Topology-preserving encoder and prior-guided decoder, Algorithm 1."""
import torch
from torch import nn
from torch.nn import functional as F
from scanning import TopologyMamba


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TNMB(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.mamba = TopologyMamba(channels, **kwargs)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 2*channels, 1), nn.GELU(),
                                 nn.Conv2d(2*channels, 2*channels, 3, padding=1, groups=2*channels),
                                 nn.GELU(), nn.Conv2d(2*channels, channels, 1))
        self.gamma1, self.gamma2 = nn.Parameter(torch.tensor(.1)), nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        x = x+self.gamma1*self.mamba(self.norm1(x))
        return x+self.gamma2*self.ffn(self.norm2(x))


class PGTMB(nn.Module):
    def __init__(self, channels, directions=4, gate_epsilon=1e-6, **kwargs):
        super().__init__()
        self.epsilon = gate_epsilon
        self.refine = nn.Conv2d(channels, 1, 3, padding=1)
        self.prior_mlp = nn.Sequential(nn.Conv2d(1, channels, 1), nn.SiLU(), nn.Conv2d(channels, channels, 1))
        self.global_mlp = nn.Sequential(nn.Linear(1, directions), nn.SiLU(), nn.Linear(directions, directions))
        self.mamba = TopologyMamba(channels, directions=directions, **kwargs)
        self.gamma = nn.Parameter(torch.tensor(.1))

    def forward(self, x, prior):
        # Literal Eq. 31: probability prior + residual, not logit(prior) + residual.
        refined = (prior+self.refine(x)).sigmoid()
        gate = refined/(refined+self.epsilon)
        modulated = x+gate*self.prior_mlp(refined)*x
        bias = self.global_mlp(refined.mean((2, 3)))
        return x+self.gamma*self.mamba(modulated, bias), refined


class TaSPNet(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192), depths=(2, 2, 2),
                 decoder_depth=2, state_dim=8, stripe_width=8, directions=4,
                 backend='parallel', gate_epsilon=1e-6):
        super().__init__()
        if len(widths) != len(depths)+1 or any(d < 1 for d in depths) or decoder_depth < 1:
            raise ValueError('Need one depth per downsampling stage and positive depths')
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)
        self.down, self.encoder = nn.ModuleList(), nn.ModuleList()
        common = dict(state_dim=state_dim, stripe_width=stripe_width, directions=directions, backend=backend)
        for i, depth in enumerate(depths):
            self.down.append(nn.Conv2d(widths[i], widths[i+1], 3, stride=2, padding=1))
            self.encoder.append(nn.Sequential(*[TNMB(widths[i+1], shifted=b % 2 == 1, **common) for b in range(depth)]))
        self.fuse, self.decoder, self.aux = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for level in range(len(widths)-2, -1, -1):
            c = widths[level]
            self.fuse.append(nn.Conv2d(widths[level+1]+c, c, 3, padding=1))
            self.decoder.append(nn.ModuleList([PGTMB(c, shifted=b % 2 == 1, gate_epsilon=gate_epsilon, **common)
                                               for b in range(decoder_depth)]))
            self.aux.append(nn.Conv2d(c+1, 1, 1))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, image, prior):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected N,3,H,W image')
        if prior is None or prior.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError('Expected aligned N,1,H,W structure prior')
        if not torch.isfinite(prior).all() or prior.min() < 0 or prior.max() > 1:
            raise ValueError('Prior must contain finite probabilities in [0,1]')
        features = [self.stem(image)]
        for down, stage in zip(self.down, self.encoder):
            features.append(stage(down(features[-1])))
        x, auxiliary, refined_priors = features[-1], [], []
        for skip, fuse, blocks, aux in zip(reversed(features[:-1]), self.fuse, self.decoder, self.aux):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = fuse(torch.cat([x, skip], 1))
            p = F.interpolate(prior, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            for block in blocks:
                x, p = block(x, p)
            auxiliary.append(aux(torch.cat([x, p], 1)))
            refined_priors.append(p)
        return {'logits': self.head(x), 'auxiliary': auxiliary, 'refined_priors': refined_priors}
