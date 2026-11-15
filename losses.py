"""BCE + Dice, deep supervision and a differentiable skeleton-overlap regularizer."""
import torch
from torch.nn import functional as F


def erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.minimum(vertical, horizontal)


def skeleton(x, iterations=10):
    def opening(value):
        return F.max_pool2d(erode(value), 3, 1, 1)
    center = F.relu(x-opening(x))
    for _ in range(iterations):
        x = erode(x)
        increment = F.relu(x-opening(x))
        center = center+(1-center)*increment
    return center


def cldice(probability, target, iterations=10, eps=1e-6):
    ps, ts = skeleton(probability, iterations), skeleton(target, iterations)
    dims = (1, 2, 3)
    precision = ((ps*target).sum(dims)+eps)/(ps.sum(dims)+eps)
    recall = ((ts*probability).sum(dims)+eps)/(ts.sum(dims)+eps)
    score = (2*precision*recall+eps)/(precision+recall+eps)
    # Empty target: penalize any foreground instead of rewarding missing skeletons.
    score = torch.where(target.sum(dims) == 0, 1-probability.mean(dims), score)
    return score


def segmentation_loss(logits, target, bce_weight=1., dice_weight=1.):
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = 1-((2*(probability*target).sum(dims)+1)/(probability.sum(dims)+target.sum(dims)+1)).mean()
    return bce_weight*F.binary_cross_entropy_with_logits(logits, target)+dice_weight*dice


def objective(result, target, bce_weight=1., dice_weight=1., aux_weight=.25,
              topology_weight=.1, skeleton_iterations=10):
    main = segmentation_loss(result['logits'], target, bce_weight, dice_weight)
    auxiliary = result['logits'].sum()*0
    for prediction in result['auxiliary']:
        resized = F.interpolate(target, prediction.shape[-2:], mode='nearest')
        auxiliary = auxiliary+aux_weight*segmentation_loss(prediction, resized, bce_weight, dice_weight)
    topology = 1-cldice(result['logits'].sigmoid(), target, skeleton_iterations).mean()
    return main+auxiliary+topology_weight*topology
