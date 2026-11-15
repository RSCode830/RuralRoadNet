"""SSPM geometry scoring and frozen SAM inference; never reads ground-truth labels."""
import hashlib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def structure_score(mask, alpha=1., beta=1.):
    """Specified implementation choice: covariance anisotropy + inverse width.

    Elongation=(lambda_max-lambda_min)/(lambda_max+lambda_min+eps).
    Thinness=1/(1+sqrt(12*lambda_min)). Coordinates are image pixels.
    """
    weight = np.asarray(mask, dtype=np.float64)
    if weight.ndim != 2 or not np.isfinite(weight).all() or weight.min() < 0 or weight.max() > 1:
        raise ValueError('Proposal must be a finite H,W probability mask')
    mass = weight.sum()
    if mass < 2:
        return 0.
    yy, xx = np.mgrid[:weight.shape[0], :weight.shape[1]]
    dx, dy = xx-(xx*weight).sum()/mass, yy-(yy*weight).sum()/mass
    cov = np.array([[(dx*dx*weight).sum(), (dx*dy*weight).sum()],
                    [(dx*dy*weight).sum(), (dy*dy*weight).sum()]])/mass
    small, large = np.maximum(np.linalg.eigvalsh(cov), 0)
    return float(alpha*(large-small)/(large+small+1e-8)+beta/(1+np.sqrt(12*small)))


def fuse_proposals(masks, shape, top_k=16, alpha=1., beta=1., sharpness=1.):
    if top_k < 1 or alpha <= 0 or beta <= 0 or sharpness <= 0:
        raise ValueError('Positive prior-scoring parameters required')
    if not masks:
        # Eq. 24, sigmoid of an empty sum is 0.5.
        return np.full(shape, .5, dtype=np.float32)
    if any(np.asarray(m).shape != tuple(shape) for m in masks):
        raise ValueError('Proposal shapes disagree')
    scores = np.asarray([structure_score(m, alpha, beta) for m in masks])
    selected = np.argsort(-scores, kind='stable')[:top_k]
    scores = sharpness*scores[selected]
    weights = np.exp(scores-scores.max())
    weights /= weights.sum()
    value = sum(w*np.asarray(masks[i], dtype=np.float64) for i, w in zip(selected, weights))
    return (1/(1+np.exp(-value))).astype(np.float32)


class FrozenSAM:
    def __init__(self, checkpoint, model_type='vit_h', device='cuda', points_per_side=32):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        self.model = sam_model_registry[model_type](checkpoint=checkpoint).to(device).eval()
        self.model.requires_grad_(False)
        # Crop layers=0 permits recovering soft proposals from the same full-image prompts.
        self.generator = SamAutomaticMaskGenerator(self.model, points_per_side=points_per_side, crop_n_layers=0)
        self.predictor = SamPredictor(self.model)

    def __call__(self, rgb, **score_options):
        import torch
        with torch.no_grad():
            proposals = self.generator.generate(rgb)
            self.predictor.set_image(rgb)
            masks = []
            # Recover probability masks using the original automatic point prompts.
            # Match the full-image candidates to each AMG binary mask by overlap.
            cache = {}
            for proposal in proposals:
                points = np.asarray(proposal['point_coords'], dtype=np.float32)
                key = tuple(points.ravel().tolist())
                if key not in cache:
                    logits, _, _ = self.predictor.predict(point_coords=points,
                        point_labels=np.ones(len(points), dtype=np.int32), multimask_output=True, return_logits=True)
                    cache[key] = logits
                logits = cache[key]
                binary = logits > self.model.mask_threshold
                original = proposal['segmentation']
                ious = (binary & original).sum((1, 2))/np.maximum((binary | original).sum((1, 2)), 1)
                masks.append(1/(1+np.exp(-np.clip(logits[int(ious.argmax())], -30, 30))))
            self.predictor.reset_image()
        return fuse_proposals(masks, rgb.shape[:2], **score_options), len(proposals)
