import numpy as np
from losses import cldice


def from_counts(tp, fp, fn):
    def divide(a, b):
        return a/b if b else 1.
    return dict(precision=divide(tp, tp+fp), recall=divide(tp, tp+fn),
                f1=divide(2*tp, 2*tp+fp+fn), road_iou=divide(tp, tp+fp+fn))


class Metrics:
    def __init__(self, threshold=.5):
        self.threshold, self.rows = threshold, []
        self.counts = np.zeros(3, dtype=np.int64)

    def update(self, logits, target):
        prediction = logits.detach().sigmoid() >= self.threshold
        truth = target.bool()
        center_scores = cldice(prediction.float(), target, 10).detach().cpu().tolist()
        for p, t, center in zip(prediction.cpu().numpy(), truth.cpu().numpy(), center_scores):
            tp, fp, fn = int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum())
            self.counts += [tp, fp, fn]
            self.rows.append(dict(from_counts(tp, fp, fn), cldice=center))

    def compute(self):
        if not self.rows:
            raise ValueError('Cannot evaluate empty dataset')
        return {'macro_percent': {k:float(np.mean([r[k] for r in self.rows])*100) for k in self.rows[0]},
                'micro_percent': {k:float(v*100) for k,v in from_counts(*self.counts).items()},
                'tp_fp_fn':self.counts.tolist(), 'images':len(self.rows), 'threshold':self.threshold}
