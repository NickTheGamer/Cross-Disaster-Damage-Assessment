# Loss for node-level damage classification.

import torch
import torch.nn.functional as F


def damage_loss(logits, y, class_weights=None):
    # -1 is un-classified: the building stays in the graph and feeds its neighbours,
    # but we never had a label for it so it contributes nothing here
    if not (y >= 0).any():
        # cross_entropy returns nan rather than complaining when everything is masked,
        # so force a zero loss in that case.
        return logits.sum() * 0
    return F.cross_entropy(logits, y, weight=class_weights, ignore_index=-1)


def inverse_frequency_weights(counts):
    # counts is a 4-vector of per-class building counts from the training folds
    counts = torch.as_tensor(counts, dtype=torch.float32)
    w = counts.sum() / (len(counts) * counts.clamp(min=1))
    return w / w.mean()


# Focal loss goes here when we get to imbalance handling - same signature as damage_loss
# so the training loop does not need to care which one it is calling.