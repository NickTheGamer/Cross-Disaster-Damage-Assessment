# Siamese ResNet over per-building crops, with a slot where the graph goes later.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet34, ResNet34_Weights


class SiameseEncoder(nn.Module):
    # shared-weight backbone runs over pre and post crops, one vector out per building
    def __init__(self, combine="subtract", pretrained=True, chunk=256,
                 use_ckpt=False, freeze_bn=True):
        super().__init__()
        net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.trunk = nn.Sequential(*list(net.children())[:-1])   # everything but the classifier
        self.combine, self.chunk, self.use_ckpt = combine, chunk, use_ckpt
        self.freeze_bn = freeze_bn
        self.out_dim = 512 if combine == "subtract" else 1024

    def train(self, mode=True):
        super().train(mode)
        # Freeze batchnorm layers so they don't update their running stats during training,
        # as each xBD chip is a batch, and we don't want neighbood info leaking
        # as that is what the graph approach is for.
        if self.freeze_bn:
            for m in self.trunk.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def _embed(self, x):
        return self.trunk(x).flatten(1)

    def forward(self, pre, post):
        # chunked so peak memory tracks chunk size rather than how many buildings the chip has;
        # checkpointing trades ~30% compute for not storing activations at all
        out = []
        for i in range(0, pre.shape[0], self.chunk):
            a, b = pre[i:i + self.chunk], post[i:i + self.chunk]
            if self.use_ckpt and self.training:
                za = checkpoint(self._embed, a, use_reentrant=False)
                zb = checkpoint(self._embed, b, use_reentrant=False)
            else:
                za, zb = self._embed(a), self._embed(b)
            out.append(zb - za if self.combine == "subtract" else torch.cat([za, zb], dim=1))
        return torch.cat(out)


class NoGraph(nn.Module):
    # ablation arm 1. SageNeck and GatedSageNeck slot in here with the same signature
    def __init__(self, dim):
        super().__init__()
        self.out_dim = dim

    def forward(self, z, edge_index=None):
        return z


class DamageNet(nn.Module):
    def __init__(self, combine="subtract", neck=None, n_classes=4, **kw):
        super().__init__()
        self.encoder = SiameseEncoder(combine=combine, **kw)
        self.neck = neck or NoGraph(self.encoder.out_dim)
        self.head = nn.Linear(self.neck.out_dim, n_classes)

    def forward(self, pre, post, edge_index=None):
        z = self.encoder(pre, post)
        z = self.neck(z, edge_index)
        return self.head(z)
