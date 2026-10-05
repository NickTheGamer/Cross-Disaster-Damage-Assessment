# Siamese ResNet over per-building crops, with a slot where the graph goes later.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SageLayer(nn.Module):
    # h' = W_self h + W_neigh mean(neighbours). With use_graph=False the neighbour term is
    # not built at all, so this is just a linear layer - same shapes, no dead weights.
    # edge_weight turns the plain mean into a weighted one: unused for now, but it is where
    # distance weighting would go and retrofitting it after a grid had run would be painful.
    def __init__(self, in_dim, out_dim, use_graph=True):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim, bias=False) if use_graph else None

    def forward(self, h, edge_index=None, edge_weight=None):
        out = self.self_lin(h)
        if self.neigh_lin is None:
            return out
        src, dst = edge_index
        w = torch.ones(src.shape[0], device=h.device, dtype=h.dtype) if edge_weight is None else edge_weight
        num = torch.zeros_like(h).index_add_(0, dst, h[src] * w[:, None])
        den = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype).index_add_(0, dst, w)
        # isolated nodes get no neighbour term rather than a divide by zero
        return out + self.neigh_lin(num / den.clamp(min=1e-6)[:, None])


class GraphNeck(nn.Module):
    # All three ablation arms are this one class, so they cannot drift apart as the code is
    # edited:
    #     use_graph=False             -> arm 1 "mlp",  no message passing
    #     use_graph=True              -> arm 2 "sage", + graph
    #     use_graph=True, gated=True  -> arm 3 "gate", + gate
    #
    # Arm 1 is the same stack with the neighbour term switched off, so each SageLayer
    # collapses to self_lin(h) and the whole neck is an MLP of identical shape. The
    # dimensions are the point: 512 -> 32 -> 32 in every arm, so the head is Linear(32, 4)
    # throughout and message passing is the only thing that varies between arms 1 and 2. An
    # identity neck would have handed arm 1 the full 512 instead, and then a loss for the
    # graph arm would be unattributable - bottleneck, or graph?
    #
    # Hyperparameters are Ismail's (2 layers, 32 hidden, dropout 0.5) so the comparison is
    # against their setup rather than one we tuned ourselves. If 32 is too narrow it is too
    # narrow for all three arms, so the ablation survives even if absolute scores suffer.
    def __init__(self, in_dim, hidden=32, dropout=0.5, use_graph=True, gated=False,
                 dist=False, sigma_init=250.0):
        super().__init__()
        self.l1 = SageLayer(in_dim, hidden, use_graph)
        self.l2 = SageLayer(hidden, hidden, use_graph)
        self.drop = nn.Dropout(dropout)
        self.gated = gated
        if gated:
            # the gate decides per node how much of the neighbour term to let through.
            # local and agg run through the *same* layers and differ only by that term, so
            # g=1 is exactly arm 1 and g=0 is exactly arm 2 - the gate interpolates between
            # the other two arms rather than between the graph and some third thing.
            self.gate = nn.Linear(2 * hidden, hidden)
        self.dist = dist
        if dist:
            # Learned in log space so sigma stays positive. Initialised at the damage
            # correlation length measured from the *training* folds' assortativity decay,
            # then free to move: a parameter fit by gradient descent on training data, not
            # a hyperparameter we picked by looking at scores. The test fold never sees it.
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
        self.out_dim = hidden
        self.last_gate = None      # stashed for the gate-vs-damage-boundary figure

    def forward(self, z, edge_index=None, edge_len=None):
        # distance decay turns Delaunay's rank-based neighbourhood ("~6 neighbours however
        # far away") into a distance-based one, which is the same in every fold
        w = torch.exp(-edge_len / self.log_sigma.exp()) if self.dist and edge_len is not None else None

        if not self.gated:
            h = self.drop(F.relu(self.l1(z, edge_index, w)))
            return self.l2(h, edge_index, w)

        # one dropout mask shared by both paths, so they differ by the neighbour term and
        # nothing else - two separate drop() calls would add random noise to the comparison
        h_local, h_agg = self.l1.self_lin(z), self.l1(z, edge_index, w)
        mask = self.drop(torch.ones_like(h_local))
        h_local, h_agg = F.relu(h_local) * mask, F.relu(h_agg) * mask

        local = self.l2.self_lin(h_local)
        agg = self.l2(h_agg, edge_index, w)
        g = torch.sigmoid(self.gate(torch.cat([local, agg], dim=1)))
        self.last_gate = g.detach()
        return g * local + (1 - g) * agg


NECKS = {
    "mlp": lambda d, **kw: GraphNeck(d, use_graph=False),
    "sage": lambda d, **kw: GraphNeck(d, use_graph=True),
    "gate": lambda d, **kw: GraphNeck(d, use_graph=True, gated=True),
    "dist": lambda d, **kw: GraphNeck(d, use_graph=True, gated=True, dist=True, **kw),
}


class DamageNet(nn.Module):
    # the three ablation arms differ by this one argument and nothing else
    def __init__(self, arm="mlp", combine="subtract", n_classes=4, sigma_init=None, **kw):
        super().__init__()
        self.encoder = SiameseEncoder(combine=combine, **kw)
        # sigma_init comes from the training folds' measured decay, per LODO run
        self.neck = NECKS[arm](self.encoder.out_dim,
                               **({"sigma_init": sigma_init} if sigma_init else {}))
        self.head = nn.Linear(self.neck.out_dim, n_classes)

    def forward(self, pre, post, edge_index=None, edge_len=None):
        z = self.encoder(pre, post)
        z = self.neck(z, edge_index, edge_len)
        return self.head(z)
