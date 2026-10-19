"""CAPTI-Net: chemical abundance and pharmacophore topology interactions."""
import torch
from torch import nn
from torch.nn import functional as F

class CAPTI(nn.Module):
    """Graph logits, abundance and topology combined in a shared assay head.

    The graph encoder is fitted separately. Default outputs match the final
    manuscript; switches reproduce its pathway and interaction controls.
    """

    def __init__(self, tasks, config):
        super().__init__()
        self.config = dict(config)
        rank = config.get('rank', 128)
        self.pair = nn.Parameter(torch.randn(2048, rank) * .03)
        self.register_buffer('fragment_mean', torch.zeros(2048))
        self.scales = nn.ModuleList([nn.Linear(152, rank) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(rank, eps=1e-6) for _ in range(8)])
        self.heads = nn.ModuleList([
            nn.Linear(2 * rank, tasks, bias=False),
            nn.Linear(6 * rank, tasks, bias=False),
        ])
        self.bias = nn.Parameter(torch.zeros(tasks))
        for head in self.heads:
            nn.init.zeros_(head.weight)

    def forward(self, abundance, topology, graph_logits):
        out = self.config.get('base_scale', .5) * graph_logits + self.bias
        dropout = self.config.get('dropout', .5)

        # Typed abundance and centered off-diagonal co-occurrence.
        if self.config.get('i1', True):
            f = F.normalize(abundance[:, :2048], p=2, dim=1)
            linear = f @ self.pair
            centered = f - self.fragment_mean
            projected = linear - self.fragment_mean @ self.pair
            products = .5 * (projected.square() - centered.square() @ self.pair.square())
            linear, products = self.norms[0](linear), self.norms[1](products)
            if not self.config.get('product_i1', True):
                products = torch.zeros_like(products)
            abundance_view = torch.cat([linear, products], 1)
            out = out + self.heads[0](F.dropout(abundance_view, dropout, self.training))

        # Pharmacophore distance scales and their pairwise interactions.
        if self.config.get('i2', True):
            a, b, c = [F.silu(self.norms[2 + i](self.scales[i](topology[:, i]))) for i in range(3)]
            cross_scale = [self.norms[5 + i](z) for i, z in enumerate([a * b, b * c, a * c])]
            if not self.config.get('product_i2', True):
                cross_scale = [torch.zeros_like(z) for z in cross_scale]
            topology_view = torch.cat([a, b, c, *cross_scale], 1)
            out = out + self.heads[1](F.dropout(topology_view, dropout, self.training))
        return out

def masked_loss(logits, target, mask):
    """Average observed-label BCE within tasks, then across observed tasks."""
    cell = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    weights = mask.float()
    counts = weights.sum(0)
    return ((cell * weights).sum(0) / counts.clamp_min(1))[counts > 0].mean()
