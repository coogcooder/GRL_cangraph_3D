"""Graph-attention actor-critic components for PPO training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SimpleGATLayer(nn.Module):
    """Lightweight GAT-like layer that supports edge_index input.

    This avoids a PyG dependency while still providing attention-based
    aggregation over neighbors. Edge features are ignored for simplicity but
    can be concatenated in the message computation if needed.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, Fin), edge_index: (2, E)
        h = self.lin(x)  # (N, heads*out_dim)
        N = h.size(0)
        h = h.view(N, self.heads, self.out_dim)

        # Prepare attention scores
        src, dst = edge_index
        h_src, h_dst = h[src], h[dst]  # (E, heads, out_dim)
        alpha = (h_src * self.attn_l).sum(-1) + (h_dst * self.attn_r).sum(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = torch.softmax(alpha, dim=0)
        alpha = self.dropout(alpha)

        # Message passing
        out = torch.zeros_like(h)
        out.index_add_(0, dst, h_src * alpha.unsqueeze(-1))
        out = out.view(N, self.heads * self.out_dim)
        return out


class GraphAttentionEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 2, num_layers: int = 2):
        super().__init__()
        blocks: List[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers):
            blocks.append(SimpleGATLayer(dim, hidden, heads=heads))
            blocks.append(nn.ReLU())
            dim = hidden * heads
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.blocks:
            if isinstance(layer, SimpleGATLayer):
                h = layer(h, edge_index)
            else:
                h = layer(h)
        # global pooling
        hg = h.transpose(0, 1)  # (F, N)
        hg = self.global_pool(hg).squeeze(-1)
        return h, hg


@dataclass
class PPOAction:
    global_action: torch.Tensor
    local_node: torch.Tensor
    local_size: torch.Tensor
    logp: torch.Tensor
    value: torch.Tensor


class ActorCriticPolicy(nn.Module):
    """Shared encoder with separate global/local policy heads and a value head."""

    def __init__(
        self,
        feature_dim: int,
        num_global_ops: int,
        num_local_sizes: int,
        num_violation_keys: int,
        hidden: int = 128,
        global_candidate_dim: int = 11,
        local_candidate_dim: int = 10,
    ):
        super().__init__()
        self.encoder = GraphAttentionEncoder(feature_dim, hidden=hidden // 2, heads=2, num_layers=2)
        enc_out = self.encoder.out_dim
        self.global_head = _mlp(enc_out, hidden, num_global_ops)
        self.node_head = _mlp(enc_out, hidden, 1)  # per-node logit
        self.global_candidate_head = _mlp(enc_out + global_candidate_dim, hidden, 1)
        self.local_candidate_head = _mlp(enc_out + local_candidate_dim, hidden, 1)
        self.size_head = _mlp(enc_out, hidden, num_local_sizes)
        self.violation_head = _mlp(enc_out, hidden, num_violation_keys)
        self.value_head = _mlp(enc_out, hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        global_cand_feats: Optional[torch.Tensor] = None,
        local_cand_feats: Optional[torch.Tensor] = None,
    ):
        node_h, graph_h = self.encoder(x, edge_index)
        if global_cand_feats is None:
            global_logits = self.global_head(graph_h)
        else:
            if global_cand_feats.numel() == 0:
                global_logits = torch.zeros((0,), device=x.device)
            else:
                g_expand = graph_h.expand(global_cand_feats.shape[0], -1)
                global_logits = self.global_candidate_head(torch.cat([g_expand, global_cand_feats], dim=-1)).squeeze(-1)

        if local_cand_feats is None:
            local_logits = self.node_head(node_h).squeeze(-1)
        else:
            if local_cand_feats.numel() == 0:
                local_logits = torch.zeros((0,), device=x.device)
            else:
                g_expand = graph_h.expand(local_cand_feats.shape[0], -1)
                local_logits = self.local_candidate_head(torch.cat([g_expand, local_cand_feats], dim=-1)).squeeze(-1)

        size_logits = self.size_head(graph_h)
        value = self.value_head(graph_h).squeeze(-1)
        return graph_h, global_logits, local_logits, size_logits
    

    def act(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        *,
        epsilon: float = 0.1,
        verbose: bool = False,
        step_label: str = "",
    ) -> PPOAction:
        meta = masks.get("meta", {}) if isinstance(masks, dict) else {}
        global_cand_feats = meta.get("global_candidate_feats")
        local_cand_feats = meta.get("local_candidate_feats")
        local_candidate_indices = meta.get("local_candidate_indices")
        global_candidate_map = meta.get("global_candidate_map")

        graph_h, global_logits, node_logits, size_logits = self(
            x,
            edge_index,
            global_cand_feats=global_cand_feats,
            local_cand_feats=local_cand_feats,
        )
        value = self.value_head(graph_h).squeeze(-1)
        # Apply masks if provided
        if masks.get("global") is not None:
            global_logits = global_logits + masks["global"]
        if masks.get("node") is not None:
            node_logits = node_logits + masks["node"]
        if masks.get("global_scores") is not None:
            global_logits = global_logits + masks["global_scores"]
        
        if masks.get("size") is not None:
            size_logits = size_logits + masks["size"]
        if global_logits.numel() == 0:
            g_action = torch.tensor(0, device=x.device)
            g_logp = torch.tensor(0.0, device=x.device)
        else:
            global_dist = torch.distributions.Categorical(logits=global_logits)
            g_action = global_dist.sample()
            g_logp = global_dist.log_prob(g_action)

        if node_logits.numel() == 0:
            n_action = torch.tensor(0, device=x.device)
            n_logp = torch.tensor(0.0, device=x.device)
        else:
            node_dist = torch.distributions.Categorical(logits=node_logits)
            n_action = node_dist.sample()
            n_logp = node_dist.log_prob(n_action)

        size_dist = torch.distributions.Categorical(logits=size_logits)
        s_action = size_dist.sample()
        s_logp = size_dist.log_prob(s_action)

        logp = g_logp + n_logp + s_logp

        if local_candidate_indices is not None and node_logits.numel() > 0:
            n_action = torch.tensor(local_candidate_indices[int(n_action.item())], device=x.device)
        if global_candidate_map is not None and global_logits.numel() > 0:
            g_action = torch.tensor(global_candidate_map[int(g_action.item())], device=x.device)

        if verbose:
            # Compact insight into policy inputs/outputs for debugging
            # "feature_stats" reports min/mean/max over all node features passed
            # into the GNN encoder for the current observation; this helps confirm
            # the graph state is being populated as expected before sampling.
            x_cpu = x.detach().cpu()
            stats = (float(x_cpu.min()), float(x_cpu.max()), float(x_cpu.mean()))
            # Show a small slice of the actual inputs heading into the encoder to
            # verify values without overwhelming the log. We log the first two node
            # feature rows and the first five edges.
            node_preview = x_cpu[:2]
            edge_preview = edge_index.detach().cpu()[:, :5]

            # Pull any action metadata (counts/labels) supplied by the environment
            meta = masks.get("meta", {}) if isinstance(masks, dict) else {}
            global_labels = meta.get("global_labels", [])
            num_global = meta.get("num_global", int(global_logits.shape[-1]))
            num_nodes = meta.get("num_nodes", int(node_logits.shape[-1]))

            print(
                f"[policy] {step_label} nodes={x.shape[0]} feats={x.shape[1]} "
                f"edges={edge_index.shape[1]} feature_stats(min/mean/max)={stats}"
            )
            print(
                "[policy] encoder input sample → nodes[:2]=",
                node_preview.numpy(),
                "edges[:,:5]=",
                edge_preview.numpy(),
            )
            print(
                # Global logits correspond to the add/remove span choices (one logit per
                # candidate girder line). The vector length equals twice the number of
                # girder spans the environment exposes. Node logits rank which member to
                # tune locally, and size logits (shown for the first two nodes) rank
                # discrete section size picks for each member.
                f"[policy] global logits (count={num_global})=",
                global_logits.detach().cpu().numpy(),
                "node logits (sample)=",
                node_logits.detach().cpu().numpy()[:5],
                "size logits (sample)=",
                size_logits.detach().cpu().numpy()[:2],
            )
            if global_labels:
                sample_labels = ", ".join(global_labels[:6])
                print(f"[policy] global action labels (first few): {sample_labels} …")
            print(
                f"[policy] chosen global={g_action.item()} node={n_action.item()} size={s_action.item()}"
                f" | num_global={num_global} num_nodes={num_nodes}"
            )
        return PPOAction(g_action, n_action, s_action, logp, value)