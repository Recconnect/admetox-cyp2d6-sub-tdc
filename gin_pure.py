"""Pure-PyTorch reimplementation of dgllife gin_supervised_contextpred.

dgllife's dgl is CPU-only on this machine (no ROCm Windows wheel), so we
reimplement the exact GIN forward pass with torch.scatter ops to run on the
ROCm GPU. Numerically verified against the dgl GIN (diff 0.0).

Graph convention (matches pretraining): smiles_to_bigraph with
add_self_loop=True + PretrainBondFeaturizer(self_loop=True).
Node feats: atomic_number, chirality_type. Edge feats: bond_type, bond_direction.
Each GINLayer owns its own edge embeddings (matches dgllife state dict).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 300


def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


class GINLayerPure(nn.Module):
    """sum_{u in N(v)} (h_u + e_uv) -> MLP(300->600->300) -> BN -> act"""

    def __init__(self, num_edge_emb_list):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, 2 * EMB_DIM),
            nn.ReLU(),
            nn.Linear(2 * EMB_DIM, EMB_DIM),
        )
        self.bn = nn.BatchNorm1d(EMB_DIM)
        self.edge_embeddings = nn.ModuleList(
            [nn.Embedding(n, EMB_DIM) for n in num_edge_emb_list])

    def forward(self, h, src, dst, cat_edge_feats):
        # h: [N, 300], src/dst: [E], cat_edge_feats: list of [E] LongTensors
        e = sum(self.edge_embeddings[i](f) for i, f in enumerate(cat_edge_feats))
        msg = h[src] + e
        agg = scatter_sum(msg, dst, h.size(0))  # [N, 300]
        return self.bn(self.mlp(agg))


class GINPure(nn.Module):
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5,
                 JK="last", dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)
        self.node_embeddings = nn.ModuleList(
            [nn.Embedding(n, EMB_DIM) for n in num_node_emb_list])
        self.gnn_layers = nn.ModuleList(
            [GINLayerPure(num_edge_emb_list) for _ in range(num_layers)])

    def forward(self, node_cat_feats, src, dst, edge_cat_feats):
        # node_cat_feats: list of [N] LongTensors (atom, chir)
        # edge_cat_feats: list of [E] LongTensors (bond_type, bond_dir)
        h0 = sum(self.node_embeddings[i](f) for i, f in enumerate(node_cat_feats))
        all_layer_node_feats = [h0]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](all_layer_node_feats[-1],
                                                src, dst, edge_cat_feats)
            if layer < self.num_layers - 1:
                node_feats = F.relu(node_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)
        return all_layer_node_feats[-1]


def build_pure_model(dgl_model, device="cpu"):
    """Copy weights from a loaded dgllife GIN into a GINPure on `device`."""
    pure = GINPure(
        num_node_emb_list=[dgl_model.node_embeddings[i].weight.size(0) for i in range(2)],
        num_edge_emb_list=[e.weight.size(0) for e in dgl_model.gnn_layers[0].edge_embeddings],
    )
    with torch.no_grad():
        for i in range(2):
            pure.node_embeddings[i].weight.copy_(dgl_model.node_embeddings[i].weight)
        for li, dl in enumerate(dgl_model.gnn_layers):
            pl = pure.gnn_layers[li]
            for j in range(2):
                pl.edge_embeddings[j].weight.copy_(
                    dl.edge_embeddings[j].weight)
            for sm, pm in zip(dl.mlp, pl.mlp):
                if isinstance(sm, nn.Linear):
                    pm.weight.copy_(sm.weight); pm.bias.copy_(sm.bias)
            pl.bn.weight.copy_(dl.bn.weight)
            pl.bn.bias.copy_(dl.bn.bias)
            pl.bn.running_mean.copy_(dl.bn.running_mean)
            pl.bn.running_var.copy_(dl.bn.running_var)
            pl.bn.num_batches_tracked.copy_(dl.bn.num_batches_tracked)
    pure.to(device)
    pure.eval()
    return pure


def state_to_pure(state_dict, device="cpu"):
    """Build GINPure from a raw state dict (keys as in dgl GIN)."""
    node_sizes = [v.size(0) for k, v in state_dict.items()
                  if k.startswith("node_embeddings.") and k.endswith(".weight")]
    edge_sizes = [v.size(0) for k, v in state_dict.items()
                  if k.startswith("gnn_layers.0.edge_embeddings.") and k.endswith(".weight")]
    n_layers = sum(1 for k in state_dict
                   if k.startswith("gnn_layers.") and k.endswith(".mlp.0.weight"))
    pure = GINPure(num_node_emb_list=node_sizes, num_edge_emb_list=edge_sizes,
                   num_layers=n_layers)
    with torch.no_grad():
        for i, s in enumerate(node_sizes):
            pure.node_embeddings[i].weight.copy_(state_dict[f"node_embeddings.{i}.weight"])
        for li in range(n_layers):
            pl = pure.gnn_layers[li]
            for j in range(2):
                pl.edge_embeddings[j].weight.copy_(
                    state_dict[f"gnn_layers.{li}.edge_embeddings.{j}.weight"])
            pl.mlp[0].weight.copy_(state_dict[f"gnn_layers.{li}.mlp.0.weight"])
            pl.mlp[0].bias.copy_(state_dict[f"gnn_layers.{li}.mlp.0.bias"])
            pl.mlp[2].weight.copy_(state_dict[f"gnn_layers.{li}.mlp.2.weight"])
            pl.mlp[2].bias.copy_(state_dict[f"gnn_layers.{li}.mlp.2.bias"])
            pl.bn.weight.copy_(state_dict[f"gnn_layers.{li}.bn.weight"])
            pl.bn.bias.copy_(state_dict[f"gnn_layers.{li}.bn.bias"])
            pl.bn.running_mean.copy_(state_dict[f"gnn_layers.{li}.bn.running_mean"])
            pl.bn.running_var.copy_(state_dict[f"gnn_layers.{li}.bn.running_var"])
            pl.bn.num_batches_tracked.copy_(state_dict[f"gnn_layers.{li}.bn.num_batches_tracked"])
    pure.to(device)
    pure.eval()
    return pure


def batch_tensors(graphs, indices, device):
    """Concatenate per-mol CPU tensors for a batch into GPU tensors.
    Returns dict with node_cat_feats, edge_cat_feats, src, dst, and per-graph
    node offsets / batch index list for mean pooling."""
    n = sum(graphs[j]["n_nodes"] for j in indices)
    E = sum(graphs[j]["src"].numel() for j in indices)
    atom_id = torch.empty(n, dtype=torch.long, device=device)
    chir = torch.empty(n, dtype=torch.long, device=device)
    bond_type = torch.empty(E, dtype=torch.long, device=device)
    bond_dir = torch.empty(E, dtype=torch.long, device=device)
    src = torch.empty(E, dtype=torch.long, device=device)
    dst = torch.empty(E, dtype=torch.long, device=device)
    graph_idx = torch.empty(n, dtype=torch.long, device=device)
    node_off, edge_off = 0, 0
    for k, j in enumerate(indices):
        g = graphs[j]
        N, El = g["n_nodes"], g["src"].numel()
        sl = slice(node_off, node_off + N)
        atom_id[sl] = g["atom_id"].to(device)
        chir[sl] = g["chir"].to(device)
        graph_idx[sl] = k
        sl_e = slice(edge_off, edge_off + El)
        bond_type[sl_e] = g["bond_type"].to(device)
        bond_dir[sl_e] = g["bond_dir"].to(device)
        src[sl_e] = g["src"].to(device) + node_off
        dst[sl_e] = g["dst"].to(device) + node_off
        node_off += N; edge_off += El
    return {"atom_id": atom_id, "chir": chir, "bond_type": bond_type,
            "bond_dir": bond_dir, "src": src, "dst": dst,
            "graph_idx": graph_idx, "n_nodes": node_off}


def mean_pool(h, ginfo):
    """h: [N, 300], ginfo from batch_tensors -> [B, 300]."""
    n_per_graph = torch.zeros(ginfo["graph_idx"].max() + 1, device=h.device)
    n_per_graph.index_add_(0, ginfo["graph_idx"], torch.ones(h.size(0), device=h.device))
    sums = torch.zeros(n_per_graph.size(0), h.size(1), device=h.device)
    sums.index_add_(0, ginfo["graph_idx"], h)
    return sums / n_per_graph.unsqueeze(1)
