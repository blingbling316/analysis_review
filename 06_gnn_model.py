import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv


#  1. GraphSAGE Model Architecture
class InductiveGraphSAGE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, dropout: float = 0.2):
        super(InductiveGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        out = F.normalize(h, p=2, dim=1)
        return out


def load_edge_index_npz(path: str, num_items: int) -> np.ndarray:
    edges_data = np.load(path)
    edge_index_np = edges_data[edges_data.files[0]]

    # Case A: (num_edges, 2) -> (2, num_edges)
    if edge_index_np.ndim == 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    # Case B: (2*num_edges,) -> (2, num_edges)
    elif edge_index_np.ndim == 1:
        edge_index_np = edge_index_np.reshape(2, -1)
    # Case C: (N, K) neighbor matrix -> COO format
    elif edge_index_np.ndim == 2 and edge_index_np.shape[0] == num_items:
        n_nodes, k_neighbors = edge_index_np.shape
        src = np.repeat(np.arange(n_nodes), k_neighbors)
        dst = edge_index_np.reshape(-1)
        edge_index_np = np.vstack([src, dst])

    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape, expected (2, E), got {edge_index_np.shape}")

    return edge_index_np.astype(np.int64)


def build_node_features(image_feat_path: str, text_feat_path: str) -> np.ndarray:
    img_feat = np.load(image_feat_path).astype(np.float32)
    txt_feat = np.load(text_feat_path).astype(np.float32)
    if img_feat.shape[0] != txt_feat.shape[0]:
        raise ValueError(f"Image/text feature row count mismatch: {img_feat.shape[0]} vs {txt_feat.shape[0]}")
    return np.concatenate([img_feat, txt_feat], axis=1).astype(np.float32)


@torch.no_grad()
def export_item_embeddings(
    image_feat_path: str,
    text_feat_path: str,
    edges_path: str,
    hidden_dim: int,
    output_dim: int,
    out_path: str,
    device: torch.device,
    dropout: float = 0.2,
):
    x_np = build_node_features(image_feat_path, text_feat_path)
    x = torch.tensor(x_np, device=device)
    edge_index_np = load_edge_index_npz(edges_path, num_items=x_np.shape[0])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    model = InductiveGraphSAGE(
        feature_dim=x.shape[1], hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout
    ).to(device)
    model.eval()
    item_emb = model(x, edge_index).cpu().numpy().astype(np.float32)
    np.save(out_path, item_emb)
    print(f"Exported item embeddings: {out_path}, shape={item_emb.shape}")


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE model definition + optional item embedding export")
    parser.add_argument("--image_feat", default="04_image_feat_aligned.npy")
    parser.add_argument("--text_feat", default="04_text_feat_aligned.npy")
    parser.add_argument("--edges", default="05_joint_knn_edges.npz")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--export", default="", help="Export item embeddings to .npy path if provided")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.export:
        for p in [args.image_feat, args.text_feat, args.edges]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")
        export_item_embeddings(
            image_feat_path=args.image_feat,
            text_feat_path=args.text_feat,
            edges_path=args.edges,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            out_path=args.export,
            device=device,
            dropout=args.dropout,
        )
    else:
        # Lightweight sanity check without export
        ok = all(os.path.exists(p) for p in [args.image_feat, args.text_feat, args.edges])
        if not ok:
            print("Aligned features or graph edge files not detected. Not required for importing the model class only.")
            return
        x_np = build_node_features(args.image_feat, args.text_feat)
        edge_index_np = load_edge_index_npz(args.edges, num_items=x_np.shape[0])
        print(f"Node features: {x_np.shape}, edge_index: {edge_index_np.shape}")


if __name__ == "__main__":
    main()