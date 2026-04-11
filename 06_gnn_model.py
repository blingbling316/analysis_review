import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ========== 1. GraphSAGE 模型架构 ==========
class InductiveGraphSAGE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, dropout: float = 0.2):
        """
        纯归纳式 GraphSAGE 模型 (不含任何 ID Embedding)
        :param feature_dim: 输入的多模态特征维度 (如 256 + 256 = 512)
        :param hidden_dim: 隐藏层维度
        :param output_dim: 最终输出的节点表征维度
        :param dropout: 第一层 ReLU 后的 dropout 概率（过大时早期难收敛，默认 0.2）
        """
        super(InductiveGraphSAGE, self).__init__()

        # SAGEConv 聚合的是节点的纯特征，不记忆 ID，非常适合冷启动/Zero-Shot
        self.conv1 = SAGEConv(feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        """
        前向传播
        :param x: 节点特征矩阵 (N, feature_dim)
        :param edge_index: 图的边索引 (2, num_edges)
        :return: 最终节点表征 (N, output_dim)
        """
        # 第一层：聚合邻居特征 -> 激活 -> Dropout 防过拟合
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 第二层：输出最终特征
        h = self.conv2(h, edge_index)

        # 推荐系统通常会对最终向量进行 L2 归一化，使得内积等价于余弦相似度
        out = F.normalize(h, p=2, dim=1)

        return out


def load_edge_index_npz(path: str, num_items: int) -> np.ndarray:
    """
    读取 05 脚本产出的 edges（npz），并尽量兼容不同保存格式，最终返回 (2, E) 的 edge_index。
    """
    edges_data = np.load(path)
    edge_index_np = edges_data[edges_data.files[0]]

    # 情况 A: (num_edges, 2) -> (2, num_edges)
    if edge_index_np.ndim == 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    # 情况 B: (2*num_edges,) -> (2, num_edges)
    elif edge_index_np.ndim == 1:
        edge_index_np = edge_index_np.reshape(2, -1)
    # 情况 C: (N, K) 邻居矩阵 -> COO
    elif edge_index_np.ndim == 2 and edge_index_np.shape[0] == num_items:
        n_nodes, k_neighbors = edge_index_np.shape
        src = np.repeat(np.arange(n_nodes), k_neighbors)
        dst = edge_index_np.reshape(-1)
        edge_index_np = np.vstack([src, dst])

    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(f"edge_index 形状不合法，期望 (2, E)，实际 {edge_index_np.shape}")

    return edge_index_np.astype(np.int64)


def build_node_features(image_feat_path: str, text_feat_path: str) -> np.ndarray:
    img_feat = np.load(image_feat_path).astype(np.float32)
    txt_feat = np.load(text_feat_path).astype(np.float32)
    if img_feat.shape[0] != txt_feat.shape[0]:
        raise ValueError(f"图像/文本特征行数不一致：{img_feat.shape[0]} vs {txt_feat.shape[0]}")
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
    print(f"✅ 已导出 item embeddings: {out_path}，shape={item_emb.shape}")


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE 模型定义 +（可选）导出物品向量")
    parser.add_argument("--image_feat", default="04_image_feat_aligned.npy")
    parser.add_argument("--text_feat", default="04_text_feat_aligned.npy")
    parser.add_argument("--edges", default="05_joint_knn_edges.npz")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--export", default="", help="如果提供路径，则会导出 item embeddings（.npy）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.export:
        for p in [args.image_feat, args.text_feat, args.edges]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"未找到文件：{p}")
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
        # 不导出时，仅做一次轻量 sanity check
        ok = all(os.path.exists(p) for p in [args.image_feat, args.text_feat, args.edges])
        if not ok:
            print("未检测到对齐特征或图边文件。仅导入模型类无需这些文件。")
            return
        x_np = build_node_features(args.image_feat, args.text_feat)
        edge_index_np = load_edge_index_npz(args.edges, num_items=x_np.shape[0])
        print(f"✅ 节点特征: {x_np.shape}，edge_index: {edge_index_np.shape}")


if __name__ == "__main__":
    main()