import argparse
import os
import numpy as np
import torch
import importlib


def load_edge_index_npz(path, num_items):
    edges_data = np.load(path)
    edge_index_np = edges_data[edges_data.files[0]]

    # 兼容几种常见形状
    if len(edge_index_np.shape) == 2 and edge_index_np.shape[1] == 2:
        # (num_edges, 2) -> (2, num_edges)
        edge_index_np = edge_index_np.T
    elif len(edge_index_np.shape) == 1:
        # (2*num_edges,) -> (2, num_edges)
        edge_index_np = edge_index_np.reshape(2, -1)
    elif len(edge_index_np.shape) == 2 and edge_index_np.shape[0] == num_items:
        # (N, K) 邻居矩阵 -> COO 边
        n_nodes, k_neighbors = edge_index_np.shape
        src = np.repeat(np.arange(n_nodes), k_neighbors)
        dst = edge_index_np.flatten()
        edge_index_np = np.vstack([src, dst])

    if edge_index_np.shape[0] != 2:
        raise ValueError(f"edge_index 形状不对，期望 (2, E)，实际: {edge_index_np.shape}")
    return edge_index_np.astype(np.int64)


@torch.no_grad()
def recommend_topn(user_id, all_item_embs, user_embedding_weight, topn=20, exclude_items=None):
    """
    all_item_embs: (N, dim) torch tensor on device
    user_embedding_weight: (num_users, dim) torch tensor on device
    """
    if user_id < 0 or user_id >= user_embedding_weight.shape[0]:
        raise ValueError(
            f"user_id 越界：{user_id}，有效范围 [0, {user_embedding_weight.shape[0] - 1}]。"
            f"（当前模型不支持新用户冷启动）"
        )

    u = user_embedding_weight[user_id]  # (dim,)
    scores = (all_item_embs * u).sum(dim=1)  # (N,)

    if exclude_items is not None and len(exclude_items) > 0:
        exclude_items = np.asarray(list(exclude_items), dtype=np.int64)
        exclude_items = exclude_items[(exclude_items >= 0) & (exclude_items < scores.shape[0])]
        scores[torch.from_numpy(exclude_items).to(scores.device)] = -1e9

    k = min(topn, scores.shape[0])
    vals, idx = torch.topk(scores, k=k, largest=True)
    return idx.cpu().numpy(), vals.cpu().numpy()


@torch.no_grad()
def similar_items(item_id, all_item_embs, topk=20):
    if item_id < 0 or item_id >= all_item_embs.shape[0]:
        raise ValueError(f"item_id 越界：{item_id}，有效范围 [0, {all_item_embs.shape[0] - 1}]。")
    q = all_item_embs[item_id]  # (dim,)
    sims = all_item_embs @ q  # 归一化后等价余弦；即便没归一化也可作为相似度分数
    sims[item_id] = -1e9
    k = min(topk, sims.shape[0] - 1)
    vals, idx = torch.topk(sims, k=k, largest=True)
    return idx.cpu().numpy(), vals.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="使用训练好的多模态推荐模型做推理")
    parser.add_argument("--ckpt", default="multimodal_recommender_final.pth", help="训练脚本保存的模型文件")
    parser.add_argument("--img", default="image_feat_aligned.npy", help="对齐后的图像特征 (N,256)")
    parser.add_argument("--txt", default="text_feat_aligned.npy", help="对齐后的文本特征 (N,256)")
    parser.add_argument("--edges", default="joint_knn_edges.npz", help="KNN 图边文件")

    sub = parser.add_subparsers(dest="cmd", required=True)
    p_rec = sub.add_parser("recommend", help="给定 user_id 输出 Top-N 推荐")
    p_rec.add_argument("--user_id", type=int, required=True)
    p_rec.add_argument("--topn", type=int, default=20)
    p_rec.add_argument("--exclude_seen", action="store_true", help="是否排除用户训练时见过的物品（需要 interactions.csv）")
    p_rec.add_argument("--interactions", default="01_elec_5core_interactions.csv", help="交互文件（用于 exclude_seen）")

    p_sim = sub.add_parser("similar", help="给定 item_id 输出 Top-K 相似物品（基于最终 item embedding）")
    p_sim.add_argument("--item_id", type=int, required=True)
    p_sim.add_argument("--topk", type=int, default=20)

    args = parser.parse_args()

    for p in [args.ckpt, args.img, args.txt, args.edges]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到文件：{p}")

    # 动态导入 GNN 定义
    gnn_module = importlib.import_module("06_gnn_model")
    InductiveGraphSAGE = gnn_module.InductiveGraphSAGE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")

    # 1) 加载特征
    img_feat = np.load(args.img).astype(np.float32)
    txt_feat = np.load(args.txt).astype(np.float32)
    x_np = np.concatenate([img_feat, txt_feat], axis=1)  # (N, 512)
    node_features = torch.tensor(x_np, device=device)
    num_items = node_features.shape[0]

    # 2) 加载边
    edge_index_np = load_edge_index_npz(args.edges, num_items=num_items)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    # 3) 从 ckpt 推断维度并加载模型
    ckpt = torch.load(args.ckpt, map_location=device)
    gnn_state = ckpt["gnn_state_dict"]
    rec_state = ckpt["rec_state_dict"]

    # 从权重形状推断 hidden_dim / output_dim
    # SAGEConv 的 lin_l.weight 形状通常是 (out_channels, in_channels)
    conv1_w = gnn_state["conv1.lin_l.weight"]
    conv2_w = gnn_state["conv2.lin_l.weight"]
    hidden_dim = conv1_w.shape[0]
    output_dim = conv2_w.shape[0]
    feature_dim = node_features.shape[1]

    gnn_model = InductiveGraphSAGE(feature_dim, hidden_dim, output_dim).to(device)
    gnn_model.load_state_dict(gnn_state)
    gnn_model.eval()

    # user_embedding.weight 形状：(num_users, output_dim)
    user_embedding_weight = rec_state["user_embedding.weight"]
    user_embedding_weight = torch.tensor(user_embedding_weight, device=device)

    # 4) 计算所有 item embedding（一次性算出来，后面复用）
    all_item_embs = gnn_model(node_features, edge_index)  # (N, output_dim)

    if args.cmd == "similar":
        idx, vals = similar_items(args.item_id, all_item_embs, topk=args.topk)
        print(f"item_id={args.item_id} 的 Top-{len(idx)} 相似物品：")
        for rank, (i, s) in enumerate(zip(idx, vals), start=1):
            print(f"{rank:02d}. item_id={int(i)}\tscore={float(s):.6f}")
        return

    # recommend
    exclude_items = None
    if args.exclude_seen:
        if not os.path.exists(args.interactions):
            raise FileNotFoundError(f"未找到交互文件：{args.interactions}")
        # 交互文件有两种常见格式：
        # - 有表头：user_id,item_id,rating,timestamp
        # - 无表头：直接四列
        # 这里做一个尽量鲁棒的读取
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("需要 pandas 才能使用 --exclude_seen") from e

        df = pd.read_csv(args.interactions)
        if "user" in df.columns and "item" in df.columns:
            ucol, icol = "user", "item"
        elif "user_id" in df.columns and "item_id" in df.columns:
            ucol, icol = "user_id", "item_id"
        else:
            # 兜底：按无表头读取
            df = pd.read_csv(args.interactions, names=["user", "item", "rating", "timestamp"])
            ucol, icol = "user", "item"

        seen = set(df.loc[df[ucol] == args.user_id, icol].astype(int).tolist())
        exclude_items = seen

    idx, vals = recommend_topn(
        user_id=args.user_id,
        all_item_embs=all_item_embs,
        user_embedding_weight=user_embedding_weight,
        topn=args.topn,
        exclude_items=exclude_items,
    )

    print(f"user_id={args.user_id} 的 Top-{len(idx)} 推荐：")
    for rank, (i, s) in enumerate(zip(idx, vals), start=1):
        print(f"{rank:02d}. item_id={int(i)}\tscore={float(s):.6f}")


if __name__ == "__main__":
    main()

