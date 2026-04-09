import argparse
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import importlib

gnn_module = importlib.import_module("06_gnn_model")
InductiveGraphSAGE = gnn_module.InductiveGraphSAGE
load_edge_index_npz = gnn_module.load_edge_index_npz
build_node_features = gnn_module.build_node_features


# ========== 1. 数据集与负采样策略 (Negative Sampling) ==========
class BPRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_items: int, seed: int = 42):
        """
        df 需要包含列：user, item, rating（timestamp 可有可无）
        """
        super(BPRDataset, self).__init__()
        self.data = df.reset_index(drop=True)
        self.users = self.data["user"].to_numpy(dtype=np.int64)
        self.pos_items = self.data["item"].to_numpy(dtype=np.int64)
        self.ratings = self.data["rating"].to_numpy(dtype=np.float32)
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

        # 构建 User 历史交互字典，用于过滤负样本（确保采样的负样本用户绝对没见过）
        print("构建用户历史交互字典以辅助负采样...")
        self.user_history = self.data.groupby("user")["item"].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos_item = int(self.pos_items[idx])
        rating = float(self.ratings[idx])

        # 动态随机负采样 (Dynamic Negative Sampling)
        neg_item = int(self.rng.integers(0, self.num_items))
        while neg_item in self.user_history[user]:
            neg_item = int(self.rng.integers(0, self.num_items))

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float32),
        )


# ========== 2. 多模态推荐打分模块 ==========
class MultimodalRecommender(nn.Module):
    def __init__(self, num_users, item_output_dim):
        super(MultimodalRecommender, self).__init__()
        # 为用户分配可学习的 Embedding
        self.user_embedding = nn.Embedding(num_users, item_output_dim)
        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)

    def forward(self, users, pos_items, neg_items, all_item_embs):
        """
        :param users: batch_size 个用户的 ID
        :param pos_items: batch_size 个正样本物品 ID
        :param neg_items: batch_size 个负样本物品 ID
        :param all_item_embs: GraphSAGE 刚刚输出的全体物品的最新表征 (N, item_output_dim)
        """
        u_emb = self.user_embedding(users)  # (batch_size, dim)
        pos_i_emb = all_item_embs[pos_items]  # (batch_size, dim)
        neg_i_emb = all_item_embs[neg_items]  # (batch_size, dim)

        # 计算内积偏好得分
        pos_scores = (u_emb * pos_i_emb).sum(dim=1)  # (batch_size,)
        neg_scores = (u_emb * neg_i_emb).sum(dim=1)  # (batch_size,)

        return pos_scores, neg_scores


# ========== 3. 加权 BPR 损失函数 ==========
def weighted_bpr_loss(pos_scores, neg_scores, ratings):
    """
    基于 1-5 星评价的加权 BPR 损失
    """
    # 基础 BPR：倾向于最大化正样本和负样本的得分差 (pos - neg)
    base_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores)

    # 评分加权策略 (Rating-weighted): 5星权重1.0，1星权重0.2
    weights = ratings.float() / 5.0
    loss = (base_loss * weights).mean()
    return loss


def _read_interactions_csv(path: str) -> pd.DataFrame:
    """
    尽量兼容两种格式：
    1) 有表头：user_id,item_id,rating,timestamp
    2) 无表头：四列依次为 user,item,rating,timestamp
    返回列名统一为：user, item, rating, timestamp(可选)
    """
    df0 = pd.read_csv(path)
    cols = set(df0.columns.astype(str).tolist())

    # 有表头
    if {"user_id", "item_id"}.issubset(cols):
        df = df0.rename(columns={"user_id": "user", "item_id": "item"}).copy()
        if "rating" not in df.columns:
            raise ValueError("交互文件缺少 rating 列")
        if "timestamp" not in df.columns:
            df["timestamp"] = 0
        return df[["user", "item", "rating", "timestamp"]]

    # 另一种常见表头
    if {"user", "item"}.issubset(cols):
        df = df0.copy()
        if "rating" not in df.columns:
            raise ValueError("交互文件缺少 rating 列")
        if "timestamp" not in df.columns:
            df["timestamp"] = 0
        return df[["user", "item", "rating", "timestamp"]]

    # 无表头兜底
    df = pd.read_csv(path, header=None, names=["user", "item", "rating", "timestamp"])
    return df


def _remap_users_to_contiguous(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    将 user 映射为 0..num_users-1，返回：
    - 重映射后的 interactions df（user 已变成 new_user_id）
    - user_map 表：old_user_id, new_user_id
    """
    old_users = df["user"].astype(str)
    uniq = pd.Index(old_users.unique())
    new_ids = np.arange(len(uniq), dtype=np.int64)
    mapper = pd.Series(new_ids, index=uniq)

    df2 = df.copy()
    df2["user"] = old_users.map(mapper).astype(np.int64)

    user_map = pd.DataFrame({"old_user_id": uniq.astype(str), "new_user_id": new_ids})
    return df2, user_map


@dataclass
class TrainConfig:
    interactions: str = "01_elec_5core_interactions.csv"
    image_feat: str = "04_image_feat_aligned.npy"
    text_feat: str = "04_text_feat_aligned.npy"
    edges: str = "05_joint_knn_edges.npz"
    out_ckpt: str = "multimodal_recommender_final.pth"
    out_user_map: str = "user_id_map.csv"
    batch_size: int = 2048
    epochs: int = 30
    lr: float = 1e-3
    # 解冻 GNN 后，GNN 使用较小学习率（若未指定 lr_gnn，则为 lr * lr_gnn_ratio）
    lr_gnn: Optional[float] = None
    lr_gnn_ratio: float = 0.1
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.2
    weight_decay: float = 1e-4
    # 前若干 epoch 只训练 user embedding，GNN 前向用 no_grad + eval（关闭 dropout）
    freeze_gnn_epochs: int = 5
    seed: int = 42
    log_every: int = 50
    log_file: str = "train.log"
    eval_users: int = 5000
    eval_neg: int = 100
    eval_k: int = 10


def _setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@torch.no_grad()
def _eval_ranking_metrics(
    gnn_model: nn.Module,
    rec_model: MultimodalRecommender,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    user_pos_items: dict[int, list[int]],
    user_history: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    n_users: int = 5000,
    n_neg: int = 100,
    k: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """
    轻量评估：对随机抽样的用户，每人抽 1 个正样本 + n_neg 个负样本，计算：
    - AUC（正样本分数 > 负样本分数的比例）
    - HitRate@K（TopK 是否命中正样本）
    - Precision@K（单正样本设定下，等于 Hit/K）
    - MRR（正样本在候选排序中的倒数排名）
    """
    gnn_model.eval()
    rec_model.eval()

    rng = np.random.default_rng(seed)
    users_all = np.fromiter(user_pos_items.keys(), dtype=np.int64)
    if users_all.size == 0:
        return {"auc": float("nan"), "hitrate@k": float("nan"), "precision@k": float("nan"), "mrr": float("nan")}

    n_users = int(min(n_users, users_all.size))
    sampled_users = rng.choice(users_all, size=n_users, replace=False)
    k = int(max(1, min(k, n_neg + 1)))

    # 全量 item embedding（评估阶段不需要梯度）
    all_item_embs = gnn_model(node_features, edge_index)  # (N, dim)
    user_emb_table = rec_model.user_embedding.weight  # (U, dim)

    hits = 0
    auc_sum = 0.0
    mrr_sum = 0.0

    for u in sampled_users.tolist():
        pos_list = user_pos_items.get(u, [])
        if not pos_list:
            continue
        pos_item = int(rng.choice(pos_list))

        seen = user_history.get(u, set())
        neg_items = []
        # 负采样：保证没见过且不等于 pos
        while len(neg_items) < n_neg:
            ni = int(rng.integers(0, num_items))
            if ni == pos_item or ni in seen:
                continue
            neg_items.append(ni)

        cand_items = np.array([pos_item] + neg_items, dtype=np.int64)
        cand_items_t = torch.from_numpy(cand_items).to(device)

        u_emb = user_emb_table[u]  # (dim,)
        cand_emb = all_item_embs[cand_items_t]  # (1+n_neg, dim)
        scores = (cand_emb * u_emb).sum(dim=1)  # (1+n_neg,)

        pos_score = float(scores[0].item())
        neg_scores = scores[1:]
        auc_sum += float((neg_scores < pos_score).float().mean().item())

        # 排序：pos 在 candidates 中的名次
        order = torch.argsort(scores, descending=True)
        rank = int((order == 0).nonzero(as_tuple=False).item()) + 1  # 1-based
        mrr_sum += 1.0 / rank
        if rank <= k:
            hits += 1

    denom = float(n_users)
    hitrate = hits / denom
    precision = hitrate / float(k)
    auc = auc_sum / denom
    mrr = mrr_sum / denom
    return {"auc": auc, "hitrate@k": hitrate, "precision@k": precision, "mrr": mrr}


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = _setup_logger(cfg.log_file)
    logger.info(f"训练设备: {device}")
    logger.info(f"启动时间: {datetime.datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"训练配置: {cfg.__dict__}")

    for p in [cfg.interactions, cfg.image_feat, cfg.text_feat, cfg.edges]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到文件：{p}")

    # 1) 节点特征 & 图
    x_np = build_node_features(cfg.image_feat, cfg.text_feat)  # (N, 512)
    num_items = x_np.shape[0]
    node_features = torch.tensor(x_np, device=device)
    edge_index_np = load_edge_index_npz(cfg.edges, num_items=num_items)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    logger.info(f"节点特征: {x_np.shape}，edge_index: {edge_index_np.shape}")

    # 2) 交互数据（读入 + user 重映射）
    df = _read_interactions_csv(cfg.interactions)
    # 物品 id 必须是 0..num_items-1
    df["item"] = df["item"].astype(np.int64)
    if df["item"].min() < 0 or df["item"].max() >= num_items:
        raise ValueError(
            f"交互数据里的 item_id 范围不合法：min={df['item'].min()} max={df['item'].max()}，"
            f"但 num_items={num_items}。请确认 item_id 是否与特征的行号对齐。"
        )
    df["rating"] = df["rating"].astype(np.float32)

    df, user_map = _remap_users_to_contiguous(df)
    user_map.to_csv(cfg.out_user_map, index=False, encoding="utf-8-sig")
    num_users = int(user_map.shape[0])
    logger.info(f"用户重映射完成：num_users={num_users}，映射表已保存：{cfg.out_user_map}")

    # 给评估用：每个用户的正样本 item 列表（训练集内抽 1 个正样本做候选评估）
    user_pos_items = df.groupby("user")["item"].apply(list).to_dict()

    dataset = BPRDataset(df=df, num_items=num_items, seed=cfg.seed)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    logger.info(f"统计 -> 总用户: {num_users}, 总物品: {num_items}, 交互数: {len(dataset)}")

    # 3) 模型
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gnn_model = InductiveGraphSAGE(
        node_features.shape[1], cfg.hidden_dim, cfg.output_dim, dropout=cfg.dropout
    ).to(device)
    rec_model = MultimodalRecommender(num_users, cfg.output_dim).to(device)

    lr_gnn_eff = cfg.lr_gnn if cfg.lr_gnn is not None else (cfg.lr * cfg.lr_gnn_ratio)
    opt_user_only = optim.Adam(rec_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_joint = optim.Adam(
        [
            {"params": gnn_model.parameters(), "lr": lr_gnn_eff, "weight_decay": cfg.weight_decay},
            {"params": rec_model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        ]
    )
    logger.info(
        "优化策略: freeze_gnn_epochs=%d | 解冻后 lr_gnn=%.6f | lr_user=%.6f | dropout=%.3f",
        cfg.freeze_gnn_epochs,
        lr_gnn_eff,
        cfg.lr,
        cfg.dropout,
    )

    # 4) 训练
    for epoch in range(cfg.epochs):
        frozen = epoch < cfg.freeze_gnn_epochs
        if frozen:
            gnn_model.eval()
            rec_model.train()
            optimizer = opt_user_only
        else:
            if cfg.freeze_gnn_epochs > 0 and epoch == cfg.freeze_gnn_epochs:
                logger.info(">>> 解冻 GNN：开始联合训练 user embedding + GraphSAGE")
            gnn_model.train()
            rec_model.train()
            optimizer = opt_joint

        total_loss = 0.0
        phase = "freeze_user" if frozen else "joint"
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [{phase}]")
        for step, (users, pos_items, neg_items, ratings) in enumerate(pbar, start=1):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            if frozen:
                # 只更新 user embedding：GNN 前向不建计算图、不更新参数
                with torch.no_grad():
                    all_item_embs = gnn_model(node_features, edge_index)
            else:
                all_item_embs = gnn_model(node_features, edge_index)

            pos_scores, neg_scores = rec_model(users, pos_items, neg_items, all_item_embs)
            loss = weighted_bpr_loss(pos_scores, neg_scores, ratings)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            if step % cfg.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(dataloader))

        # 每个 epoch 结束做一次轻量评估（不需要额外 val/test 文件）
        metrics = _eval_ranking_metrics(
            gnn_model=gnn_model,
            rec_model=rec_model,
            node_features=node_features,
            edge_index=edge_index,
            user_pos_items=user_pos_items,
            user_history=dataset.user_history,
            num_items=num_items,
            device=device,
            n_users=cfg.eval_users,
            n_neg=cfg.eval_neg,
            k=cfg.eval_k,
            seed=cfg.seed + epoch,
        )

        logger.info(
            "Epoch %d/%d | phase=%s | loss=%.6f | AUC=%.4f | HitRate@%d=%.4f | Precision@%d=%.6f | MRR=%.4f",
            epoch + 1,
            cfg.epochs,
            phase,
            avg_loss,
            metrics["auc"],
            cfg.eval_k,
            metrics["hitrate@k"],
            cfg.eval_k,
            metrics["precision@k"],
            metrics["mrr"],
        )

    # 5) 保存
    torch.save(
        {
            "gnn_state_dict": gnn_model.state_dict(),
            "rec_state_dict": rec_model.state_dict(),
            "config": cfg.__dict__,
        },
        cfg.out_ckpt,
    )
    logger.info(f"✅ 模型训练完成并已保存：{cfg.out_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="多模态 GraphSAGE + BPR 训练（更鲁棒版）")
    parser.add_argument("--interactions", default="01_elec_5core_interactions.csv")
    parser.add_argument("--image_feat", default="04_image_feat_aligned.npy")
    parser.add_argument("--text_feat", default="04_text_feat_aligned.npy")
    parser.add_argument("--edges", default="05_joint_knn_edges.npz")
    parser.add_argument("--out_ckpt", default="multimodal_recommender_final.pth")
    parser.add_argument("--out_user_map", default="user_id_map.csv")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="用户 embedding 学习率（联合训练时）")
    parser.add_argument(
        "--lr_gnn",
        type=float,
        default=None,
        help="GNN 学习率；默认不指定时使用 lr * lr_gnn_ratio",
    )
    parser.add_argument("--lr_gnn_ratio", type=float, default=0.1, help="未指定 lr_gnn 时，GNN lr = lr * ratio")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2, help="GraphSAGE 第一层后的 dropout 概率")
    parser.add_argument(
        "--freeze_gnn_epochs",
        type=int,
        default=5,
        help="前多少个 epoch 只训练 user embedding（GNN 冻结前向）",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--log_file", default="train.log")
    parser.add_argument("--eval_users", type=int, default=5000)
    parser.add_argument("--eval_neg", type=int, default=100)
    parser.add_argument("--eval_k", type=int, default=10)
    args = parser.parse_args()

    cfg = TrainConfig(
        interactions=args.interactions,
        image_feat=args.image_feat,
        text_feat=args.text_feat,
        edges=args.edges,
        out_ckpt=args.out_ckpt,
        out_user_map=args.out_user_map,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_gnn=args.lr_gnn,
        lr_gnn_ratio=args.lr_gnn_ratio,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        freeze_gnn_epochs=args.freeze_gnn_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        log_every=args.log_every,
        log_file=args.log_file,
        eval_users=args.eval_users,
        eval_neg=args.eval_neg,
        eval_k=args.eval_k,
    )
    train(cfg)


if __name__ == "__main__":
    main()