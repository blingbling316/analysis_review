import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# 统一对照实验脚本
# 支持：
# 1) aligned + GNN
# 2) raw + GNN
# 3) aligned + noGNN
# 4) raw + noGNN
#
# 所有实验共用：
# - 同一份 item-cold-start split (04_item_cold_split.npz)
# - 同一套 train/val/test 用户过滤逻辑
# - 同一套评估逻辑
#
# 输出：
# - 每个实验一个 train_log.txt
# - 每个实验一个 history.csv
# - 全部实验一个 ablation_summary.csv
# ============================================================


@dataclass
class Config:
    # ---------- 特征文件 ----------
    raw_img_feat_path: str = "03_image_feat.npy"
    raw_txt_feat_path: str = "02_text_feat.npy"
    aligned_img_feat_path: str = "04_image_feat_aligned_item_coldstart.npy"
    aligned_txt_feat_path: str = "04_text_feat_aligned_item_coldstart.npy"

    # ---------- 公共输入 ----------
    edge_path: str = "05_joint_knn_edges_02.npz"
    interaction_path: str = "01_elec_5core_interactions.csv"
    split_path: str = "04_item_cold_split.npz"

    # ---------- 随机性 ----------
    random_seed: int = 42

    # ---------- 训练 ----------
    epochs: int = 5
    batch_size: int = 128
    steps_per_epoch: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_amp: bool = True
    max_history: int = 20

    # ---------- GNN ----------
    gnn_hidden_dim: int = 128
    gnn_out_dim: int = 64
    gnn_dropout: float = 0.1
    fanouts: Tuple[int, int] = (15, 10)

    # ---------- noGNN ----------
    mlp_hidden_dim: int = 256
    mlp_out_dim: int = 64
    mlp_dropout: float = 0.1

    # ---------- 评估 ----------
    eval_top_k: Tuple[int, ...] = (10, 20, 50)
    eval_negatives: int = 99
    eval_max_users: int = 500

    # ---------- 运行哪些实验 ----------
    # 可选: aligned_gnn, raw_gnn, aligned_nognn, raw_nognn
    experiments: Tuple[str, ...] = (
        "aligned_gnn",
        "raw_gnn",
        "aligned_nognn",
        "raw_nognn",
    )

    # ---------- 输出 ----------
    save_root: str = "outputs_ablation_item_coldstart"

    # ---------- 设备 ----------
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
os.makedirs(cfg.save_root, exist_ok=True)
print(f"Device: {cfg.device}, AMP: {cfg.use_amp}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.random_seed)
_rng = np.random.default_rng(cfg.random_seed)


# ============================================================
# 日志
# ============================================================
class TxtLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.fp = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        print(msg)
        self.fp.write(msg + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()


# ============================================================
# 通用工具
# ============================================================
def clip_history(hist: List[int], max_len: int) -> List[int]:
    if len(hist) <= max_len:
        return hist
    pick = _rng.choice(len(hist), size=max_len, replace=False)
    return [hist[i] for i in pick.tolist()]


def build_user_histories(df: pd.DataFrame, item_mask: np.ndarray) -> Dict[int, List[int]]:
    sub = df[df["item_id"].isin(np.where(item_mask)[0])]
    return sub.groupby("user_id")["item_id"].apply(list).to_dict()


def load_split(num_items: int, logger: TxtLogger):
    if not os.path.exists(cfg.split_path):
        raise FileNotFoundError(f"找不到 split 文件: {cfg.split_path}")

    data = np.load(cfg.split_path)
    required = {"train_mask", "val_mask", "test_mask"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"split 文件缺字段，实际字段: {data.files}")

    train_mask = data["train_mask"].astype(bool)
    val_mask = data["val_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)

    if len(train_mask) != num_items:
        raise ValueError(f"split 长度和特征行数不一致: split={len(train_mask)}, num_items={num_items}")

    logger.log(f"[INFO] Loaded split from: {cfg.split_path}")
    logger.log(
        f"[INFO] Item split | train={train_mask.sum():,} | val-cold={val_mask.sum():,} | test-cold={test_mask.sum():,}"
    )
    logger.log(
        f"[INFO] Split overlap check | train&val={int((train_mask & val_mask).sum())} | "
        f"train&test={int((train_mask & test_mask).sum())} | "
        f"val&test={int((val_mask & test_mask).sum())}"
    )
    return train_mask, val_mask, test_mask


def prepare_data(num_items: int, logger: TxtLogger):
    interactions = pd.read_csv(cfg.interaction_path)
    train_mask, val_mask, test_mask = load_split(num_items, logger)

    train_user_items = build_user_histories(interactions, train_mask)
    val_user_items = build_user_histories(interactions, val_mask)
    test_user_items = build_user_histories(interactions, test_mask)

    train_users = [u for u, items in train_user_items.items() if len(items) >= 2]
    valid_val_users = [u for u in val_user_items if len(train_user_items.get(u, [])) >= 1 and len(val_user_items[u]) >= 1]
    valid_test_users = [u for u in test_user_items if len(train_user_items.get(u, [])) >= 1 and len(test_user_items[u]) >= 1]

    train_item_ids = np.where(train_mask)[0]

    logger.log(f"[INFO] Total interactions: {len(interactions):,}")
    train_interactions = interactions[interactions["item_id"].isin(np.where(train_mask)[0])]
    val_interactions = interactions[interactions["item_id"].isin(np.where(val_mask)[0])]
    test_interactions = interactions[interactions["item_id"].isin(np.where(test_mask)[0])]
    logger.log(
        f"[INFO] Interactions by split | train={len(train_interactions):,} | "
        f"val-cold={len(val_interactions):,} | test-cold={len(test_interactions):,}"
    )

    logger.log(f"[INFO] Train users usable for BPR: {len(train_users):,}")
    logger.log(f"[INFO] Val users with warm support + cold query: {len(valid_val_users):,}")
    logger.log(f"[INFO] Test users with warm support + cold query: {len(valid_test_users):,}")
    logger.log(
        f"[INFO] Valid user coverage | val={len(valid_val_users)/max(len(val_user_items),1):.4f} | "
        f"test={len(valid_test_users)/max(len(test_user_items),1):.4f}"
    )

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_user_items": train_user_items,
        "val_user_items": val_user_items,
        "test_user_items": test_user_items,
        "train_users": train_users,
        "valid_val_users": valid_val_users,
        "valid_test_users": valid_test_users,
        "train_item_ids": train_item_ids,
    }


def sample_train_batch(train_users: List[int], train_user_items: Dict[int, List[int]], train_item_ids: np.ndarray):
    users = random.choices(train_users, k=cfg.batch_size)
    pos_items: List[int] = []
    neg_items: List[int] = []
    histories: List[List[int]] = []

    for u in users:
        items = train_user_items[u]
        pos = random.choice(items)
        hist = [i for i in items if i != pos]
        if not hist:
            hist = items[:]
        hist = clip_history(hist, cfg.max_history)

        pos_set = set(items)
        neg = int(_rng.choice(train_item_ids))
        while neg in pos_set:
            neg = int(_rng.choice(train_item_ids))

        pos_items.append(pos)
        neg_items.append(neg)
        histories.append(hist)

    return pos_items, neg_items, histories


def bpr_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
    pos_score = (user_emb * pos_emb).sum(dim=1)
    neg_score = (user_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos_score - neg_score).mean()


# ============================================================
# 图工具 / GNN
# ============================================================
def build_csr(num_nodes: int, row: np.ndarray, col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(row, kind="mergesort")
    row = row[order]
    col = col[order]
    counts = np.bincount(row, minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, col.astype(np.int64)


def load_graph(num_nodes: int, logger: TxtLogger):
    logger.log("[INFO] Loading graph...")
    data = np.load(cfg.edge_path)
    if not {"row", "col"}.issubset(set(data.files)):
        raise ValueError(f"边文件必须包含 row/col，实际: {data.files}")

    row = data["row"].astype(np.int64)
    col = data["col"].astype(np.int64)

    valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes) & (row != col)
    row = row[valid]
    col = col[valid]

    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])

    edge_pairs = np.stack([row_sym, col_sym], axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)
    row_sym, col_sym = edge_pairs[:, 0], edge_pairs[:, 1]

    logger.log(f"[INFO] Graph edges after clean+symmetrize: {len(row_sym):,}")
    return build_csr(num_nodes, row_sym, col_sym)


def sample_neighbors(node: int, indptr: np.ndarray, indices: np.ndarray, fanout: int) -> np.ndarray:
    start, end = indptr[node], indptr[node + 1]
    nbrs = indices[start:end]
    if len(nbrs) == 0:
        return nbrs
    if len(nbrs) <= fanout:
        return nbrs
    pick = _rng.choice(len(nbrs), size=fanout, replace=False)
    return nbrs[pick]


def build_sampled_subgraph(seed_nodes: np.ndarray, indptr: np.ndarray, indices: np.ndarray, fanouts: Tuple[int, ...]):
    seed_nodes = np.unique(seed_nodes.astype(np.int64))
    all_nodes = set(seed_nodes.tolist())
    current = seed_nodes
    sampled_edges = []

    for fanout in fanouts:
        next_nodes = []
        for src in current.tolist():
            nbrs = sample_neighbors(src, indptr, indices, fanout)
            if len(nbrs) == 0:
                continue
            for dst in nbrs.tolist():
                sampled_edges.append((src, dst))
                sampled_edges.append((dst, src))
            next_nodes.extend(nbrs.tolist())
        if not next_nodes:
            break
        next_nodes = np.unique(np.array(next_nodes, dtype=np.int64))
        for n in next_nodes.tolist():
            all_nodes.add(int(n))
        current = next_nodes

    all_nodes = np.array(sorted(all_nodes), dtype=np.int64)
    local_id = {g: i for i, g in enumerate(all_nodes.tolist())}

    if sampled_edges:
        edge_index = np.array(
            [[local_id[s], local_id[d]] for s, d in sampled_edges if s in local_id and d in local_id],
            dtype=np.int64,
        ).T
        edge_index = np.unique(edge_index, axis=1)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return all_nodes, edge_index, local_id


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.normalize(h, p=2, dim=1)


def make_training_subgraph(x_np: np.ndarray, indptr: np.ndarray, indices: np.ndarray, pos_items: List[int], neg_items: List[int], histories: List[List[int]]):
    seed = []
    seed.extend(pos_items)
    seed.extend(neg_items)
    for h in histories:
        seed.extend(h)
    seed = np.array(seed, dtype=np.int64)

    nodes, edge_index_np, local_id = build_sampled_subgraph(seed, indptr, indices, cfg.fanouts)
    x_sub = torch.from_numpy(x_np[nodes]).to(cfg.device)
    edge_sub = torch.from_numpy(edge_index_np).long().to(cfg.device)

    pos_local = torch.tensor([local_id[i] for i in pos_items], dtype=torch.long, device=cfg.device)
    neg_local = torch.tensor([local_id[i] for i in neg_items], dtype=torch.long, device=cfg.device)
    hist_local = [[local_id[i] for i in h] for h in histories]
    return x_sub, edge_sub, pos_local, neg_local, hist_local


def aggregate_user_embedding(item_emb: torch.Tensor, hist_local: List[List[int]]) -> torch.Tensor:
    vecs = []
    for hist in hist_local:
        idx = torch.tensor(hist, dtype=torch.long, device=item_emb.device)
        vecs.append(item_emb[idx].mean(dim=0))
    user_emb = torch.stack(vecs, dim=0)
    return F.normalize(user_emb, p=2, dim=1)


@torch.no_grad()
def evaluate_gnn(model: nn.Module, x_np: np.ndarray, indptr: np.ndarray, indices: np.ndarray, support_user_items: Dict[int, List[int]], target_user_items: Dict[int, List[int]], valid_users: List[int], candidate_mask: np.ndarray):
    if not valid_users:
        return float("nan"), {k: float("nan") for k in cfg.eval_top_k}

    users = valid_users
    if len(users) > cfg.eval_max_users:
        users = _rng.choice(users, size=cfg.eval_max_users, replace=False).tolist()

    candidate_items = np.where(candidate_mask)[0]
    recalls = defaultdict(float)
    auc_labels = []
    auc_scores = []

    model.eval()
    pbar = tqdm(users, desc="Eval-GNN", leave=False)
    for u in pbar:
        support = clip_history(support_user_items[u], cfg.max_history)
        pos = int(_rng.choice(target_user_items[u]))
        neg_pool = candidate_items[candidate_items != pos]
        neg_items = _rng.choice(neg_pool, size=min(cfg.eval_negatives, len(neg_pool)), replace=False)

        cand = np.concatenate([[pos], neg_items]).astype(np.int64)
        seed = np.concatenate([np.array(support, dtype=np.int64), cand])
        nodes, edge_index_np, local_id = build_sampled_subgraph(seed, indptr, indices, cfg.fanouts)

        x_sub = torch.from_numpy(x_np[nodes]).to(cfg.device)
        edge_sub = torch.from_numpy(edge_index_np).long().to(cfg.device)
        sub_emb = model(x_sub, edge_sub)

        support_local = torch.tensor([local_id[i] for i in support], dtype=torch.long, device=cfg.device)
        cand_local = torch.tensor([local_id[i] for i in cand.tolist()], dtype=torch.long, device=cfg.device)

        user_emb = F.normalize(sub_emb[support_local].mean(dim=0, keepdim=True), p=2, dim=1)
        cand_emb = sub_emb[cand_local]
        scores = torch.matmul(cand_emb, user_emb.squeeze(0)).cpu().numpy()

        pos_score = scores[0]
        auc_labels.extend([1] + [0] * (len(scores) - 1))
        auc_scores.extend(scores.tolist())

        rank = int((scores >= pos_score).sum())
        for k in cfg.eval_top_k:
            if rank <= k:
                recalls[k] += 1.0

    auc = roc_auc_score(np.array(auc_labels), np.array(auc_scores))
    for k in cfg.eval_top_k:
        recalls[k] /= len(users)
    return float(auc), dict(recalls)


# ============================================================
# noGNN
# ============================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


@torch.no_grad()
def evaluate_nognn(model: nn.Module, x_np: np.ndarray, support_user_items: Dict[int, List[int]], target_user_items: Dict[int, List[int]], valid_users: List[int], candidate_mask: np.ndarray):
    if not valid_users:
        return float("nan"), {k: float("nan") for k in cfg.eval_top_k}

    users = valid_users
    if len(users) > cfg.eval_max_users:
        users = _rng.choice(users, size=cfg.eval_max_users, replace=False).tolist()

    candidate_items = np.where(candidate_mask)[0]
    recalls = defaultdict(float)
    auc_labels = []
    auc_scores = []

    model.eval()
    pbar = tqdm(users, desc="Eval-noGNN", leave=False)
    for u in pbar:
        support = clip_history(support_user_items[u], cfg.max_history)
        pos = int(_rng.choice(target_user_items[u]))
        neg_pool = candidate_items[candidate_items != pos]
        neg_items = _rng.choice(neg_pool, size=min(cfg.eval_negatives, len(neg_pool)), replace=False)

        support_feat = torch.from_numpy(x_np[np.array(support, dtype=np.int64)]).to(cfg.device)
        cand = np.concatenate([[pos], neg_items]).astype(np.int64)
        cand_feat = torch.from_numpy(x_np[cand]).to(cfg.device)

        support_emb = model(support_feat)
        cand_emb = model(cand_feat)
        user_emb = F.normalize(support_emb.mean(dim=0, keepdim=True), p=2, dim=1)
        scores = torch.matmul(cand_emb, user_emb.squeeze(0)).cpu().numpy()

        pos_score = scores[0]
        auc_labels.extend([1] + [0] * (len(scores) - 1))
        auc_scores.extend(scores.tolist())

        rank = int((scores >= pos_score).sum())
        for k in cfg.eval_top_k:
            if rank <= k:
                recalls[k] += 1.0

    auc = roc_auc_score(np.array(auc_labels), np.array(auc_scores))
    for k in cfg.eval_top_k:
        recalls[k] /= len(users)
    return float(auc), dict(recalls)


# ============================================================
# 特征装载
# ============================================================
def load_feature_matrix(kind: str, logger: TxtLogger) -> np.ndarray:
    if kind == "aligned":
        img_path = cfg.aligned_img_feat_path
        txt_path = cfg.aligned_txt_feat_path
    elif kind == "raw":
        img_path = cfg.raw_img_feat_path
        txt_path = cfg.raw_txt_feat_path
    else:
        raise ValueError(f"未知 feature kind: {kind}")

    logger.log(f"[INFO] Loading feature kind: {kind}")
    logger.log(f"[INFO] image feature file: {img_path}")
    logger.log(f"[INFO] text feature file:  {txt_path}")

    img_feat = np.load(img_path).astype(np.float32)
    txt_feat = np.load(txt_path).astype(np.float32)
    if img_feat.shape[0] != txt_feat.shape[0]:
        raise ValueError(f"图像/文本特征行数不一致: {img_feat.shape[0]} vs {txt_feat.shape[0]}")

    x_np = np.concatenate([img_feat, txt_feat], axis=1).astype(np.float32)
    logger.log(f"[INFO] Feature matrix: {x_np.shape}")
    return x_np


# ============================================================
# 单个实验
# ============================================================
def run_experiment(exp_name: str, feature_kind: str, use_gnn: bool, shared_data: dict) -> dict:
    exp_dir = os.path.join(cfg.save_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = TxtLogger(os.path.join(exp_dir, "train_log.txt"))
    logger.log("=" * 80)
    logger.log(f"Experiment: {exp_name}")
    logger.log(json.dumps({"feature_kind": feature_kind, "use_gnn": use_gnn}, ensure_ascii=False))
    logger.log("=" * 80)

    x_np = load_feature_matrix(feature_kind, logger)
    num_items = x_np.shape[0]

    data = shared_data.get("prepared_data")
    if data is None:
        data = prepare_data(num_items, logger)
        shared_data["prepared_data"] = data
    else:
        logger.log("[INFO] Reusing prepared split/user-history data.")
        logger.log(f"[INFO] Split file used: {cfg.split_path}")

    indptr, indices = None, None
    if use_gnn:
        graph_cache = shared_data.get("graph_cache")
        if graph_cache is None:
            indptr, indices = load_graph(num_items, logger)
            shared_data["graph_cache"] = (indptr, indices)
        else:
            indptr, indices = graph_cache
            logger.log("[INFO] Reusing cleaned graph CSR cache.")

    in_dim = x_np.shape[1]
    if use_gnn:
        model = GraphSAGE(in_dim, cfg.gnn_hidden_dim, cfg.gnn_out_dim, cfg.gnn_dropout).to(cfg.device)
    else:
        model = MLPEncoder(in_dim, cfg.mlp_hidden_dim, cfg.mlp_out_dim, cfg.mlp_dropout).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (cfg.use_amp and cfg.device.type == "cuda") else None

    best_val_r20 = -1.0
    history_rows = []
    best_model_path = os.path.join(exp_dir, "best_model.pth")

    logger.log("[INFO] Start training...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(range(cfg.steps_per_epoch), desc=f"{exp_name} Epoch {epoch:02d}")

        for _ in pbar:
            optimizer.zero_grad(set_to_none=True)
            pos_items, neg_items, histories = sample_train_batch(
                data["train_users"], data["train_user_items"], data["train_item_ids"]
            )

            if use_gnn:
                x_sub, edge_sub, pos_local, neg_local, hist_local = make_training_subgraph(
                    x_np, indptr, indices, pos_items, neg_items, histories
                )
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        sub_emb = model(x_sub, edge_sub)
                        user_emb = aggregate_user_embedding(sub_emb, hist_local)
                        pos_emb = sub_emb[pos_local]
                        neg_emb = sub_emb[neg_local]
                        loss = bpr_loss(user_emb, pos_emb, neg_emb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    sub_emb = model(x_sub, edge_sub)
                    user_emb = aggregate_user_embedding(sub_emb, hist_local)
                    pos_emb = sub_emb[pos_local]
                    neg_emb = sub_emb[neg_local]
                    loss = bpr_loss(user_emb, pos_emb, neg_emb)
                    loss.backward()
                    optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}", nodes=x_sub.size(0), edges=edge_sub.size(1))
            else:
                pos_feat = torch.from_numpy(x_np[np.array(pos_items, dtype=np.int64)]).to(cfg.device)
                neg_feat = torch.from_numpy(x_np[np.array(neg_items, dtype=np.int64)]).to(cfg.device)
                hist_feat_list = [torch.from_numpy(x_np[np.array(h, dtype=np.int64)]).to(cfg.device) for h in histories]

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        pos_emb = model(pos_feat)
                        neg_emb = model(neg_feat)
                        user_vecs = []
                        for hist_feat in hist_feat_list:
                            hist_emb = model(hist_feat)
                            user_vecs.append(hist_emb.mean(dim=0))
                        user_emb = F.normalize(torch.stack(user_vecs, dim=0), p=2, dim=1)
                        loss = bpr_loss(user_emb, pos_emb, neg_emb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pos_emb = model(pos_feat)
                    neg_emb = model(neg_feat)
                    user_vecs = []
                    for hist_feat in hist_feat_list:
                        hist_emb = model(hist_feat)
                        user_vecs.append(hist_emb.mean(dim=0))
                    user_emb = F.normalize(torch.stack(user_vecs, dim=0), p=2, dim=1)
                    loss = bpr_loss(user_emb, pos_emb, neg_emb)
                    loss.backward()
                    optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / cfg.steps_per_epoch

        if use_gnn:
            val_auc, val_recalls = evaluate_gnn(
                model=model,
                x_np=x_np,
                indptr=indptr,
                indices=indices,
                support_user_items=data["train_user_items"],
                target_user_items=data["val_user_items"],
                valid_users=data["valid_val_users"],
                candidate_mask=data["val_mask"],
            )
        else:
            val_auc, val_recalls = evaluate_nognn(
                model=model,
                x_np=x_np,
                support_user_items=data["train_user_items"],
                target_user_items=data["val_user_items"],
                valid_users=data["valid_val_users"],
                candidate_mask=data["val_mask"],
            )

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "val_auc": val_auc,
            **{f"val_r@{k}": val_recalls[k] for k in cfg.eval_top_k},
        }
        history_rows.append(row)

        msg = f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
        for k in cfg.eval_top_k:
            msg += f" | Val R@{k}: {val_recalls[k]:.4f}"
        logger.log(msg)

        if val_recalls[20] > best_val_r20:
            best_val_r20 = val_recalls[20]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_r20": best_val_r20,
                    "config": asdict(cfg),
                    "experiment": exp_name,
                },
                best_model_path,
            )
            logger.log(f"[INFO] Saved best model to: {best_model_path}")

    history_df = pd.DataFrame(history_rows)
    history_csv = os.path.join(exp_dir, "history.csv")
    history_df.to_csv(history_csv, index=False)
    logger.log(f"[INFO] Saved history csv to: {history_csv}")

    ckpt = torch.load(best_model_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.log("[INFO] Final evaluation is performed on cold items only, using warm support histories.")

    if use_gnn:
        test_auc, test_recalls = evaluate_gnn(
            model=model,
            x_np=x_np,
            indptr=indptr,
            indices=indices,
            support_user_items=data["train_user_items"],
            target_user_items=data["test_user_items"],
            valid_users=data["valid_test_users"],
            candidate_mask=data["test_mask"],
        )
    else:
        test_auc, test_recalls = evaluate_nognn(
            model=model,
            x_np=x_np,
            support_user_items=data["train_user_items"],
            target_user_items=data["test_user_items"],
            valid_users=data["valid_test_users"],
            candidate_mask=data["test_mask"],
        )

    logger.log("\n========== Final Test (cold items only) ==========")
    logger.log(f"Test AUC: {test_auc:.4f}")
    for k in cfg.eval_top_k:
        logger.log(f"Test R@{k}: {test_recalls[k]:.4f}")

    result = {
        "experiment": exp_name,
        "feature_kind": feature_kind,
        "use_gnn": use_gnn,
        "best_val_r20": best_val_r20,
        "test_auc": test_auc,
        **{f"test_r@{k}": test_recalls[k] for k in cfg.eval_top_k},
        "log_txt": os.path.join(exp_dir, "train_log.txt"),
        "history_csv": history_csv,
    }
    logger.log("[INFO] Experiment done.")
    logger.close()
    return result


# ============================================================
# 主流程
# ============================================================
def main():
    shared_data: Dict[str, object] = {}
    results = []

    exp_map = {
        "aligned_gnn": ("aligned", True),
        "raw_gnn": ("raw", True),
        "aligned_nognn": ("aligned", False),
        "raw_nognn": ("raw", False),
    }

    for exp_name in cfg.experiments:
        if exp_name not in exp_map:
            raise ValueError(f"未知实验名: {exp_name}")
        feature_kind, use_gnn = exp_map[exp_name]
        result = run_experiment(exp_name, feature_kind, use_gnn, shared_data)
        results.append(result)

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(cfg.save_root, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("All experiments finished.")
    print(summary_df[["experiment", "test_auc", "test_r@10", "test_r@20", "test_r@50"]])
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
