import os
import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 去掉 BPR 的对照脚本
# 目标：只用内容特征做冷启动推荐，不做任何排序训练
# 形式：
#   user_emb = warm history item embeddings 的均值
#   item_emb = raw / aligned 内容特征（可选 L2 normalize）
#   score   = cosine / dot product
#
# 输出：
#   - 每个实验一个 txt 日志
#   - 每个实验一个 metrics.json
#   - 总表 no_bpr_summary.csv
#
# 推荐对照：
#   aligned_nognn      vs aligned_nobpr
#   raw_nognn          vs raw_nobpr
# ============================================================


@dataclass
class Config:
    raw_img_feat_path: str = "03_image_feat.npy"
    raw_txt_feat_path: str = "02_text_feat.npy"
    aligned_img_feat_path: str = "04_image_feat_aligned_item_coldstart.npy"
    aligned_txt_feat_path: str = "04_text_feat_aligned_item_coldstart.npy"

    interaction_path: str = "01_elec_5core_interactions.csv"
    split_path: str = "04_item_cold_split.npz"

    random_seed: int = 42
    max_history: int = 20
    eval_negatives: int = 99
    eval_top_k: Tuple[int, ...] = (10, 20, 50)
    eval_max_users: int = 5000

    # 跑哪些 no-BPR 实验：raw_nobpr / aligned_nobpr
    experiments: Tuple[str, ...] = ("aligned_nobpr", "raw_nobpr")

    save_root: str = "outputs_no_bpr_item_coldstart"


cfg = Config()
os.makedirs(cfg.save_root, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


set_seed(cfg.random_seed)
_rng = np.random.default_rng(cfg.random_seed)


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
# 数据
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

    valid_val_users = [u for u in val_user_items if len(train_user_items.get(u, [])) >= 1 and len(val_user_items[u]) >= 1]
    valid_test_users = [u for u in test_user_items if len(train_user_items.get(u, [])) >= 1 and len(test_user_items[u]) >= 1]

    logger.log(f"[INFO] Total interactions: {len(interactions):,}")
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
        "valid_val_users": valid_val_users,
        "valid_test_users": valid_test_users,
    }


# ============================================================
# 特征
# ============================================================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


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
    x_np = l2_normalize(x_np)
    logger.log(f"[INFO] Feature matrix: {x_np.shape}")
    return x_np


# ============================================================
# no-BPR 评估
# ============================================================
def evaluate_nobpr(
    x_np: np.ndarray,
    support_user_items: Dict[int, List[int]],
    target_user_items: Dict[int, List[int]],
    valid_users: List[int],
    candidate_mask: np.ndarray,
    logger: TxtLogger,
):
    if not valid_users:
        return float("nan"), {k: float("nan") for k in cfg.eval_top_k}

    users = valid_users
    if len(users) > cfg.eval_max_users:
        users = _rng.choice(users, size=cfg.eval_max_users, replace=False).tolist()

    candidate_items = np.where(candidate_mask)[0]
    recalls = defaultdict(float)
    auc_labels = []
    auc_scores = []

    logger.log(f"[INFO] Evaluation users sampled: {len(users):,}")
    logger.log(f"[INFO] Candidate cold items: {len(candidate_items):,}")

    for u in users:
        support = clip_history(support_user_items[u], cfg.max_history)
        pos = int(_rng.choice(target_user_items[u]))

        neg_pool = candidate_items[candidate_items != pos]
        if len(neg_pool) > cfg.eval_negatives:
            neg_items = _rng.choice(neg_pool, size=cfg.eval_negatives, replace=False)
        else:
            neg_items = neg_pool

        cand = np.concatenate([[pos], neg_items]).astype(np.int64)
        user_emb = l2_normalize(x_np[np.array(support, dtype=np.int64)].mean(axis=0, keepdims=True))[0]
        cand_emb = x_np[cand]
        scores = cand_emb @ user_emb

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
# 单个实验
# ============================================================
def run_experiment(exp_name: str, feature_kind: str, shared_data: dict) -> dict:
    exp_dir = os.path.join(cfg.save_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = TxtLogger(os.path.join(exp_dir, "train_log.txt"))

    logger.log("=" * 80)
    logger.log(f"Experiment: {exp_name}")
    logger.log(json.dumps({"feature_kind": feature_kind, "use_bpr": False, "use_gnn": False}, ensure_ascii=False))
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

    logger.log("[INFO] No training is performed in this script.")
    logger.log("[INFO] user_emb = mean(warm support item embeddings)")
    logger.log("[INFO] item score = cosine similarity between user_emb and candidate cold item embeddings")

    val_auc, val_recalls = evaluate_nobpr(
        x_np=x_np,
        support_user_items=data["train_user_items"],
        target_user_items=data["val_user_items"],
        valid_users=data["valid_val_users"],
        candidate_mask=data["val_mask"],
        logger=logger,
    )
    logger.log("\n========== Validation (cold items only) ==========")
    logger.log(f"Val AUC: {val_auc:.4f}")
    for k in cfg.eval_top_k:
        logger.log(f"Val R@{k}: {val_recalls[k]:.4f}")

    test_auc, test_recalls = evaluate_nobpr(
        x_np=x_np,
        support_user_items=data["train_user_items"],
        target_user_items=data["test_user_items"],
        valid_users=data["valid_test_users"],
        candidate_mask=data["test_mask"],
        logger=logger,
    )
    logger.log("\n========== Final Test (cold items only) ==========")
    logger.log(f"Test AUC: {test_auc:.4f}")
    for k in cfg.eval_top_k:
        logger.log(f"Test R@{k}: {test_recalls[k]:.4f}")

    metrics = {
        "experiment": exp_name,
        "feature_kind": feature_kind,
        "use_bpr": False,
        "use_gnn": False,
        "val_auc": val_auc,
        **{f"val_r@{k}": val_recalls[k] for k in cfg.eval_top_k},
        "test_auc": test_auc,
        **{f"test_r@{k}": test_recalls[k] for k in cfg.eval_top_k},
        "log_txt": os.path.join(exp_dir, "train_log.txt"),
    }

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.log(f"[INFO] Saved metrics json to: {metrics_path}")
    logger.close()
    return metrics


# ============================================================
# 主流程
# ============================================================
def main():
    shared_data: Dict[str, object] = {}
    results = []

    exp_map = {
        "aligned_nobpr": "aligned",
        "raw_nobpr": "raw",
    }

    for exp_name in cfg.experiments:
        if exp_name not in exp_map:
            raise ValueError(f"未知实验名: {exp_name}")
        result = run_experiment(exp_name, exp_map[exp_name], shared_data)
        results.append(result)

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(cfg.save_root, "no_bpr_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("All no-BPR experiments finished.")
    print(summary_df[["experiment", "test_auc", "test_r@10", "test_r@20", "test_r@50"]])
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
