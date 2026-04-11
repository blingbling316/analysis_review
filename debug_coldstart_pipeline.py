import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def stat_line(name: str, value) -> str:
    return f"[INFO] {name}: {value}"


def warn_line(msg: str) -> str:
    return f"[WARN] {msg}"


def fail_line(msg: str) -> str:
    return f"[FAIL] {msg}"


def ok_line(msg: str) -> str:
    return f"[OK] {msg}"


@dataclass
class Config:
    interactions: str
    meta: Optional[str]
    raw_text_feat: Optional[str]
    raw_image_feat: Optional[str]
    aligned_text_feat: Optional[str]
    aligned_image_feat: Optional[str]
    knn_neighbors: Optional[str]
    knn_scores: Optional[str]
    knn_edges: Optional[str]
    val_ratio: float
    random_seed: int
    sample_items: int
    neighbor_topk: int


def load_jsonl_meta(meta_path: str) -> Dict[int, dict]:
    items = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            item_id = obj.get("item_id")
            if item_id is None:
                continue
            items[int(item_id)] = obj
    return items


def describe_feature_matrix(name: str, arr: np.ndarray) -> Tuple[List[str], Dict[str, float]]:
    lines: List[str] = []
    metrics: Dict[str, float] = {}
    lines.append(stat_line(f"{name}.shape", arr.shape))
    lines.append(stat_line(f"{name}.dtype", arr.dtype))

    if arr.ndim != 2:
        lines.append(fail_line(f"{name} 不是二维矩阵，实际 shape={arr.shape}"))
        return lines, metrics

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    zero_rows = int(np.isclose(np.abs(arr).sum(axis=1), 0.0).sum())
    row_norms = np.linalg.norm(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), axis=1)

    metrics["n_rows"] = float(arr.shape[0])
    metrics["n_cols"] = float(arr.shape[1])
    metrics["zero_row_ratio"] = float(zero_rows / max(arr.shape[0], 1))
    metrics["nan_count"] = float(nan_count)
    metrics["inf_count"] = float(inf_count)
    metrics["norm_p50"] = float(np.percentile(row_norms, 50))

    lines.append(stat_line(f"{name}.nan_count", nan_count))
    lines.append(stat_line(f"{name}.inf_count", inf_count))
    lines.append(stat_line(f"{name}.zero_rows", f"{zero_rows}/{arr.shape[0]} ({pct(zero_rows / max(arr.shape[0], 1))})"))
    lines.append(
        stat_line(
            f"{name}.row_norms",
            {
                "min": float(row_norms.min()) if len(row_norms) else None,
                "p10": float(np.percentile(row_norms, 10)) if len(row_norms) else None,
                "p50": float(np.percentile(row_norms, 50)) if len(row_norms) else None,
                "p90": float(np.percentile(row_norms, 90)) if len(row_norms) else None,
                "max": float(row_norms.max()) if len(row_norms) else None,
            },
        )
    )

    if nan_count > 0:
        lines.append(fail_line(f"{name} 含 NaN，先不要继续训练。"))
    if inf_count > 0:
        lines.append(fail_line(f"{name} 含 Inf，先不要继续训练。"))
    if zero_rows > 0:
        lines.append(warn_line(f"{name} 有全零行；如果这是图像特征，通常意味着无图或下载失败。"))
    if arr.shape[0] > 0 and zero_rows / arr.shape[0] > 0.2:
        lines.append(fail_line(f"{name} 全零行比例超过 20%，很可能会明显拖垮建图和训练。"))

    return lines, metrics


def describe_interactions(path: str) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    df = pd.read_csv(path)
    lines: List[str] = []
    metrics: Dict[str, float] = {}

    required = {"user_id", "item_id"}
    missing = required - set(df.columns)
    if missing:
        lines.append(fail_line(f"交互文件缺字段: {sorted(missing)}"))
        return df, lines, metrics

    lines.append(stat_line("interactions.shape", df.shape))
    lines.append(stat_line("n_users", int(df["user_id"].nunique())))
    lines.append(stat_line("n_items", int(df["item_id"].nunique())))
    lines.append(stat_line("item_id.min", int(df["item_id"].min())))
    lines.append(stat_line("item_id.max", int(df["item_id"].max())))
    lines.append(stat_line("duplicate_rows", int(df.duplicated().sum())))

    item_counts = df.groupby("item_id").size().values
    user_counts = df.groupby("user_id").size().values
    lines.append(
        stat_line(
            "item_interaction_count",
            {
                "min": int(item_counts.min()),
                "p10": float(np.percentile(item_counts, 10)),
                "p50": float(np.percentile(item_counts, 50)),
                "p90": float(np.percentile(item_counts, 90)),
                "max": int(item_counts.max()),
            },
        )
    )
    lines.append(
        stat_line(
            "user_interaction_count",
            {
                "min": int(user_counts.min()),
                "p10": float(np.percentile(user_counts, 10)),
                "p50": float(np.percentile(user_counts, 50)),
                "p90": float(np.percentile(user_counts, 90)),
                "max": int(user_counts.max()),
            },
        )
    )

    if df["item_id"].min() != 0:
        lines.append(warn_line("item_id 最小值不是 0，确认后续特征矩阵是否仍按 item_id 行对齐。"))
    if not np.issubdtype(df["item_id"].dtype, np.integer):
        lines.append(warn_line("item_id 不是整数类型，确认是否已完成映射。"))

    metrics["n_users"] = float(df["user_id"].nunique())
    metrics["n_items"] = float(df["item_id"].nunique())
    metrics["item_id_max"] = float(df["item_id"].max())
    return df, lines, metrics


def compare_feature_shapes(df: pd.DataFrame, matrices: Dict[str, np.ndarray]) -> List[str]:
    lines: List[str] = []
    if df.empty:
        return lines

    max_item_id = int(df["item_id"].max())
    for name, arr in matrices.items():
        if arr.ndim != 2:
            continue
        n_rows = arr.shape[0]
        if max_item_id >= n_rows:
            lines.append(fail_line(f"{name} 行数={n_rows}，但交互里最大 item_id={max_item_id}，明显越界。"))
        else:
            lines.append(ok_line(f"{name} 行数覆盖交互中的 item_id 范围。"))

    names = list(matrices.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = matrices[names[i]], matrices[names[j]]
            if a.ndim == 2 and b.ndim == 2:
                if a.shape[0] != b.shape[0]:
                    lines.append(fail_line(f"{names[i]} 与 {names[j]} 行数不一致: {a.shape[0]} vs {b.shape[0]}"))
                else:
                    lines.append(ok_line(f"{names[i]} 与 {names[j]} 行数一致: {a.shape[0]}"))
    return lines


def cosine_neighbors(feat: np.ndarray, item_id: int, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    x = feat.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x = x / norms
    q = x[item_id:item_id + 1]
    scores = (x @ q.T).reshape(-1)
    order = np.argsort(-scores)
    order = order[order != item_id][:topk]
    return order, scores[order]


def print_neighbor_preview(name: str, feat: np.ndarray, meta: Dict[int, dict], sample_ids: List[int], topk: int) -> List[str]:
    lines: List[str] = []
    if feat.ndim != 2 or feat.shape[0] == 0:
        return lines

    for item_id in sample_ids:
        if item_id >= feat.shape[0]:
            continue
        item_title = meta.get(item_id, {}).get("title", "<NO_TITLE>") if meta else "<NO_META>"
        lines.append(f"\n[{name}] item_id={item_id} title={item_title}")
        nbrs, scores = cosine_neighbors(feat, item_id, topk=topk)
        for rank, (nbr, score) in enumerate(zip(nbrs, scores), start=1):
            nbr_title = meta.get(int(nbr), {}).get("title", "<NO_TITLE>") if meta else "<NO_META>"
            lines.append(f"  top{rank}: item_id={int(nbr)} score={float(score):.4f} title={nbr_title}")
    return lines


def load_knn(neighbor_path: Optional[str], score_path: Optional[str], edge_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]]:
    neighbors = np.load(neighbor_path) if neighbor_path and os.path.exists(neighbor_path) else None
    scores = np.load(score_path) if score_path and os.path.exists(score_path) else None
    edges = None
    if edge_path and os.path.exists(edge_path):
        e = np.load(edge_path)
        row = e["row"] if "row" in e.files else None
        col = e["col"] if "col" in e.files else None
        data = e["data"] if "data" in e.files else None
        if row is not None and col is not None:
            edges = (row, col, data)
    return neighbors, scores, edges


def diagnose_knn(num_items: int, neighbors: Optional[np.ndarray], scores: Optional[np.ndarray], edges) -> List[str]:
    lines: List[str] = []

    if neighbors is not None:
        lines.append(stat_line("knn_neighbors.shape", neighbors.shape))
        invalid = int(((neighbors < 0) | (neighbors >= num_items)).sum())
        self_hits = int((neighbors == np.arange(neighbors.shape[0])[:, None]).sum()) if neighbors.ndim == 2 else -1
        lines.append(stat_line("knn_neighbors.invalid_entries", invalid))
        lines.append(stat_line("knn_neighbors.self_hits", self_hits))
        if invalid > 0:
            lines.append(fail_line("邻居矩阵存在非法 item_id。"))
        if self_hits > 0:
            lines.append(warn_line("邻居矩阵仍含自身节点；如果你预期已经去掉自身，这里需要复查。"))

    if scores is not None:
        flat = scores.reshape(-1)
        lines.append(stat_line("knn_scores.shape", scores.shape))
        lines.append(
            stat_line(
                "knn_scores.stats",
                {
                    "min": float(flat.min()),
                    "p10": float(np.percentile(flat, 10)),
                    "p50": float(np.percentile(flat, 50)),
                    "p90": float(np.percentile(flat, 90)),
                    "max": float(flat.max()),
                    "mean": float(flat.mean()),
                },
            )
        )
        low_ratio = float((flat < 0.1).mean())
        lines.append(stat_line("knn_scores.<0.1_ratio", pct(low_ratio)))
        if low_ratio > 0.3:
            lines.append(warn_line("不少边的相似度很低；建议检查 K 是否太大，或特征是否噪声较多。"))

    if edges is not None:
        row, col, data = edges
        lines.append(stat_line("edge_count", len(row)))
        invalid = int(((row < 0) | (row >= num_items) | (col < 0) | (col >= num_items)).sum())
        self_loops = int((row == col).sum())
        lines.append(stat_line("edge_invalid_entries", invalid))
        lines.append(stat_line("edge_self_loops", self_loops))
        if invalid > 0:
            lines.append(fail_line("边列表里存在非法节点。"))
        if self_loops > 0:
            lines.append(warn_line("边列表里存在 self-loop。"))
        if data is not None:
            lines.append(
                stat_line(
                    "edge_weight.stats",
                    {
                        "min": float(data.min()),
                        "p10": float(np.percentile(data, 10)),
                        "p50": float(np.percentile(data, 50)),
                        "p90": float(np.percentile(data, 90)),
                        "max": float(data.max()),
                        "mean": float(data.mean()),
                    },
                )
            )

        # 抽样估算 reciprocal ratio
        sample_n = min(len(row), 200_000)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(row), size=sample_n, replace=False) if len(row) > sample_n else np.arange(len(row))
        r = row[idx].astype(np.int64)
        c = col[idx].astype(np.int64)
        all_codes = row.astype(np.int64) * np.int64(num_items) + col.astype(np.int64)
        code_set = set(all_codes.tolist())
        reciprocal = sum(int((int(c_i) * num_items + int(r_i)) in code_set) for r_i, c_i in zip(r, c))
        reciprocal_ratio = reciprocal / max(sample_n, 1)
        lines.append(stat_line("edge_reciprocal_ratio(sampled)", pct(reciprocal_ratio)))
        if reciprocal_ratio < 0.3:
            lines.append(warn_line("图的双向边比例偏低，后续可考虑 mutual KNN 或对称化。"))

    return lines


def compare_raw_vs_aligned(raw_feat: Optional[np.ndarray], aligned_feat: Optional[np.ndarray], sample_ids: List[int], topk: int, name: str) -> List[str]:
    lines: List[str] = []
    if raw_feat is None or aligned_feat is None:
        return lines
    if raw_feat.shape[0] != aligned_feat.shape[0]:
        lines.append(warn_line(f"{name} raw/aligned 行数不一致，跳过邻居重叠比较。"))
        return lines

    overlaps = []
    for item_id in sample_ids:
        if item_id >= raw_feat.shape[0]:
            continue
        raw_nbrs, _ = cosine_neighbors(raw_feat, item_id, topk)
        ali_nbrs, _ = cosine_neighbors(aligned_feat, item_id, topk)
        overlap = len(set(raw_nbrs.tolist()) & set(ali_nbrs.tolist())) / max(topk, 1)
        overlaps.append(overlap)
    if overlaps:
        lines.append(stat_line(f"{name}_raw_vs_aligned_neighbor_overlap@{topk}", {
            "mean": float(np.mean(overlaps)),
            "min": float(np.min(overlaps)),
            "max": float(np.max(overlaps)),
        }))
        if float(np.mean(overlaps)) < 0.2:
            lines.append(warn_line(f"{name} 对齐前后近邻变化很大；需要人工抽样确认是变好还是变坏。"))
    return lines


def diagnose_val_fallback(interactions: pd.DataFrame, num_items: int, val_ratio: float, random_seed: int) -> List[str]:
    lines: List[str] = []
    users = interactions["user_id"].unique()
    np.random.seed(random_seed)
    train_users = set(np.random.choice(users, size=int((1 - val_ratio) * len(users)), replace=False))
    val_users = set(users) - train_users

    train_user_items = interactions[interactions["user_id"].isin(train_users)].groupby("user_id")["item_id"].apply(list).to_dict()
    val_user_items = interactions[interactions["user_id"].isin(val_users)].groupby("user_id")["item_id"].apply(list).to_dict()
    valid_val_users = [u for u in val_user_items if len(val_user_items[u]) > 0]
    fallback_users = [u for u in valid_val_users if len(train_user_items.get(u, [])) == 0]

    lines.append(stat_line("split.train_users", len(train_users)))
    lines.append(stat_line("split.val_users", len(val_users)))
    lines.append(stat_line("split.valid_val_users", len(valid_val_users)))
    lines.append(stat_line("split.global_mean_fallback_users", f"{len(fallback_users)}/{len(valid_val_users)} ({pct(len(fallback_users) / max(len(valid_val_users), 1))})"))

    if len(fallback_users) > 0:
        lines.append(warn_line("按当前 07 的评估逻辑，这些验证用户会退化成 global_mean 用户向量。"))
    if len(fallback_users) / max(len(valid_val_users), 1) > 0.5:
        lines.append(fail_line("超过一半验证用户会退化成 global_mean；当前评估结果参考价值很弱。"))

    train_items = set(interactions[interactions["user_id"].isin(train_users)]["item_id"].unique())
    val_items = set(interactions[interactions["user_id"].isin(val_users)]["item_id"].unique())
    cold_val_items = val_items - train_items
    lines.append(stat_line("split.train_items", len(train_items)))
    lines.append(stat_line("split.val_items", len(val_items)))
    lines.append(stat_line("split.val_items_not_in_train", f"{len(cold_val_items)}/{len(val_items)} ({pct(len(cold_val_items) / max(len(val_items), 1))})"))

    if max(train_items) >= num_items:
        lines.append(fail_line("训练划分后的 item_id 超出特征矩阵范围。"))
    return lines


def basic_rule_based_summary(feature_metrics: Dict[str, Dict[str, float]]) -> List[str]:
    lines: List[str] = []
    image_key_candidates = [k for k in feature_metrics if "image" in k.lower()]
    for k in image_key_candidates:
        zero_ratio = feature_metrics[k].get("zero_row_ratio", 0.0)
        if zero_ratio > 0.2:
            lines.append(warn_line(f"{k} 全零比例偏高，优先检查图片下载与缺图处理。"))
    text_key_candidates = [k for k in feature_metrics if "text" in k.lower()]
    for k in text_key_candidates:
        if feature_metrics[k].get("norm_p50", 0.0) < 1e-6:
            lines.append(fail_line(f"{k} 中位范数几乎为 0，文本向量可能没有正确生成。"))
    return lines


def choose_sample_item_ids(interactions: pd.DataFrame, n: int, max_id: int) -> List[int]:
    item_ids = interactions["item_id"].drop_duplicates().tolist()
    item_ids = [int(i) for i in item_ids if 0 <= int(i) < max_id]
    if not item_ids:
        return []
    rng = np.random.default_rng(123)
    n = min(n, len(item_ids))
    return sorted(rng.choice(item_ids, size=n, replace=False).tolist())


def setup_loggers(log_path: Optional[str]) -> logging.Logger:
    """写入 log 文件（若给定路径）；同时镜像到控制台。"""
    logger = logging.getLogger("coldstart_pipeline_debug")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")
    if log_path:
        d = os.path.dirname(os.path.abspath(log_path))
        if d:
            os.makedirs(d, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="诊断你的冷启动推荐 pipeline：数据对齐、特征质量、KNN 图、07 的评估退化。")
    parser.add_argument("--interactions", default="01_elec_5core_interactions.csv")
    parser.add_argument("--meta", default="01_elec_5core_meta.jsonl")
    parser.add_argument("--raw_text_feat", default="02_text_feat.npy")
    parser.add_argument("--raw_image_feat", default="03_image_feat.npy")
    parser.add_argument("--aligned_text_feat", default="04_text_feat_aligned_02.npy")
    parser.add_argument("--aligned_image_feat", default="04_image_feat_aligned_02.npy")
    parser.add_argument("--knn_neighbors", default="05_joint_knn_neighbors_02.npy")
    parser.add_argument("--knn_scores", default="05_joint_knn_scores_02.npy")
    parser.add_argument("--knn_edges", default="05_joint_knn_edges_02.npz")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--sample_items", type=int, default=5)
    parser.add_argument("--neighbor_topk", type=int, default=5)
    parser.add_argument(
        "--log-file",
        default="debug_coldstart_pipeline.log",
        help="诊断输出写入的日志文件路径（UTF-8）；设为空字符串则仅打印到控制台。",
    )
    args = parser.parse_args()

    log = setup_loggers(args.log_file or None)

    cfg = Config(
        interactions=args.interactions,
        meta=args.meta if args.meta and os.path.exists(args.meta) else None,
        raw_text_feat=args.raw_text_feat if args.raw_text_feat and os.path.exists(args.raw_text_feat) else None,
        raw_image_feat=args.raw_image_feat if args.raw_image_feat and os.path.exists(args.raw_image_feat) else None,
        aligned_text_feat=args.aligned_text_feat if args.aligned_text_feat and os.path.exists(args.aligned_text_feat) else None,
        aligned_image_feat=args.aligned_image_feat if args.aligned_image_feat and os.path.exists(args.aligned_image_feat) else None,
        knn_neighbors=args.knn_neighbors if args.knn_neighbors and os.path.exists(args.knn_neighbors) else None,
        knn_scores=args.knn_scores if args.knn_scores and os.path.exists(args.knn_scores) else None,
        knn_edges=args.knn_edges if args.knn_edges and os.path.exists(args.knn_edges) else None,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        sample_items=args.sample_items,
        neighbor_topk=args.neighbor_topk,
    )

    log.info("=" * 80)
    log.info("Cold-start Pipeline Diagnostic")
    log.info("=" * 80)

    if not os.path.exists(cfg.interactions):
        raise FileNotFoundError(f"找不到 interactions 文件: {cfg.interactions}")

    interactions, lines, interaction_metrics = describe_interactions(cfg.interactions)
    log.info("\n[SECTION] Interactions")
    for line in lines:
        log.info(line)

    meta = load_jsonl_meta(cfg.meta) if cfg.meta else {}
    if cfg.meta:
        log.info("\n[SECTION] Meta")
        log.info(stat_line("meta_items", len(meta)))

    matrices: Dict[str, np.ndarray] = {}
    feature_metrics: Dict[str, Dict[str, float]] = {}
    for name, path in [
        ("raw_text_feat", cfg.raw_text_feat),
        ("raw_image_feat", cfg.raw_image_feat),
        ("aligned_text_feat", cfg.aligned_text_feat),
        ("aligned_image_feat", cfg.aligned_image_feat),
    ]:
        if path:
            arr = np.load(path)
            matrices[name] = arr
            log.info(f"\n[SECTION] Feature Matrix: {name}")
            desc_lines, metrics = describe_feature_matrix(name, arr)
            feature_metrics[name] = metrics
            for line in desc_lines:
                log.info(line)
        else:
            log.info(f"\n[SECTION] Feature Matrix: {name}")
            log.info(warn_line("文件不存在，跳过。"))

    log.info("\n[SECTION] Alignment Sanity")
    for line in compare_feature_shapes(interactions, matrices):
        log.info(line)

    all_num_items = [arr.shape[0] for arr in matrices.values() if arr.ndim == 2]
    if all_num_items:
        num_items = min(all_num_items)
    else:
        num_items = int(interaction_metrics.get("item_id_max", -1)) + 1

    log.info("\n[SECTION] Rule-based Summary")
    for line in basic_rule_based_summary(feature_metrics):
        log.info(line)

    log.info("\n[SECTION] KNN Graph")
    neighbors, scores, edges = load_knn(cfg.knn_neighbors, cfg.knn_scores, cfg.knn_edges)
    for line in diagnose_knn(num_items=num_items, neighbors=neighbors, scores=scores, edges=edges):
        log.info(line)

    log.info("\n[SECTION] Val Split Fallback Check (mirrors current 07 logic)")
    for line in diagnose_val_fallback(interactions, num_items=num_items, val_ratio=cfg.val_ratio, random_seed=cfg.random_seed):
        log.info(line)

    sample_ids = choose_sample_item_ids(interactions, cfg.sample_items, max_id=num_items)
    log.info("\n[SECTION] Sampled item_ids for neighbor preview")
    log.info(stat_line("sample_item_ids", sample_ids))

    if "raw_text_feat" in matrices:
        log.info("\n[SECTION] Raw Text Neighbors")
        for line in print_neighbor_preview("raw_text_feat", matrices["raw_text_feat"], meta, sample_ids, cfg.neighbor_topk):
            log.info(line)

    if "aligned_text_feat" in matrices:
        log.info("\n[SECTION] Aligned Text Neighbors")
        for line in print_neighbor_preview("aligned_text_feat", matrices["aligned_text_feat"], meta, sample_ids, cfg.neighbor_topk):
            log.info(line)

    if "raw_image_feat" in matrices:
        log.info("\n[SECTION] Raw Image Neighbors")
        for line in print_neighbor_preview("raw_image_feat", matrices["raw_image_feat"], meta, sample_ids, cfg.neighbor_topk):
            log.info(line)

    if "aligned_image_feat" in matrices:
        log.info("\n[SECTION] Aligned Image Neighbors")
        for line in print_neighbor_preview("aligned_image_feat", matrices["aligned_image_feat"], meta, sample_ids, cfg.neighbor_topk):
            log.info(line)

    log.info("\n[SECTION] Raw vs Aligned Neighbor Drift")
    for line in compare_raw_vs_aligned(matrices.get("raw_text_feat"), matrices.get("aligned_text_feat"), sample_ids, cfg.neighbor_topk, "text"):
        log.info(line)
    for line in compare_raw_vs_aligned(matrices.get("raw_image_feat"), matrices.get("aligned_image_feat"), sample_ids, cfg.neighbor_topk, "image"):
        log.info(line)

    log.info("\n[SECTION] Final Hints")
    log.info("1) 如果 raw/aligned 特征大量全零、NaN 或 shape 对不上，先停在特征层排查。")
    log.info("2) 如果 KNN score 很低、reciprocal ratio 很低、人工看近邻不合理，先改建图。")
    log.info("3) 如果 07 的 global_mean fallback 用户占比很高，先别信当前验证指标。")
    log.info("4) 如果上面都正常，但训练效果仍差，再去改 loss、split 和负采样。")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
