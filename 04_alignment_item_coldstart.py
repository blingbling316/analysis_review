import os
import json
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# 1) Split train / val / test by item
# 2) Train cross-modal alignment only on warm (train) items
# 3) Evaluate text-image retrieval on val-cold items
# 4) Export full aligned features and save split for reuse in step 07


@dataclass
class Config:
    # ---------- input ----------
    image_feat_file: str = "03_image_feat.npy"
    text_feat_file: str = "02_text_feat.npy"
    interactions_file: str = "01_elec_5core_interactions.csv"

    # ---------- output ----------
    save_model_path: str = "04_cross_modal_alignment_item_coldstart.pt"
    output_image_feat: str = "04_image_feat_aligned_item_coldstart.npy"
    output_text_feat: str = "04_text_feat_aligned_item_coldstart.npy"
    split_file: str = "04_item_cold_split.npz"
    split_meta_file: str = "04_item_cold_split_meta.json"

    # ---------- item split ----------
    random_seed: int = 42
    val_item_ratio: float = 0.10
    test_item_ratio: float = 0.10
    reuse_existing_split: bool = True

    embed_dim: int = 256
    hidden_dim: int = 512
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3
    temperature: float = 0.07
    weight_decay: float = 1e-5

    # ---------- val ----------
    eval_batch_size: int = 2048
    eval_top_k: Tuple[int, ...] = (1, 5, 10)
    val_eval_max_items: int = 50000
    eval_chunk_size: int = 512

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
print(f"Using device: {cfg.device}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.random_seed)


def create_item_split(num_items: int, val_ratio: float, test_ratio: float, seed: int):
    item_ids = np.arange(num_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(item_ids)

    n_val = int(num_items * val_ratio)
    n_test = int(num_items * test_ratio)
    n_train = num_items - n_val - n_test
    if n_train <= 0:
        raise ValueError("Training item count must be greater than 0")

    train_items = np.sort(item_ids[:n_train])
    val_items = np.sort(item_ids[n_train:n_train + n_val])
    test_items = np.sort(item_ids[n_train + n_val:])

    train_mask = np.zeros(num_items, dtype=bool)
    val_mask = np.zeros(num_items, dtype=bool)
    test_mask = np.zeros(num_items, dtype=bool)
    train_mask[train_items] = True
    val_mask[val_items] = True
    test_mask[test_items] = True

    return train_mask, val_mask, test_mask


def save_split(path_npz: str, path_meta: str, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray, seed: int) -> None:
    np.savez(path_npz, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    meta = {
        "random_seed": seed,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_or_create_split(num_items: int):
    if cfg.reuse_existing_split and os.path.exists(cfg.split_file):
        data = np.load(cfg.split_file)
        required = {"train_mask", "val_mask", "test_mask"}
        if not required.issubset(set(data.files)):
            raise ValueError(f"Split file missing required fields: {required - set(data.files)}")
        train_mask = data["train_mask"].astype(bool)
        val_mask = data["val_mask"].astype(bool)
        test_mask = data["test_mask"].astype(bool)
        if len(train_mask) != num_items or len(val_mask) != num_items or len(test_mask) != num_items:
            raise ValueError("Split file length does not match feature matrix row count")
        print(f"Loaded existing split from: {cfg.split_file}")
        return train_mask, val_mask, test_mask

    train_mask, val_mask, test_mask = create_item_split(
        num_items=num_items,
        val_ratio=cfg.val_item_ratio,
        test_ratio=cfg.test_item_ratio,
        seed=cfg.random_seed,
    )
    save_split(cfg.split_file, cfg.split_meta_file, train_mask, val_mask, test_mask, cfg.random_seed)
    print(f"Saved new split to: {cfg.split_file}")
    return train_mask, val_mask, test_mask


class AlignDataset(Dataset):
    def __init__(self, img_feat: np.ndarray, txt_feat: np.ndarray, item_mask: np.ndarray):
        self.item_ids = np.where(item_mask)[0].astype(np.int64)
        self.img_feat = torch.from_numpy(img_feat[self.item_ids])
        self.txt_feat = torch.from_numpy(txt_feat[self.item_ids])

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx: int):
        return self.img_feat[idx], self.txt_feat[idx], int(self.item_ids[idx])


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


def info_nce_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = img_emb @ txt_emb.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_img = F.cross_entropy(logits, labels)
    loss_txt = F.cross_entropy(logits.T, labels)
    return (loss_img + loss_txt) / 2


# Evaluation: Text-image retrieval Recall@K on val-cold items
@torch.no_grad()
def encode_full(
    model_img: nn.Module,
    model_txt: nn.Module,
    img_feat: np.ndarray,
    txt_feat: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    model_img.eval()
    model_txt.eval()

    aligned_img = []
    aligned_txt = []
    n = img_feat.shape[0]
    for i in tqdm(range(0, n, batch_size), desc="Encoding full features", leave=False):
        img_batch = torch.from_numpy(img_feat[i:i + batch_size]).to(device)
        txt_batch = torch.from_numpy(txt_feat[i:i + batch_size]).to(device)
        aligned_img.append(model_img(img_batch).cpu().numpy())
        aligned_txt.append(model_txt(txt_batch).cpu().numpy())
    return np.concatenate(aligned_img, axis=0), np.concatenate(aligned_txt, axis=0)


@torch.no_grad()
def retrieval_recall_at_k(img_emb: np.ndarray, txt_emb: np.ndarray, item_mask: np.ndarray, k_list: Tuple[int, ...], max_items: int, chunk_size: int):
    item_ids = np.where(item_mask)[0]
    if len(item_ids) == 0:
        return {f"i2t_r@{k}": float("nan") for k in k_list} | {f"t2i_r@{k}": float("nan") for k in k_list}

    if len(item_ids) > max_items:
        rng = np.random.default_rng(cfg.random_seed)
        item_ids = np.sort(rng.choice(item_ids, size=max_items, replace=False))

    img = img_emb[item_ids].astype(np.float32)
    txt = txt_emb[item_ids].astype(np.float32)
    n = len(item_ids)
    max_k = max(k_list)

    metrics: Dict[str, float] = {}

    def chunked_recall(query: np.ndarray, cand: np.ndarray, prefix: str) -> Dict[str, float]:
        hits = {k: 0 for k in k_list}
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            scores = query[start:end] @ cand.T  # (chunk, n)
            local_targets = np.arange(start, end)

            # Use partial argpartition to avoid full sorting and high memory usage
            top_idx_unsorted = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
            top_scores = np.take_along_axis(scores, top_idx_unsorted, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)

            for row_i, target in enumerate(local_targets):
                row_top = top_idx[row_i]
                for k in k_list:
                    if target in row_top[:k]:
                        hits[k] += 1

        return {f"{prefix}_r@{k}": hits[k] / max(n, 1) for k in k_list}

    metrics.update(chunked_recall(img, txt, "i2t"))
    metrics.update(chunked_recall(txt, img, "t2i"))
    return metrics


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("Loading raw features...")
    img_feat = np.load(cfg.image_feat_file).astype(np.float32)
    txt_feat = np.load(cfg.text_feat_file).astype(np.float32)
    if img_feat.shape[0] != txt_feat.shape[0]:
        raise ValueError(f"Image / text feature row count mismatch: {img_feat.shape[0]} vs {txt_feat.shape[0]}")

    num_items = img_feat.shape[0]
    print(f"Total items: {num_items:,}")

    # Interactions are only used for split statistics, not for item splitting
    print("Loading interactions for statistics...")
    interactions = pd.read_csv(cfg.interactions_file)
    print(f"Total interactions: {len(interactions):,}")

    train_mask, val_mask, test_mask = load_or_create_split(num_items)
    print(
        f"Item split | train: {train_mask.sum():,} | val-cold: {val_mask.sum():,} | test-cold: {test_mask.sum():,}"
    )

    train_interactions = interactions[interactions["item_id"].isin(np.where(train_mask)[0])]
    val_interactions = interactions[interactions["item_id"].isin(np.where(val_mask)[0])]
    test_interactions = interactions[interactions["item_id"].isin(np.where(test_mask)[0])]
    print(
        f"Interactions by split | train: {len(train_interactions):,} | val-cold: {len(val_interactions):,} | test-cold: {len(test_interactions):,}"
    )

    train_dataset = AlignDataset(img_feat, txt_feat, train_mask)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(cfg.device.type == "cuda"),
    )
    print(f"Training warm items: {len(train_dataset):,}")
    print(f"Training batches per epoch: {len(train_loader):,}")

    model_img = ProjectionHead(input_dim=img_feat.shape[1], hidden_dim=cfg.hidden_dim, output_dim=cfg.embed_dim).to(cfg.device)
    model_txt = ProjectionHead(input_dim=txt_feat.shape[1], hidden_dim=cfg.hidden_dim, output_dim=cfg.embed_dim).to(cfg.device)
    optimizer = torch.optim.Adam(
        list(model_img.parameters()) + list(model_txt.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_score = -1.0
    history = []

    print("\nStart training item-cold-start alignment...")
    for epoch in range(1, cfg.epochs + 1):
        model_img.train()
        model_txt.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for img_batch, txt_batch, _item_ids in pbar:
            img_batch = img_batch.to(cfg.device, non_blocking=True)
            txt_batch = txt_batch.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            img_emb = model_img(img_batch)
            txt_emb = model_txt(txt_batch)
            loss = info_nce_loss(img_emb, txt_emb, cfg.temperature)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)

        aligned_img, aligned_txt = encode_full(
            model_img=model_img,
            model_txt=model_txt,
            img_feat=img_feat,
            txt_feat=txt_feat,
            batch_size=cfg.eval_batch_size,
            device=cfg.device,
        )
        val_metrics = retrieval_recall_at_k(
            img_emb=aligned_img,
            txt_emb=aligned_txt,
            item_mask=val_mask,
            k_list=cfg.eval_top_k,
            max_items=cfg.val_eval_max_items,
            chunk_size=cfg.eval_chunk_size,
        )

        # Use average Recall@10 on val set for early stopping
        score = (val_metrics.get("i2t_r@10", 0.0) + val_metrics.get("t2i_r@10", 0.0)) / 2.0
        history_row = {"epoch": epoch, "loss": avg_loss, **val_metrics}
        history.append(history_row)

        msg = f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}"
        for k in cfg.eval_top_k:
            msg += f" | i2t R@{k}: {val_metrics[f'i2t_r@{k}']:.4f}"
            msg += f" | t2i R@{k}: {val_metrics[f't2i_r@{k}']:.4f}"
        print(msg)

        if score > best_val_score:
            best_val_score = score
            torch.save(
                {
                    "model_img": model_img.state_dict(),
                    "model_txt": model_txt.state_dict(),
                    "best_val_score": best_val_score,
                    "config": cfg.__dict__,
                },
                cfg.save_model_path,
            )
            print(f"Saved best alignment checkpoint to: {cfg.save_model_path}")

    history_df = pd.DataFrame(history)
    history_path = "04_alignment_item_coldstart_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    # Load best model and export full aligned features
    print("\nLoading best checkpoint and exporting aligned features...")
    checkpoint = torch.load(cfg.save_model_path, map_location="cpu")
    model_img.load_state_dict(checkpoint["model_img"])
    model_txt.load_state_dict(checkpoint["model_txt"])
    model_img.to(cfg.device)
    model_txt.to(cfg.device)

    aligned_img, aligned_txt = encode_full(
        model_img=model_img,
        model_txt=model_txt,
        img_feat=img_feat,
        txt_feat=txt_feat,
        batch_size=cfg.eval_batch_size,
        device=cfg.device,
    )

    np.save(cfg.output_image_feat, aligned_img.astype(np.float32))
    np.save(cfg.output_text_feat, aligned_txt.astype(np.float32))
    print(f"Saved aligned image features: {cfg.output_image_feat} shape {aligned_img.shape}")
    print(f"Saved aligned text features:  {cfg.output_text_feat} shape {aligned_txt.shape}")

    # Final val/test retrieval metrics for verification
    val_metrics = retrieval_recall_at_k(aligned_img, aligned_txt, val_mask, cfg.eval_top_k, cfg.val_eval_max_items, cfg.eval_chunk_size)
    test_metrics = retrieval_recall_at_k(aligned_img, aligned_txt, test_mask, cfg.eval_top_k, cfg.val_eval_max_items, cfg.eval_chunk_size)

    print("\n========== Final Retrieval Metrics ==========")
    print("[VAL-COLD]")
    for k in cfg.eval_top_k:
        print(f"i2t R@{k}: {val_metrics[f'i2t_r@{k}']:.4f} | t2i R@{k}: {val_metrics[f't2i_r@{k}']:.4f}")
    print("[TEST-COLD]")
    for k in cfg.eval_top_k:
        print(f"i2t R@{k}: {test_metrics[f'i2t_r@{k}']:.4f} | t2i R@{k}: {test_metrics[f't2i_r@{k}']:.4f}")

    print("\nDone.")
    print(f"Important: Reuse this split file for step 07: {cfg.split_file}")


if __name__ == "__main__":
    main()