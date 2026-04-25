import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


# step3: Fusion Graph + Residual Graph Enhancement
# 1) Graph structure = Content-based KNN graph + Behavior co-occurrence graph
# 2) Mini-batch subgraph training is still used
# 3) Hard negative sampling is applied
# 4) Item representation = alpha * content_proj + (1-alpha) * gnn_emb

@dataclass
class Config:

    img_feat_path: str = '04_image_feat_aligned_item_coldstart.npy'
    txt_feat_path: str = '04_text_feat_aligned_item_coldstart.npy'
    edge_path: str = '05_joint_knn_edges_item_coldstart.npz'
    interaction_path: str = '01_elec_5core_interactions.csv'

    split_path: str = '04_item_cold_split.npz'
    reuse_existing_split: bool = True
    random_seed: int = 42
    val_item_ratio: float = 0.10
    test_item_ratio: float = 0.10

    in_dim: int = 512
    hidden_dim: int = 256
    out_dim: int = 512
    dropout: float = 0.1
    residual_alpha: float = 0.7

    epochs: int = 5
    batch_size: int = 128
    steps_per_epoch: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_amp: bool = True
    hard_negative_ratio: float = 0.30

    fanouts: Tuple[int, int] = (15, 10)
    max_history: int = 20


    eval_top_k: Tuple[int, ...] = (10, 20, 50)
    eval_negatives: int = 99
    eval_max_users: int = 500

    save_dir: str = 'outputs_item_coldstart_step3_residual'
    best_model_name: str = 'best_model_item_coldstart_step3_residual.pth'

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
print(f'Device: {cfg.device}, AMP: {cfg.use_amp}')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.random_seed)
_rng = np.random.default_rng(cfg.random_seed)


# CSR
def build_csr(num_nodes: int, row: np.ndarray, col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(row, kind='mergesort')
    row = row[order]
    col = col[order]
    counts = np.bincount(row, minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, col.astype(np.int64)


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



def load_features() -> np.ndarray:
    print('[INFO] Loading aligned features...')
    print(f'[INFO] image feature file: {cfg.img_feat_path}')
    print(f'[INFO] text feature file:  {cfg.txt_feat_path}')

    img_feat = np.load(cfg.img_feat_path).astype(np.float32)
    txt_feat = np.load(cfg.txt_feat_path).astype(np.float32)
    if img_feat.shape[0] != txt_feat.shape[0]:
        raise ValueError(f"Image / text feature row count mismatch: {img_feat.shape[0]} vs {txt_feat.shape[0]}")

    x_np = np.concatenate([img_feat, txt_feat], axis=1).astype(np.float32)
    print(f'[INFO] Feature matrix: {x_np.shape}')
    return x_np


def load_graph(num_nodes: int):
    print('[INFO] Loading graph...')
    data = np.load(cfg.edge_path)
    if not {'row', 'col'}.issubset(set(data.files)):
        raise ValueError(f"Edge file must contain 'row' and 'col' fields, found: {data.files}")

    row = data['row'].astype(np.int64)
    col = data['col'].astype(np.int64)

    valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes) & (row != col)
    row = row[valid]
    col = col[valid]

    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    edge_pairs = np.stack([row_sym, col_sym], axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)
    row_sym, col_sym = edge_pairs[:, 0], edge_pairs[:, 1]

    print(f'[INFO] Graph edges after clean+symmetrize: {len(row_sym):,}')
    return build_csr(num_nodes, row_sym, col_sym)


def create_item_split(num_items: int):
    item_ids = np.arange(num_items)
    _rng.shuffle(item_ids)

    n_val = int(num_items * cfg.val_item_ratio)
    n_test = int(num_items * cfg.test_item_ratio)
    n_train = num_items - n_val - n_test
    if n_train <= 0:
        raise ValueError('The number of training items must be greater than 0')

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


def load_or_create_split(num_items: int):
    if cfg.reuse_existing_split and os.path.exists(cfg.split_path):
        data = np.load(cfg.split_path)
        required = {'train_mask', 'val_mask', 'test_mask'}
        if not required.issubset(set(data.files)):
            raise ValueError(f"Split file missing required fields, found fields: {data.files}")

        train_mask = data['train_mask'].astype(bool)
        val_mask = data['val_mask'].astype(bool)
        test_mask = data['test_mask'].astype(bool)
        if len(train_mask) != num_items:
            raise ValueError(f"Split length does not match feature rows: split={len(train_mask)}, num_items={num_items}")
        print(f'[INFO] Loaded split from: {cfg.split_path}')
    else:
        train_mask, val_mask, test_mask = create_item_split(num_items)
        np.savez(cfg.split_path, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        print(f'[INFO] Created new split and saved to: {cfg.split_path}')

    print(f'[INFO] Item split | train={train_mask.sum():,} | val-cold={val_mask.sum():,} | test-cold={test_mask.sum():,}')
    overlap_tv = int((train_mask & val_mask).sum())
    overlap_tt = int((train_mask & test_mask).sum())
    overlap_vt = int((val_mask & test_mask).sum())
    print(f'[INFO] Split overlap check | train&val={overlap_tv} | train&test={overlap_tt} | val&test={overlap_vt}')
    return train_mask, val_mask, test_mask


def build_user_histories(df: pd.DataFrame, item_mask: np.ndarray) -> Dict[int, List[int]]:
    sub = df[df['item_id'].isin(np.where(item_mask)[0])]
    return sub.groupby('user_id')['item_id'].apply(list).to_dict()


def prepare_data(num_items: int):
    interactions = pd.read_csv(cfg.interaction_path)
    train_mask, val_mask, test_mask = load_or_create_split(num_items)

    train_user_items = build_user_histories(interactions, train_mask)
    val_user_items = build_user_histories(interactions, val_mask)
    test_user_items = build_user_histories(interactions, test_mask)

    train_users = [u for u, items in train_user_items.items() if len(items) >= 2]
    valid_val_users = [u for u in val_user_items if len(train_user_items.get(u, [])) >= 1 and len(val_user_items[u]) >= 1]
    valid_test_users = [u for u in test_user_items if len(train_user_items.get(u, [])) >= 1 and len(test_user_items[u]) >= 1]
    train_item_ids = np.where(train_mask)[0]

    print(f'[INFO] Total interactions: {len(interactions):,}')
    train_interactions = interactions[interactions['item_id'].isin(np.where(train_mask)[0])]
    val_interactions = interactions[interactions['item_id'].isin(np.where(val_mask)[0])]
    test_interactions = interactions[interactions['item_id'].isin(np.where(test_mask)[0])]
    print(f'[INFO] Interactions by split | train={len(train_interactions):,} | val-cold={len(val_interactions):,} | test-cold={len(test_interactions):,}')
    print(f'[INFO] Train users usable for BPR: {len(train_users):,}')
    print(f'[INFO] Val users with warm support + cold query: {len(valid_val_users):,}')
    print(f'[INFO] Test users with warm support + cold query: {len(valid_test_users):,}')
    print(f'[INFO] Valid user coverage | val={len(valid_val_users)/max(len(val_user_items),1):.4f} | test={len(valid_test_users)/max(len(test_user_items),1):.4f}')

    return {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_user_items': train_user_items,
        'val_user_items': val_user_items,
        'test_user_items': test_user_items,
        'train_users': train_users,
        'valid_val_users': valid_val_users,
        'valid_test_users': valid_test_users,
        'train_item_ids': train_item_ids,
    }


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


def clip_history(hist: List[int], max_len: int) -> List[int]:
    if len(hist) <= max_len:
        return hist
    return hist[-max_len:]


def get_node_neighbors(node: int, indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    start, end = indptr[node], indptr[node + 1]
    return indices[start:end]


def sample_negative_item(pos_item: int, user_pos: set, train_item_ids: np.ndarray, indptr: np.ndarray, indices: np.ndarray) -> int:
    use_hard_negative = random.random() < cfg.hard_negative_ratio
    if use_hard_negative:
        nbrs = get_node_neighbors(pos_item, indptr, indices)
        if len(nbrs) > 0:
            hard_candidates = [int(n) for n in nbrs.tolist() if int(n) not in user_pos]
            if hard_candidates:
                return int(random.choice(hard_candidates))

    neg = int(_rng.choice(train_item_ids))
    while neg in user_pos:
        neg = int(_rng.choice(train_item_ids))
    return neg


def sample_train_batch(train_users: List[int], train_user_items: Dict[int, List[int]], train_item_ids: np.ndarray, indptr: np.ndarray, indices: np.ndarray):
    users = random.choices(train_users, k=cfg.batch_size)
    pos_items = []
    neg_items = []
    histories = []

    for u in users:
        items = train_user_items[u]
        pos = random.choice(items)
        hist = [i for i in items if i != pos]
        if not hist:
            hist = items[:]
        hist = clip_history(hist, cfg.max_history)

        user_pos = set(items)
        neg = sample_negative_item(pos, user_pos, train_item_ids, indptr, indices)

        pos_items.append(pos)
        neg_items.append(neg)
        histories.append(hist)

    return users, pos_items, neg_items, histories


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


def fuse_item_embedding(x_sub: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
    content_emb = F.normalize(x_sub, p=2, dim=1)
    fused = cfg.residual_alpha * content_emb + (1.0 - cfg.residual_alpha) * gnn_emb
    return F.normalize(fused, p=2, dim=1)


def aggregate_user_embedding(item_emb: torch.Tensor, hist_local: List[List[int]]) -> torch.Tensor:
    vecs = []
    for hist in hist_local:
        idx = torch.tensor(hist, dtype=torch.long, device=item_emb.device)
        vecs.append(item_emb[idx].mean(dim=0))
    user_emb = torch.stack(vecs, dim=0)
    return F.normalize(user_emb, p=2, dim=1)


def bpr_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
    pos_score = (user_emb * pos_emb).sum(dim=1)
    neg_score = (user_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos_score - neg_score).mean()

# Evaluation: Per-user Subgraph

@torch.no_grad()
def evaluate(model: nn.Module, x_np: np.ndarray, indptr: np.ndarray, indices: np.ndarray, support_user_items: Dict[int, List[int]], target_user_items: Dict[int, List[int]], valid_users: List[int], candidate_mask: np.ndarray):
    if not valid_users:
        return float('nan'), {k: float('nan') for k in cfg.eval_top_k}

    users = valid_users
    if len(users) > cfg.eval_max_users:
        users = _rng.choice(users, size=cfg.eval_max_users, replace=False).tolist()

    candidate_items = np.where(candidate_mask)[0]
    recalls = defaultdict(float)
    auc_labels = []
    auc_scores = []

    model.eval()
    pbar = tqdm(users, desc='Eval', leave=False)
    for u in pbar:
        support = clip_history(support_user_items[u], cfg.max_history)
        pos = int(_rng.choice(target_user_items[u]))

        neg_pool = candidate_items[candidate_items != pos]
        if len(neg_pool) > cfg.eval_negatives:
            neg_items = _rng.choice(neg_pool, size=cfg.eval_negatives, replace=False)
        else:
            neg_items = neg_pool

        cand = np.concatenate([[pos], neg_items]).astype(np.int64)
        seed = np.concatenate([np.array(support, dtype=np.int64), cand])
        nodes, edge_index_np, local_id = build_sampled_subgraph(seed, indptr, indices, cfg.fanouts)

        x_sub = torch.from_numpy(x_np[nodes]).to(cfg.device)
        edge_sub = torch.from_numpy(edge_index_np).long().to(cfg.device)
        gnn_emb = model(x_sub, edge_sub)
        sub_emb = fuse_item_embedding(x_sub, gnn_emb)

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


# main

def main():
    x_np = load_features()
    num_items = x_np.shape[0]
    if x_np.shape[1] != cfg.in_dim:
        raise ValueError(f"in_dim does not match feature dimension: cfg.in_dim={cfg.in_dim}, feature_dim={x_np.shape[1]}")
    indptr, indices = load_graph(num_items)
    data = prepare_data(num_items)

    print('[INFO] step3 uses fused graph: content KNN + co-occurrence graph.')
    print(f'[INFO] split file used by 07: {cfg.split_path}')
    print(f'[INFO] hard negative ratio: {cfg.hard_negative_ratio:.2f}')
    print(f'[INFO] residual alpha: {cfg.residual_alpha:.2f}')

    model = GraphSAGE(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if (cfg.use_amp and cfg.device.type == 'cuda') else None

    best_val_r20 = -1.0
    history = []

    print('\nStart training minibatch item-cold-start recommender...')
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(range(cfg.steps_per_epoch), desc=f'Epoch {epoch:02d}')

        for _ in pbar:
            optimizer.zero_grad(set_to_none=True)
            _, pos_items, neg_items, histories = sample_train_batch(
                data['train_users'], data['train_user_items'], data['train_item_ids'], indptr, indices
            )
            x_sub, edge_sub, pos_local, neg_local, hist_local = make_training_subgraph(
                x_np, indptr, indices, pos_items, neg_items, histories
            )

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    gnn_emb = model(x_sub, edge_sub)
                    sub_emb = fuse_item_embedding(x_sub, gnn_emb)
                    user_emb = aggregate_user_embedding(sub_emb, hist_local)
                    pos_emb = sub_emb[pos_local]
                    neg_emb = sub_emb[neg_local]
                    loss = bpr_loss(user_emb, pos_emb, neg_emb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                gnn_emb = model(x_sub, edge_sub)
                sub_emb = fuse_item_embedding(x_sub, gnn_emb)
                user_emb = aggregate_user_embedding(sub_emb, hist_local)
                pos_emb = sub_emb[pos_local]
                neg_emb = sub_emb[neg_local]
                loss = bpr_loss(user_emb, pos_emb, neg_emb)
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}', nodes=x_sub.size(0), edges=edge_sub.size(1))

        avg_loss = epoch_loss / cfg.steps_per_epoch
        val_auc, val_recalls = evaluate(
            model=model,
            x_np=x_np,
            indptr=indptr,
            indices=indices,
            support_user_items=data['train_user_items'],
            target_user_items=data['val_user_items'],
            valid_users=data['valid_val_users'],
            candidate_mask=data['val_mask'],
        )

        row = {'epoch': epoch, 'loss': avg_loss, 'val_auc': val_auc, **{f'val_r@{k}': val_recalls[k] for k in cfg.eval_top_k}}
        history.append(row)

        msg = f'Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}'
        for k in cfg.eval_top_k:
            msg += f' | Val R@{k}: {val_recalls[k]:.4f}'
        print(msg)

        if val_recalls[20] > best_val_r20:
            best_val_r20 = val_recalls[20]
            best_path = os.path.join(cfg.save_dir, cfg.best_model_name)
            torch.save({'model_state_dict': model.state_dict(), 'best_val_r20': best_val_r20, 'config': cfg.__dict__}, best_path)
            print(f'Saved best model to: {best_path}')

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(cfg.save_dir, 'train_history_item_coldstart_step3_residual.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f'History saved to: {hist_path}')

    best_path = os.path.join(cfg.save_dir, cfg.best_model_name)
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt['model_state_dict'])

    print('[INFO] Final evaluation is performed on cold items only, using warm support histories.')
    test_auc, test_recalls = evaluate(
        model=model,
        x_np=x_np,
        indptr=indptr,
        indices=indices,
        support_user_items=data['train_user_items'],
        target_user_items=data['test_user_items'],
        valid_users=data['valid_test_users'],
        candidate_mask=data['test_mask'],
    )

    print('\n========== Final Test (cold items only) ==========')
    print(f'Test AUC: {test_auc:.4f}')
    for k in cfg.eval_top_k:
        print(f'Test R@{k}: {test_recalls[k]:.4f}')


if __name__ == '__main__':
    main()
