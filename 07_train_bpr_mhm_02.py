import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time
from collections import defaultdict
import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class Config:
    img_feat_path = "04_image_feat_aligned_02.npy"
    txt_feat_path = "04_text_feat_aligned_02.npy"
    edge_path = "05_joint_knn_edges_02.npz"
    interaction_path = "01_elec_5core_interactions.csv"
    user_emb_cache = "user_emb_cache.pt"
    val_ratio = 0.2
    random_seed = 42
    in_dim = 512
    hidden_dim = 256
    out_dim = 128
    dropout = 0.1
    epochs = 20
    batch_size = 1024
    steps_per_epoch = 2000
    lr = 0.001
    weight_decay = 1e-5
    use_amp = True
    use_checkpoint = False      # 若OOM仍发生，改为True
    val_sample_size = 5000
    eval_top_k = [10, 20, 50]
    save_dir = "outputs_isolated"
    checkpoint_interval = 2
    resume_from_checkpoint = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
print(f"Device: {cfg.device}, AMP: {cfg.use_amp}, Checkpoint: {cfg.use_checkpoint}")

def save_checkpoint(epoch, model, optimizer, scaler, train_losses, val_aucs, val_recalls, best_auc, path):
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'train_losses': train_losses, 'val_aucs': val_aucs,
        'val_recalls': dict(val_recalls), 'best_auc': best_auc,
    }, path)

def load_checkpoint(model, optimizer, scaler):
    ckpt_files = glob.glob(os.path.join(cfg.save_dir, "checkpoint_epoch_*.pt"))
    if not ckpt_files:
        return 0, [], [], defaultdict(list), 0.0, scaler
    ckpt = torch.load(max(ckpt_files, key=os.path.getmtime), map_location=cfg.device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['epoch'], ckpt['train_losses'], ckpt['val_aucs'], defaultdict(list, ckpt['val_recalls']), ckpt['best_auc'], scaler

# --------------------- 加载数据 ---------------------
print("Loading features...")
img_feat = np.load(cfg.img_feat_path)
txt_feat = np.load(cfg.txt_feat_path)
x = torch.tensor(np.concatenate([img_feat, txt_feat], axis=1), dtype=torch.float32).to(cfg.device)
num_nodes = x.size(0)

print("Loading KNN edges...")
edges = np.load(cfg.edge_path)
row, col = edges["row"], edges["col"]

print("Loading interactions...")
interactions = pd.read_csv(cfg.interaction_path)
users = interactions['user_id'].unique()
np.random.seed(cfg.random_seed)
train_users = set(np.random.choice(users, size=int((1 - cfg.val_ratio) * len(users)), replace=False))
val_users = set(users) - train_users

train_items = set(interactions[interactions['user_id'].isin(train_users)]['item_id'].unique())
train_items = np.array(list(train_items))
train_mask = np.zeros(num_nodes, dtype=bool)
train_mask[train_items] = True

valid_edge = train_mask[row] & train_mask[col]
train_edge_index = torch.tensor([row[valid_edge], col[valid_edge]], dtype=torch.long).to(cfg.device)
data = Data(x=x, edge_index=train_edge_index).to(cfg.device)
print(f"Train nodes: {len(train_items)}, train edges: {train_edge_index.size(1):,}")

train_user_items = interactions[interactions['user_id'].isin(train_users)].groupby('user_id')['item_id'].apply(list).to_dict()
val_user_items = interactions[interactions['user_id'].isin(val_users)].groupby('user_id')['item_id'].apply(list).to_dict()
valid_val_users = [u for u in val_user_items if len(val_user_items[u]) > 0]
print(f"Valid val users: {len(valid_val_users)}")

# --------------------- 模型 ---------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        if cfg.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.conv1, x, edge_index, use_reentrant=False)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if cfg.use_checkpoint and self.training:
            x = checkpoint(self.conv2, x, edge_index, use_reentrant=False)
        else:
            x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout).to(cfg.device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = torch.amp.GradScaler('cuda') if cfg.use_amp else None

def sample_batch(edge_index, num_nodes, batch_size):
    idx = torch.randint(0, edge_index.size(1), (batch_size,), device=edge_index.device)
    return edge_index[0, idx], edge_index[1, idx], torch.randint(0, num_nodes, (batch_size,), device=edge_index.device)

def compute_bpr_loss(emb, u, pos, neg):
    pos_score = (emb[u] * emb[pos]).sum(dim=1)
    neg_score = (emb[u] * emb[neg]).sum(dim=1)
    return -torch.mean(F.logsigmoid((pos_score - neg_score).float()))

# --------------------- 评估函数 ---------------------
@torch.no_grad()
def evaluate(full_emb, user_emb_dict, val_user_items, valid_users, num_nodes, k_list, sample_size):
    if len(valid_users) > sample_size:
        users = np.random.choice(valid_users, sample_size, replace=False)
    else:
        users = valid_users
    pos_items = [np.random.choice(val_user_items[u]) for u in users]
    user_emb = torch.stack([user_emb_dict[u].to(full_emb.device) for u in users])
    pos_emb = full_emb[pos_items]
    neg_items = np.random.randint(0, num_nodes, size=len(users))
    neg_emb = full_emb[neg_items]
    pos_scores = (user_emb * pos_emb).sum(dim=1).cpu().numpy()
    neg_scores = (user_emb * neg_emb).sum(dim=1).cpu().numpy()
    auc = roc_auc_score(np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)]),
                        np.concatenate([pos_scores, neg_scores]))
    recalls = defaultdict(float)
    neg_pool = np.random.randint(0, num_nodes, size=(len(users), 99))
    for i, (u, pos) in enumerate(zip(users, pos_items)):
        neg_cand = neg_pool[i]
        if pos in neg_cand:
            neg_cand = np.setdiff1d(neg_cand, [pos])[:99]
        cand = np.concatenate([[pos], neg_cand])
        scores = torch.matmul(full_emb[cand], user_emb[i]).cpu().numpy()
        rank = (scores >= scores[0]).sum()
        for k in k_list:
            if rank <= k:
                recalls[k] += 1
    for k in k_list:
        recalls[k] /= len(users)
    return auc, recalls

# --------------------- 用户嵌入缓存 ---------------------
cache_path = cfg.user_emb_cache
if os.path.exists(cache_path):
    user_emb_dict = torch.load(cache_path, map_location='cpu')
else:
    print("Computing user embeddings cache for validation users...")
    model.eval()
    with torch.no_grad():
        full_emb = model(data.x, data.edge_index).cpu()
    global_mean = full_emb.mean(dim=0)
    user_emb_dict = {}
    for u in tqdm(valid_val_users, desc="Caching val user emb"):
        items = train_user_items.get(u, [])
        user_emb_dict[u] = full_emb[items].mean(dim=0) if items else global_mean
    torch.save(user_emb_dict, cache_path)

# --------------------- 训练 ---------------------
start_epoch, train_losses, val_aucs, val_recalls, best_auc, scaler = load_checkpoint(model, optimizer, scaler) if cfg.resume_from_checkpoint else (0, [], [], defaultdict(list), 0.0, scaler)
start_time = time.time()

for epoch in range(start_epoch + 1, cfg.epochs + 1):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(range(cfg.steps_per_epoch), desc=f"Epoch {epoch:02d}")
    for _ in pbar:
        optimizer.zero_grad()
        if cfg.use_amp and cfg.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                emb = model(data.x, data.edge_index)
                u, pos, neg = sample_batch(train_edge_index, num_nodes, cfg.batch_size)
                loss = compute_bpr_loss(emb, u, pos, neg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            emb = model(data.x, data.edge_index)
            u, pos, neg = sample_batch(train_edge_index, num_nodes, cfg.batch_size)
            loss = compute_bpr_loss(emb, u, pos, neg)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    avg_loss = epoch_loss / cfg.steps_per_epoch
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        full_emb = model(data.x, data.edge_index).cpu()
    auc, recalls = evaluate(full_emb, user_emb_dict, val_user_items, valid_val_users, num_nodes, cfg.eval_top_k, cfg.val_sample_size)
    val_aucs.append(auc)
    for k in cfg.eval_top_k:
        val_recalls[k].append(recalls[k])
    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | " + " | ".join([f"R@{k}: {recalls[k]:.4f}" for k in cfg.eval_top_k]))

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pth"))
    if epoch % cfg.checkpoint_interval == 0 or epoch == cfg.epochs:
        save_checkpoint(epoch, model, optimizer, scaler, train_losses, val_aucs, val_recalls, best_auc, os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch}.pt"))

print(f"Done in {(time.time()-start_time)/60:.2f} min")