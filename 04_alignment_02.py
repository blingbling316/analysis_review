import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ========== 配置 ==========
IMAGE_FEAT_FILE = '03_image_feat.npy'
TEXT_FEAT_FILE = '02_text_feat.npy'
INTERACTIONS_FILE = '01_elec_5core_interactions.csv'   # 原始交互文件
VAL_RATIO = 0.2                                     # 验证用户比例（或按时间划分）
SPLIT_METHOD = 'user'                               # 'user' 按用户划分，'temporal' 按时间划分

EMBED_DIM = 256
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07
SAVE_MODEL_PATH = '04_cross_modal_alignment_02.pt'
OUTPUT_IMAGE_FEAT = '04_image_feat_aligned_02.npy'
OUTPUT_TEXT_FEAT = '04_text_feat_aligned_02.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== 1. 加载特征 ==========
print("Loading features...")
img_feat = np.load(IMAGE_FEAT_FILE).astype(np.float32)
txt_feat = np.load(TEXT_FEAT_FILE).astype(np.float32)
assert img_feat.shape[0] == txt_feat.shape[0]
N = img_feat.shape[0]
print(f"Total items: {N}")

# ========== 2. 读取交互数据，划分训练集用户/物品 ==========
print("Loading interactions and splitting...")
df = pd.read_csv(INTERACTIONS_FILE)
print(f"Total interactions: {len(df)}")

# 按用户划分
if SPLIT_METHOD == 'user':
    users = df['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    split = int(len(users) * (1 - VAL_RATIO))
    train_users = set(users[:split])
    val_users = set(users[split:])
    train_df = df[df['user_id'].isin(train_users)]
    val_df = df[df['user_id'].isin(val_users)]
elif SPLIT_METHOD == 'temporal':
    # 按时间戳排序后划分
    df = df.sort_values('timestamp')
    split = int(len(df) * (1 - VAL_RATIO))
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]
else:
    raise ValueError("SPLIT_METHOD must be 'user' or 'temporal'")

# 提取训练集涉及的所有物品 ID（这些是对齐模型可以见到的物品）
train_items = set(train_df['item_id'].unique())
val_items = set(val_df['item_id'].unique())
print(f"Train users: {len(train_df['user_id'].unique())}, Val users: {len(val_df['user_id'].unique())}")
print(f"Train items: {len(train_items)}, Val items: {len(val_items)}")
print(f"Train items in feature matrix: {len(train_items & set(range(N)))}")

# 生成训练掩码
train_mask = np.zeros(N, dtype=bool)
train_mask[list(train_items)] = True

# ========== 3. 数据集（仅训练物品）==========
class AlignDataset(Dataset):
    def __init__(self, img_feat, txt_feat, mask):
        self.img_feat = torch.from_numpy(img_feat[mask])
        self.txt_feat = torch.from_numpy(txt_feat[mask])

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, idx):
        return self.img_feat[idx], self.txt_feat[idx]

dataset = AlignDataset(img_feat, txt_feat, train_mask)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"Training batches per epoch: {len(dataloader)}")

# ========== 4. 映射网络 ==========
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=1)

model_img = ProjectionHead(input_dim=2048).to(device)
model_txt = ProjectionHead(input_dim=384).to(device)

def info_nce_loss(img_emb, txt_emb, temperature=TEMPERATURE):
    batch_size = img_emb.shape[0]
    logits = img_emb @ txt_emb.T / temperature
    labels = torch.arange(batch_size, device=img_emb.device)
    loss_img = nn.functional.cross_entropy(logits, labels)
    loss_txt = nn.functional.cross_entropy(logits.T, labels)
    return (loss_img + loss_txt) / 2

# ========== 5. 训练循环（仅训练物品）==========
optimizer = optim.Adam(list(model_img.parameters()) + list(model_txt.parameters()), lr=LEARNING_RATE)
best_loss = float('inf')

print("Start training (only on training items)...")
for epoch in range(1, EPOCHS+1):
    model_img.train()
    model_txt.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for img_batch, txt_batch in pbar:
        img_batch = img_batch.to(device)
        txt_batch = txt_batch.to(device)

        optimizer.zero_grad()
        img_emb = model_img(img_batch)
        txt_emb = model_txt(txt_batch)
        loss = info_nce_loss(img_emb, txt_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_img': model_img.state_dict(),
            'model_txt': model_txt.state_dict(),
        }, SAVE_MODEL_PATH)
        print(f"Best model saved with loss {best_loss:.4f}")

# ========== 6. 加载最佳模型，提取全量对齐特征 ==========
print("Loading best model and extracting aligned features for ALL items...")
checkpoint = torch.load(SAVE_MODEL_PATH, map_location='cpu')
model_img.load_state_dict(checkpoint['model_img'])
model_txt.load_state_dict(checkpoint['model_txt'])
model_img.to(device)
model_txt.to(device)
model_img.eval()
model_txt.eval()

batch_size = 1024
aligned_img_list = []
aligned_txt_list = []

with torch.no_grad():
    for i in tqdm(range(0, N, batch_size), desc="Extracting aligned features"):
        img_batch = torch.from_numpy(img_feat[i:i+batch_size]).to(device)
        txt_batch = torch.from_numpy(txt_feat[i:i+batch_size]).to(device)
        aligned_img = model_img(img_batch).cpu().numpy()
        aligned_txt = model_txt(txt_batch).cpu().numpy()
        aligned_img_list.append(aligned_img)
        aligned_txt_list.append(aligned_txt)

aligned_img = np.concatenate(aligned_img_list, axis=0)
aligned_txt = np.concatenate(aligned_txt_list, axis=0)

np.save(OUTPUT_IMAGE_FEAT, aligned_img)
np.save(OUTPUT_TEXT_FEAT, aligned_txt)
print(f"Saved aligned image features: {OUTPUT_IMAGE_FEAT} shape {aligned_img.shape}")
print(f"Saved aligned text features: {OUTPUT_TEXT_FEAT} shape {aligned_txt.shape}")
print("Done.")