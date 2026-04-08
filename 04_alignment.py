import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ========== 配置 ==========
IMAGE_FEAT_FILE = '03_image_feat.npy'
TEXT_FEAT_FILE = '02_text_feat.npy'
EMBED_DIM = 256           # 映射后的公共维度
BATCH_SIZE = 256          # 根据 GPU 显存调整
EPOCHS = 20
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07        # InfoNCE 温度参数
SAVE_MODEL_PATH = '04_cross_modal_alignment.pt'
OUTPUT_IMAGE_FEAT = '04_image_feat_aligned.npy'
OUTPUT_TEXT_FEAT = '04_text_feat_aligned.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== 1. 加载特征 ==========
print("Loading features...")
img_feat = np.load(IMAGE_FEAT_FILE).astype(np.float32)   # (N, 2048)
txt_feat = np.load(TEXT_FEAT_FILE).astype(np.float32)    # (N, 384)
assert img_feat.shape[0] == txt_feat.shape[0]
N = img_feat.shape[0]
print(f"Number of items: {N}")

# ========== 2. 数据集（返回图像特征和文本特征）==========
class AlignDataset(Dataset):
    def __init__(self, img_feat, txt_feat):
        self.img_feat = torch.from_numpy(img_feat)
        self.txt_feat = torch.from_numpy(txt_feat)

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, idx):
        return self.img_feat[idx], self.txt_feat[idx]

dataset = AlignDataset(img_feat, txt_feat)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ========== 3. 映射网络（两个独立的 MLP）==========
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
        return nn.functional.normalize(self.net(x), p=2, dim=1)  # L2 归一化

model_img = ProjectionHead(input_dim=2048).to(device)
model_txt = ProjectionHead(input_dim=384).to(device)

# ========== 4. InfoNCE 损失函数 ==========
def info_nce_loss(img_emb, txt_emb, temperature=TEMPERATURE):
    """
    img_emb: (batch_size, embed_dim)
    txt_emb: (batch_size, embed_dim)
    返回标量损失
    """
    batch_size = img_emb.shape[0]
    # 计算相似度矩阵 (batch_size, batch_size)
    logits = img_emb @ txt_emb.T / temperature
    # 正样本是矩阵的对角线
    labels = torch.arange(batch_size, device=img_emb.device)
    loss_img = nn.functional.cross_entropy(logits, labels)   # 图像 -> 文本
    loss_txt = nn.functional.cross_entropy(logits.T, labels) # 文本 -> 图像
    return (loss_img + loss_txt) / 2

# ========== 5. 训练循环 ==========
optimizer = optim.Adam(list(model_img.parameters()) + list(model_txt.parameters()), lr=LEARNING_RATE)
best_loss = float('inf')

print("Start training...")
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

    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_img': model_img.state_dict(),
            'model_txt': model_txt.state_dict(),
        }, SAVE_MODEL_PATH)
        print(f"Best model saved with loss {best_loss:.4f}")

# ========== 6. 加载最佳模型，提取全部对齐特征 ==========
print("Loading best model and extracting aligned features...")
checkpoint = torch.load(SAVE_MODEL_PATH, map_location='cpu')
model_img.load_state_dict(checkpoint['model_img'])
model_txt.load_state_dict(checkpoint['model_txt'])
model_img.to(device)
model_txt.to(device)
model_img.eval()
model_txt.eval()

# 分批处理全部数据，避免内存溢出
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

# 保存
np.save(OUTPUT_IMAGE_FEAT, aligned_img)
np.save(OUTPUT_TEXT_FEAT, aligned_txt)
print(f"Saved aligned image features: {OUTPUT_IMAGE_FEAT} shape {aligned_img.shape}")
print(f"Saved aligned text features: {OUTPUT_TEXT_FEAT} shape {aligned_txt.shape}")
print("Done.")