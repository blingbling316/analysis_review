import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import importlib

# ========== 动态导入 06_gnn_model ==========
try:
    gnn_module = importlib.import_module("06_gnn_model")
    InductiveGraphSAGE = gnn_module.InductiveGraphSAGE
    print("成功导入 06_gnn_model.py 中的 GraphSAGE 模型")
except ModuleNotFoundError:
    raise Exception("❌ 找不到 06_gnn_model.py 文件，请确保它与此脚本在同一目录下！")


# ========== 1. 数据集与负采样策略 (Negative Sampling) ==========
class BPRDataset(Dataset):
    def __init__(self, interaction_file, num_items):
        """
        加载交互数据，并为每个正样本构建负采样逻辑
        """
        super(BPRDataset, self).__init__()
        print(f"正在加载交互数据: {interaction_file}")

        # 读取 CSV
        self.data = pd.read_csv(interaction_file, names=['user', 'item', 'rating', 'timestamp'])

        # 【修复逻辑 1】：检查并跳过可能存在的表头 (如果第一行的 user 不是数字)
        first_val = str(self.data['user'].iloc[0])
        if not first_val.isdigit():
            print("检测到 CSV 表头，正在跳过第一行...")
            self.data = self.data.iloc[1:].reset_index(drop=True)

        # 【修复逻辑 2】：强制将列转换为整型(int)和浮点型(float)
        self.users = self.data['user'].astype(np.int64).values
        self.pos_items = self.data['item'].astype(np.int64).values
        self.ratings = self.data['rating'].astype(np.float32).values
        self.num_items = num_items

        # 构建 User 历史交互字典，用于过滤负样本
        print("构建用户历史交互字典以辅助负采样...")
        self.user_history = self.data.groupby('user')['item'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # 【修复逻辑 3】：在取出数据时，确保吐出的是纯净的 Python 数值类型
        user = int(self.users[idx])
        pos_item = int(self.pos_items[idx])
        rating = float(self.ratings[idx])

        # 动态随机负采样
        neg_item = int(np.random.randint(0, self.num_items))
        while neg_item in self.user_history.get(user, set()):
            neg_item = int(np.random.randint(0, self.num_items))

        return user, pos_item, neg_item, rating


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


# ========== 4. 主训练流程 ==========
def train():
    # --- 配置项 ---
    # 请确保将这里的 INTERACTION_FILE 替换为你01脚本实际输出的文件名
    INTERACTION_FILE = '01_elec_5core_interactions.csv'
    IMAGE_FEAT_PATH = '04_image_feat_aligned.npy'
    TEXT_FEAT_PATH = '04_text_feat_aligned.npy'
    EDGES_PATH = '05_joint_knn_edges.npz'

    BATCH_SIZE = 2048
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 256
    OUTPUT_DIM = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    # --- 1. 加载节点多模态特征 ---
    img_feat = np.load(IMAGE_FEAT_PATH).astype(np.float32)
    txt_feat = np.load(TEXT_FEAT_PATH).astype(np.float32)
    x_np = np.concatenate([img_feat, txt_feat], axis=1)  # (N, 512)
    node_features = torch.tensor(x_np).to(device)
    num_items = node_features.shape[0]

    # --- 2. 加载图拓扑结构 (加入与 06 相同的鲁棒修复逻辑) ---
    edges_data = np.load(EDGES_PATH)
    edge_index_np = edges_data[edges_data.files[0]]

    if len(edge_index_np.shape) == 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    elif len(edge_index_np.shape) == 1:
        edge_index_np = edge_index_np.reshape(2, -1)
    elif len(edge_index_np.shape) == 2 and edge_index_np.shape[0] == num_items:
        N_nodes, K_neighbors = edge_index_np.shape
        src = np.repeat(np.arange(N_nodes), K_neighbors)
        dst = edge_index_np.flatten()
        edge_index_np = np.vstack([src, dst])

    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    print(f"图边结构加载完毕，最终形状: {edge_index.shape}")

    # --- 3. 准备 Dataset 和 DataLoader ---
    if not os.path.exists(INTERACTION_FILE):
        raise FileNotFoundError(f"❌ 找不到交互文件 {INTERACTION_FILE}，请检查 01 脚本的输出文件名是否正确。")

    dataset = BPRDataset(INTERACTION_FILE, num_items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Windows 下建议 num_workers 设为 0
    num_users = len(dataset.user_history)
    print(f"统计 -> 总用户: {num_users}, 总物品: {num_items}, 交互数: {len(dataset)}")

    # --- 4. 初始化模型和优化器 ---
    feature_dim = node_features.shape[1]
    gnn_model = InductiveGraphSAGE(feature_dim, HIDDEN_DIM, OUTPUT_DIM).to(device)
    rec_model = MultimodalRecommender(num_users, OUTPUT_DIM).to(device)

    # 联合优化 GNN 参数和 User Embedding 参数
    optimizer = optim.Adam(
        list(gnn_model.parameters()) + list(rec_model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )

    # --- 5. 训练循环 ---
    for epoch in range(EPOCHS):
        gnn_model.train()
        rec_model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for users, pos_items, neg_items, ratings in pbar:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            # Step 1: 全局 GNN 消息传递，获取最新的全体 Item 隐向量
            all_item_embs = gnn_model(node_features, edge_index)

            # Step 2: 获取正负样本打分
            pos_scores, neg_scores = rec_model(users, pos_items, neg_items, all_item_embs)

            # Step 3: 计算加权 BPR 损失
            loss = weighted_bpr_loss(pos_scores, neg_scores, ratings)

            # Step 4: 反向传播与梯度更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 完成. 平均 BPR Loss: {avg_loss:.4f}")

    # --- 6. 保存模型 ---
    torch.save({
        'gnn_state_dict': gnn_model.state_dict(),
        'rec_state_dict': rec_model.state_dict(),
    }, 'multimodal_recommender_final.pth')
    print("✅ 模型训练完成并已保存为 multimodal_recommender_final.pth")


if __name__ == "__main__":
    train()