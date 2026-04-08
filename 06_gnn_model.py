import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os


# ========== 1. GraphSAGE 模型架构 ==========
class InductiveGraphSAGE(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        """
        纯归纳式 GraphSAGE 模型 (不含任何 ID Embedding)
        :param feature_dim: 输入的多模态特征维度 (如 256 + 256 = 512)
        :param hidden_dim: 隐藏层维度
        :param output_dim: 最终输出的节点表征维度
        """
        super(InductiveGraphSAGE, self).__init__()

        # SAGEConv 聚合的是节点的纯特征，不记忆 ID，非常适合冷启动/Zero-Shot
        self.conv1 = SAGEConv(feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        前向传播
        :param x: 节点特征矩阵 (N, feature_dim)
        :param edge_index: 图的边索引 (2, num_edges)
        :return: 最终节点表征 (N, output_dim)
        """
        # 第一层：聚合邻居特征 -> 激活 -> Dropout 防过拟合
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        # 第二层：输出最终特征
        h = self.conv2(h, edge_index)

        # 推荐系统通常会对最终向量进行 L2 归一化，使得内积等价于余弦相似度
        out = F.normalize(h, p=2, dim=1)

        return out


# ========== 2. 测试与数据组装部分 ==========
if __name__ == "__main__":
    # --- 配置参数 ---
    IMAGE_FEAT_PATH = '04_image_feat_aligned.npy'  # 04 脚本输出的对齐图像特征
    TEXT_FEAT_PATH = '04_text_feat_aligned.npy'  # 04 脚本输出的对齐文本特征
    EDGES_PATH = '05_joint_knn_edges.npz'  # 05 脚本输出的 KNN 边文件

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if os.path.exists(IMAGE_FEAT_PATH) and os.path.exists(TEXT_FEAT_PATH) and os.path.exists(EDGES_PATH):
        print("正在加载对齐特征和图结构...")

        # 1. 加载节点特征并拼接
        img_feat = np.load(IMAGE_FEAT_PATH).astype(np.float32)
        txt_feat = np.load(TEXT_FEAT_PATH).astype(np.float32)

        # (N, 256) 和 (N, 256) 拼接 -> (N, 512)
        x_np = np.concatenate([img_feat, txt_feat], axis=1)
        x = torch.tensor(x_np).to(device)
        print(f"节点特征矩阵大小: {x.shape}")

        # 2. 加载边结构 (带有鲁棒的维度修复逻辑)
        edges_data = np.load(EDGES_PATH)
        # 获取 npz 文件中的第一个数组
        edge_index_np = edges_data[edges_data.files[0]]
        print(f"原始加载的 Numpy 边矩阵大小: {edge_index_np.shape}")

        # 【核心修复逻辑】：确保 edge_index 最终是 (2, num_edges) 的二维矩阵
        if len(edge_index_np.shape) == 2 and edge_index_np.shape[1] == 2:
            # 情况A: 形状是 (num_edges, 2)，需要转置
            edge_index_np = edge_index_np.T
        elif len(edge_index_np.shape) == 1:
            # 情况B: 形状是一维数组，需要 reshape 回去
            edge_index_np = edge_index_np.reshape(2, -1)
        elif len(edge_index_np.shape) == 2 and edge_index_np.shape[0] == x_np.shape[0]:
            # 情况C: 05 脚本保存的是 (N, K) 的邻居矩阵，将其转换为边列表 (Edge List)
            print("检测到输入为邻居矩阵，正在转换为 PyG Edge Index...")
            N, K = edge_index_np.shape
            src = np.repeat(np.arange(N), K)  # 源节点
            dst = edge_index_np.flatten()  # 目标节点
            edge_index_np = np.vstack([src, dst])  # 组合成 (2, N*K)

        print(f"修正后的 PyG 边矩阵大小: {edge_index_np.shape}")

        # 转换为 PyTorch 的 LongTensor
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

        # 3. 实例化模型
        N, feature_dim = x.shape
        hidden_dim = 256
        output_dim = 128  # 最终送到推荐损失函数的维度

        model = InductiveGraphSAGE(feature_dim, hidden_dim, output_dim).to(device)

        # 4. 执行前向传播 (测试提取特征)
        model.eval()  # 测试阶段设为 eval 模式 (关闭 dropout)
        with torch.no_grad():
            item_embeddings = model(x, edge_index)

        print(f"✅ 测试成功！GraphSAGE 成功输出了所有 Item 的高阶隐向量，形状为: {item_embeddings.shape}")

    else:
        print("❌ 错误: 未找到 04 或 05 脚本的输出文件。请确认文件路径或先运行前置步骤。")