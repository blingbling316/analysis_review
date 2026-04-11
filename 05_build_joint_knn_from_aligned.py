import numpy as np
import faiss
from tqdm import tqdm
import os

# ========== 配置 ==========
IMAGE_ALIGNED_FILE = '04_image_feat_aligned_02.npy'   # 对齐后的图像特征 (N, 256)
TEXT_ALIGNED_FILE  = '04_text_feat_aligned_02.npy'    # 对齐后的文本特征 (N, 256)
OUTPUT_NEIGHBORS = '05_joint_knn_neighbors_02.npy'
OUTPUT_SCORES    = '05_joint_knn_scores_02.npy'
OUTPUT_EDGES     = '05_joint_knn_edges_02.npz'
K = 20                      # 每个节点的邻居数（不含自身）

# ========== 1. 加载对齐特征 ==========
print("加载对齐后的图像特征...")
if not os.path.exists(IMAGE_ALIGNED_FILE):
    raise FileNotFoundError(f"未找到 {IMAGE_ALIGNED_FILE}，请先运行跨模态对齐训练脚本。")
img_feat = np.load(IMAGE_ALIGNED_FILE).astype(np.float32)

print("加载对齐后的文本特征...")
if not os.path.exists(TEXT_ALIGNED_FILE):
    raise FileNotFoundError(f"未找到 {TEXT_ALIGNED_FILE}，请先运行跨模态对齐训练脚本。")
txt_feat = np.load(TEXT_ALIGNED_FILE).astype(np.float32)

assert img_feat.shape[0] == txt_feat.shape[0], "图像和文本特征行数不一致"
N = img_feat.shape[0]
print(f"节点总数: {N}")
print(f"图像特征维度: {img_feat.shape[1]}, 文本特征维度: {txt_feat.shape[1]}")

# ========== 2. 拼接特征 ==========
print("拼接多模态特征...")
joint_feat = np.concatenate([img_feat, txt_feat], axis=1)   # (N, 512)
print(f"联合特征维度: {joint_feat.shape[1]}")

# ========== 3. L2 归一化（使内积等价于余弦相似度）==========
print("归一化联合特征...")
faiss.normalize_L2(joint_feat)

# ========== 4. 构建 FAISS 索引 ==========
print("构建 FAISS 内积索引...")
dim = joint_feat.shape[1]
index = faiss.IndexFlatIP(dim)   # 内积索引（余弦相似度）
index.add(joint_feat)

# ========== 5. 检索每个节点的 K+1 个最近邻（分块 + 进度条）==========
print(f"检索每个节点的 {K+1} 个最近邻...")
CHUNK_SIZE = 10000   # 每批处理的节点数，可根据内存调整
neighbors_list = []
scores_list = []

with tqdm(total=N, desc="FAISS 搜索") as pbar:
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        chunk_vecs = joint_feat[start:end]
        scores_chunk, neighbors_chunk = index.search(chunk_vecs, K + 1)
        neighbors_list.append(neighbors_chunk)
        scores_list.append(scores_chunk)
        pbar.update(end - start)

neighbors = np.concatenate(neighbors_list, axis=0)
scores = np.concatenate(scores_list, axis=0)

# 去掉自身
neighbors = neighbors[:, 1:]
scores = scores[:, 1:]

# ========== 6. 保存结果 ==========
print("保存邻居索引和相似度分数...")
np.save(OUTPUT_NEIGHBORS, neighbors)
np.save(OUTPUT_SCORES, scores)

# 可选：保存为边列表（COO格式）
print("生成边列表...")
rows = np.repeat(np.arange(N), K)
cols = neighbors.flatten()
vals = scores.flatten()
np.savez(OUTPUT_EDGES, row=rows, col=cols, data=vals)

print("=" * 50)
print(f"联合 K‑NN 图构建完成")
print(f"邻居矩阵: {OUTPUT_NEIGHBORS} ({neighbors.shape})")
print(f"分数矩阵: {OUTPUT_SCORES} ({scores.shape})")
print(f"边列表:   {OUTPUT_EDGES}")
print("=" * 50)