
import numpy as np

# 1. 加载你生成的那个特征文件
file_path = 'text_feat.npy'

try:
    feat = np.load(file_path)
    # 2. 打印 shape
    print("=" * 30)
    print(f"矩阵的形状 (embeddings.shape): {feat.shape}")
    print(f"物品总数 (n_items): {feat.shape[0]}")
    print(f"向量维度 (dim): {feat.shape[1]}")
    print("=" * 30)
except FileNotFoundError:
    print(f"❌ 找不到文件 {file_path}，请确认文件是否生成成功。")