import numpy as np
import ujson as json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os


def run_text_feature_extraction():
    # --- 配置区域 ---
    INPUT_META_FILE = '01_elec_5core_meta.jsonl'  # 你之前生成的元数据文件
    OUTPUT_FEAT_FILE = 'text_feat.npy'  # 准备生成的特征文件
    MODEL_NAME = 'all-MiniLM-L6-v2'  # BERT模型：快且准，维度384
    BATCH_SIZE = 128  # 批处理大小，显存大可调至256

    # 1. 环境检查
    print("=" * 30)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到设备: {device.upper(x)}")
    if not os.path.exists(INPUT_META_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_META_FILE}，请确认路径！")
        return

    # 2. 加载预训练模型
    print(f"正在加载 BERT 模型 ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 3. 读取元数据
    items_raw = []
    print("正在加载元数据并同步 ID 顺序...")
    with open(INPUT_META_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取中"):
            try:
                items_raw.append(json.loads(line))
            except:
                continue

    # ⚠️ 关键步骤：按 item_id 升序排列，确保特征矩阵的行号与模型的数字ID对应
    items_raw.sort(key=lambda x: x['item_id'])

    # 4. 文本清洗与拼接
    clean_texts = []
    for item in tqdm(items_raw, desc="文本预处理"):
        title = str(item.get('title', '')).strip()
        # 处理 categories，可能是嵌套列表
        cats = item.get('categories', [])
        if isinstance(cats, list):
            # 展平列表并转为字符串
            cat_str = " ".join([str(c) for c in cats])
        else:
            cat_str = str(cats)

        # 拼接 Title 和 Category 增加语义丰富度
        combined_text = f"{title} {cat_str}".strip()

        # 处理空文本，防止模型报错
        if len(combined_text) < 2:
            combined_text = "electronic product"
        clean_texts.append(combined_text)

    # 5. 执行推理 (Encoding)
    print(f"正在转换文本为向量 (共 {len(clean_texts)} 条)...")
    # 这里会自动调用 GPU (如果存在)
    embeddings = model.encode(
        clean_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 6. 保存结果
    print(f"正在写入磁盘: {OUTPUT_FEAT_FILE}")
    np.save(OUTPUT_FEAT_FILE, embeddings)

    print("=" * 30)
    print(f"✅ 处理完成！")
    print(f"最终特征矩阵形状: {embeddings.shape}")
    print(f"文件位置: {os.path.abspath(OUTPUT_FEAT_FILE)}")
    print("=" * 30)


if __name__ == "__main__":
    run_text_feature_extraction()