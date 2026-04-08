import pandas as pd
import ujson as json
from collections import Counter
import os


def fast_preprocess(review_path, meta_path, output_prefix, threshold=5):
    # --- 阶段 1: 统计频率 (流式扫描，内存开销极低) ---
    print("Phase 1: Counting IDs...")
    user_counts = Counter()
    item_counts = Counter()
    print("开始读数……")
    # 仅读取必要的列，进一步降低开销
    reader = pd.read_json(review_path, lines=True, chunksize=100000,dtype={'user_id': str, 'parent_asin': str})

    print("开始计数……")
    for chunk in reader:
        user_counts.update(chunk['user_id'].tolist())
        item_counts.update(chunk['parent_asin'].tolist())
    print("开始筛选……")
    # 筛选满足 5-core 的 ID 集合
    valid_users = {k for k, v in user_counts.items() if v >= threshold}
    valid_items = {k for k, v in item_counts.items() if v >= threshold}
    del user_counts, item_counts  # 释放大对象
    print(f"Found {len(valid_users)} users and {len(valid_items)} items meeting {threshold}-core.")

    # --- 阶段 2: 映射 ID 并流式写入磁盘 ---
    print("Phase 2: Mapping IDs and Filtering...")
    user_map = {}
    item_map = {}
    u_idx, i_idx = 0, 0

    print("创建交互文件并逐行写入……")
    with open(f"{output_prefix}_interactions.csv", 'w') as out_f:
        out_f.write("user_id,item_id,rating,timestamp\n")

        reader = pd.read_json(review_path, lines=True, chunksize=500000)
        for chunk in reader:
            # 过滤不符合 5-core 的行
            mask = chunk['user_id'].isin(valid_users) & chunk['parent_asin'].isin(valid_items)
            filtered_chunk = chunk[mask]

            for _, row in filtered_chunk.iterrows():
                uid, iid = row['user_id'], row['parent_asin']
                if uid not in user_map:
                    user_map[uid] = u_idx
                    u_idx += 1
                if iid not in item_map:
                    item_map[iid] = i_idx
                    i_idx += 1

                out_f.write(f"{user_map[uid]},{item_map[iid]},{row['rating']},{row['timestamp']}\n")

    print("处理元数据……")
    # --- 阶段 3: 处理元数据 (仅针对入选的 item) ---
    print("Phase 3: Extracting Meta...")
    with open(f"{output_prefix}_meta.jsonl", 'w') as out_m:
        with open(meta_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                asin = item.get('parent_asin')  # 注意2023版字段名
                if asin in item_map:
                    # 仅保留需要的字段以减小体积
                    clean_item = {
                        'item_id': item_map[asin],
                        'title': item.get('title', ''),
                        'images': item.get('images', []),  # 为后续图像提取留存
                        'categories': item.get('categories', [])
                    }
                    out_m.write(json.dumps(clean_item) + '\n')

    print("Preprocess finished successfully!")
    return item_map

# 运行 (确保文件名路径正确)
item_map = fast_preprocess('Digital_Music.jsonl/Electronics.jsonl', 'Digital_Music.jsonl/meta_Electronics.jsonl', 'elec_5core')