import pandas as pd
import numpy as np
import ujson as json
from tqdm import tqdm
import os


def split_data(input_csv, output_dir, train_ratio=0.8, val_ratio=0.1):
    print("正在加载交互数据...")
    df = pd.read_csv(input_csv)

    # 按用户分组
    user_group = df.groupby('user_id')

    train_dict = {}
    val_dict = {}
    test_dict = {}

    print("开始按用户切分数据...")
    # 使用 tqdm 监控切分进度
    for user, group in tqdm(user_group, desc="Splitting"):
        # 获取该用户的所有交互 item_id
        items = group['item_id'].tolist()

        # 如果交互太少（虽然 5-core 保证了至少 5 个），做个保险
        if len(items) < 3:
            train_dict[str(user)] = items
            continue

        # 打乱顺序
        np.random.shuffle(items)

        n = len(items)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dict[str(user)] = items[:train_end]
        val_dict[str(user)] = items[train_end:val_end]
        test_dict[str(user)] = items[val_end:]

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为 json
    print("正在保存文件...")
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_dict, f)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_dict, f)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_dict, f)

    print(f"✅ 数据切分完成！文件已保存在: {output_dir}")


if __name__ == "__main__":
    # 请确保文件名与你之前生成的 csv 一致
    split_data('elec_5core_interactions.csv', 'data/Electronics')