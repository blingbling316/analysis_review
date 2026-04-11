import pandas as pd


def get_non_null_counts(file_path, chunk_size=200000):
    column_counts = None
    total_rows = 0

    # 针对 .jsonl 文件进行分块读取
    reader = pd.read_json(file_path, lines=True, chunksize=chunk_size)

    print(f"Scanning {file_path}...")
    for i, chunk in enumerate(reader):
        # 初始化统计容器
        if column_counts is None:
            column_counts = chunk.notnull().sum()
        else:
            column_counts += chunk.notnull().sum()

        total_rows += len(chunk)
        if (i + 1) % 5 == 0:
            print(f"Processed {total_rows} rows...")

    # 计算百分比
    stats_df = pd.DataFrame({
        'Non-Null Count': column_counts,
        'Null Count': total_rows - column_counts,
        'Completeness (%)': (column_counts / total_rows) * 100
    })

    print("\n--- Final Statistics ---")
    print(stats_df)
    return stats_df

# 使用示例
#meta_stats = get_non_null_counts('Digital_Music.jsonl/meta_Electronics.jsonl')
meta_stats = get_non_null_counts(r'D:\00_CityU-Data Science\SDSC6001\03_Project\Dataset\')