import pandas as pd
import ujson as json
from collections import Counter
import os


def fast_preprocess(review_path, meta_path, output_prefix, threshold=5):
    # Phase 1: Count frequencies
    print("Phase 1: Counting IDs...")
    user_counts = Counter()
    item_counts = Counter()
    print("Starting to read data...")
    reader = pd.read_json(review_path, lines=True, chunksize=100000,dtype={'user_id': str, 'parent_asin': str})

    print("Starting counting...")
    for chunk in reader:
        user_counts.update(chunk['user_id'].tolist())
        item_counts.update(chunk['parent_asin'].tolist())
    print("Starting filtering...")
    # Filter ID sets that meet the 5-core requirement
    valid_users = {k for k, v in user_counts.items() if v >= threshold}
    valid_items = {k for k, v in item_counts.items() if v >= threshold}
    del user_counts, item_counts  # Free large objects
    print(f"Found {len(valid_users)} users and {len(valid_items)} items meeting {threshold}-core.")

    # Phase 2: Map IDs and write to disk in streaming mode
    print("Phase 2: Mapping IDs and Filtering...")
    user_map = {}
    item_map = {}
    u_idx, i_idx = 0, 0

    print("Creating interaction file and writing line by line...")
    with open(f"{output_prefix}_interactions.csv", 'w') as out_f:
        out_f.write("user_id,item_id,rating,timestamp\n")

        reader = pd.read_json(review_path, lines=True, chunksize=500000)
        for chunk in reader:
            # Filter rows that don't meet the 5-core requirement
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

    print("Processing metadata...")
    # Phase 3: Process metadata
    print("Phase 3: Extracting Meta...")
    with open(f"{output_prefix}_meta.jsonl", 'w') as out_m:
        with open(meta_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                asin = item.get('parent_asin')
                if asin in item_map:
                    clean_item = {
                        'item_id': item_map[asin],
                        'title': item.get('title', ''),
                        'images': item.get('images', []),
                        'categories': item.get('categories', [])
                    }
                    out_m.write(json.dumps(clean_item) + '\n')

    print("Preprocess finished successfully!")
    return item_map

item_map = fast_preprocess('Digital_Music.jsonl/Electronics.jsonl', 'Digital_Music.jsonl/meta_Electronics.jsonl', 'elec_5core')