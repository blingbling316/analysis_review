import os
from collections import defaultdict
from itertools import combinations

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== Configuration ==========
IMAGE_ALIGNED_FILE = '04_image_feat_aligned_item_coldstart.npy'
TEXT_ALIGNED_FILE = '04_text_feat_aligned_item_coldstart.npy'
INTERACTION_FILE = '01_elec_5core_interactions.csv'

OUTPUT_NEIGHBORS = '05_joint_knn_neighbors_item_coldstart.npy'
OUTPUT_SCORES = '05_joint_knn_scores_item_coldstart.npy'
OUTPUT_EDGES = '05_joint_knn_edges_item_coldstart.npz'

K_CONTENT = 40          # KNN neighbors per node for content graph (excluding self)
K_COOC = 20             # Max co-occurrence neighbors per node
IMAGE_WEIGHT = 0.30
TEXT_WEIGHT = 0.70
CHUNK_SIZE = 10000
GPU_ID = 0              # GPU device ID


# 1. Load Aligned Features
print('Loading aligned image features...')
if not os.path.exists(IMAGE_ALIGNED_FILE):
    raise FileNotFoundError(f'File not found: {IMAGE_ALIGNED_FILE}, please run step 04 first.')
img_feat = np.load(IMAGE_ALIGNED_FILE).astype(np.float32)

print('Loading aligned text features...')
if not os.path.exists(TEXT_ALIGNED_FILE):
    raise FileNotFoundError(f'File not found: {TEXT_ALIGNED_FILE}, please run step 04 first.')
txt_feat = np.load(TEXT_ALIGNED_FILE).astype(np.float32)

if img_feat.shape[0] != txt_feat.shape[0]:
    raise ValueError(f'Image and text feature row count mismatch: {img_feat.shape[0]} vs {txt_feat.shape[0]}')

N = img_feat.shape[0]
print(f'Total nodes: {N}')
print(f'Image feature dim: {img_feat.shape[1]}, Text feature dim: {txt_feat.shape[1]}')


#2. Build Content Similarity Graph
print('Normalizing image/text features...')
faiss.normalize_L2(img_feat)
faiss.normalize_L2(txt_feat)

print('Concatenating multimodal features with weights...')
joint_feat = np.concatenate([IMAGE_WEIGHT * img_feat, TEXT_WEIGHT * txt_feat], axis=1)
faiss.normalize_L2(joint_feat)
print(f'Joint feature dim: {joint_feat.shape[1]} | image_weight={IMAGE_WEIGHT}, text_weight={TEXT_WEIGHT}')

print('Building FAISS GPU inner product index...')
dim = joint_feat.shape[1]
if not hasattr(faiss, 'StandardGpuResources'):
    raise RuntimeError('Current FAISS does not support GPU, please install faiss-gpu.')

gpu_num = faiss.get_num_gpus()
if gpu_num <= 0:
    raise RuntimeError('No available GPU detected, cannot run in GPU mode.')
if GPU_ID < 0 or GPU_ID >= gpu_num:
    raise ValueError(f'GPU_ID={GPU_ID} is out of range [0, {gpu_num - 1}]')

cpu_index = faiss.IndexFlatIP(dim)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, GPU_ID, cpu_index)
index.add(joint_feat)
print(f'FAISS GPU search enabled | gpu_id={GPU_ID} | available GPUs={gpu_num}')

print(f'Retrieving {K_CONTENT + 1} nearest neighbors for each node...')
neighbors_list = []
scores_list = []
with tqdm(total=N, desc='FAISS Search') as pbar:
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        chunk_vecs = joint_feat[start:end]
        scores_chunk, neighbors_chunk = index.search(chunk_vecs, K_CONTENT + 1)
        neighbors_list.append(neighbors_chunk)
        scores_list.append(scores_chunk)
        pbar.update(end - start)

neighbors = np.concatenate(neighbors_list, axis=0)[:, 1:]
scores = np.concatenate(scores_list, axis=0)[:, 1:]

np.save(OUTPUT_NEIGHBORS, neighbors)
np.save(OUTPUT_SCORES, scores)
print(f'Saved content graph neighbors: {OUTPUT_NEIGHBORS} {neighbors.shape}')
print(f'Saved content graph scores: {OUTPUT_SCORES} {scores.shape}')


# 3. Build Item-Item Co-occurrence Graph
def build_cooccurrence_edges(interaction_file: str, num_items: int, topk: int):
    print('Building item co-occurrence graph from interaction data...')
    if not os.path.exists(interaction_file):
        raise FileNotFoundError(f'File not found: {interaction_file}')

    df = pd.read_csv(interaction_file, usecols=['user_id', 'item_id'])
    df['item_id'] = df['item_id'].astype(int)

    co_counts = defaultdict(lambda: defaultdict(int))
    user_groups = df.groupby('user_id')['item_id'].apply(list)

    for items in tqdm(user_groups, desc='Counting Co-occurrences'):
        uniq = sorted(set(int(x) for x in items if 0 <= int(x) < num_items))
        if len(uniq) < 2:
            continue
        for a, b in combinations(uniq, 2):
            co_counts[a][b] += 1
            co_counts[b][a] += 1

    rows = []
    cols = []
    for i in tqdm(range(num_items), desc='Truncating Co-occurrence Neighbors'):
        if i not in co_counts:
            continue
        nbrs = sorted(co_counts[i].items(), key=lambda x: (-x[1], x[0]))[:topk]
        for j, _ in nbrs:
            rows.append(i)
            cols.append(int(j))

    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


co_rows, co_cols = build_cooccurrence_edges(INTERACTION_FILE, N, K_COOC)
print(f'Co-occurrence graph edges (directed): {len(co_rows):,}')


# 4. Fuse Content and Co-occurrence Graphs
print('Fusing content graph and co-occurrence graph...')
content_rows = np.repeat(np.arange(N, dtype=np.int64), neighbors.shape[1])
content_cols = neighbors.reshape(-1).astype(np.int64)

all_rows = np.concatenate([content_rows, co_rows])
all_cols = np.concatenate([content_cols, co_cols])

valid = (all_rows >= 0) & (all_rows < N) & (all_cols >= 0) & (all_cols < N) & (all_rows != all_cols)
all_rows = all_rows[valid]
all_cols = all_cols[valid]

# Symmetrize + Deduplicate
rows_sym = np.concatenate([all_rows, all_cols])
cols_sym = np.concatenate([all_cols, all_rows])
edge_pairs = np.stack([rows_sym, cols_sym], axis=1)
edge_pairs = np.unique(edge_pairs, axis=0)
rows_sym = edge_pairs[:, 0]
cols_sym = edge_pairs[:, 1]

np.savez(OUTPUT_EDGES, row=rows_sym, col=cols_sym)


print('Fused graph construction completed')
print(f'Content graph K: {K_CONTENT}')
print(f'Co-occurrence graph topK: {K_COOC}')
print(f'Final edge file: {OUTPUT_EDGES}')
print(f'Final edge count (directed, deduplicated + symmetrized): {len(rows_sym):,}')
