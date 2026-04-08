import numpy as np
import ujson as json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc

# ---------- 强制限制 CPU 线程数（避免 15000% 问题）----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ---------- 配置 ----------
SAVE_INTERVAL = 50000          # 每 5 万条保存一次中间结果
OUTPUT_FILE = 'image_feat.npy'
CHECKPOINT_DIR = 'image_checkpoints'
BATCH_SIZE = 128                # GPU 批量大小（根据显存调整，8GB 可跑 64~128）
DOWNLOAD_WORKERS = 8           # 并发下载线程数（纯 I/O，不占 CPU）

# ---------- 模型与预处理 ----------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),   # ResNet50 需要 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])  # 输出 2048 维特征
model.to(device)
model.eval()

# ---------- 下载函数（只下载原始字节，不做任何预处理）----------
def download_only(item, idx):
    """返回 (idx, raw_bytes) 或 (idx, None)"""
    images = item.get('images', [])
    url = ""
    if images:
        img_info = images[0]
        url = img_info.get('large') or img_info.get('url')
    if not url:
        return idx, None
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return idx, resp.content
        else:
            return idx, None
    except Exception:
        return idx, None

# ---------- 主流程 ----------
def run_extraction():
    # 1. 加载元数据并排序
    print("加载元数据...")
    items = []
    with open('01_elec_5core_meta.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    items.sort(key=lambda x: x['item_id'])
    n_items = len(items)
    print(f"总物品数: {n_items}")

    # 2. 断点续传：检查已有 checkpoint
    final_features = np.zeros((n_items, 2048), dtype=np.float32)
    processed_up_to = 0
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    # 查找最新的 checkpoint
    ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_') and f.endswith('.npy')]
    if ckpt_files:
        # 提取起始索引并排序
        starts = [int(f.split('_')[1].split('.')[0]) for f in ckpt_files]
        max_start = max(starts)
        processed_up_to = max_start + SAVE_INTERVAL
        # 加载该 checkpoint 到 final_features
        latest_ckpt = os.path.join(CHECKPOINT_DIR, f'checkpoint_{max_start}.npy')
        final_features = np.load(latest_ckpt)
        print(f"从 checkpoint 恢复，已处理至索引 {processed_up_to}")

    # 3. 分批处理
    for start_idx in range(processed_up_to, n_items, SAVE_INTERVAL):
        end_idx = min(start_idx + SAVE_INTERVAL, n_items)
        chunk_items = items[start_idx:end_idx]
        print(f"\n处理批次 [{start_idx}:{end_idx}] (共 {len(chunk_items)} 条)")

        # ----- 3.1 多线程下载（仅下载原始字节）-----
        download_results = {}
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = {executor.submit(download_only, item, start_idx + i): (start_idx + i) for i, item in enumerate(chunk_items)}
            with tqdm(total=len(futures), desc="下载图片") as pbar:
                for future in as_completed(futures):
                    idx, raw_bytes = future.result()
                    download_results[idx] = raw_bytes
                    pbar.update(1)

        # ----- 3.2 主线程串行预处理 + 批量推理（带进度条）-----
        batch_indices = []
        batch_tensors = []
        # 使用 tqdm 显示预处理进度
        for idx in tqdm(range(start_idx, end_idx), desc="预处理+推理"):
            raw_bytes = download_results.get(idx)
            if raw_bytes is None:
                continue
            try:
                img = Image.open(BytesIO(raw_bytes)).convert('RGB')
                tensor = preprocess(img)
                batch_indices.append(idx)
                batch_tensors.append(tensor)
            except Exception:
                continue

            # 达到批次大小或最后一条时执行推理
            if len(batch_tensors) >= BATCH_SIZE or (idx == end_idx - 1 and batch_tensors):
                batch = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    feats = model(batch).cpu().numpy().reshape(len(batch), -1)
                for i, orig_idx in enumerate(batch_indices):
                    final_features[orig_idx] = feats[i].astype(np.float32)
                # 清空 batch，释放显存
                batch_indices = []
                batch_tensors = []
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        # ----- 3.3 保存本批 checkpoint -----
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{start_idx}.npy')
        np.save(ckpt_path, final_features)
        print(f"已保存 checkpoint: {ckpt_path}")

        # 可选：删除旧的 checkpoint 节省磁盘空间（保留最近两个）
        old_ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')])
        if len(old_ckpts) > 2:
            for f in old_ckpts[:-2]:
                os.remove(os.path.join(CHECKPOINT_DIR, f))

    # 4. 最终保存
    print(f"\n保存最终特征矩阵到 {OUTPUT_FILE}")
    np.save(OUTPUT_FILE, final_features)
    print("✅ 全部完成！")

if __name__ == "__main__":
    run_extraction()