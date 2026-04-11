import numpy as np
import ujson as json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os

SAVE_INTERVAL = 50000  # 每 5 万个物品保存一次，防止白跑
OUTPUT_FILE = 'image_feat.npy'
TEMP_DIR = 'image_checkpoints'
BATCH_SIZE = 64
NUM_WORKERS = 8

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    # transforms.CenterCrop(224), # 我们图片是完整的，这里是否还有需要你裁减，有待商议
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()


class ImageFeatureDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        item_id = item['item_id']
        images = item.get('images', [])
        url = ""
        if images and len(images) > 0:
            img_info = images[0]
            url = img_info.get('large') or img_info.get('url')

        if not url:
            return idx, torch.zeros(3, 256, 256, dtype=torch.float32)

        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_t = self.preprocess(img)
            return idx, img_t  # 返回预处理后的tensor，不在这里转GPU
        except:
            return idx, torch.zeros(3, 256, 256, dtype=torch.float32)


def run_full_extraction():
    # 读取并排序
    print("正在加载元数据...")
    items = []
    with open('01_elec_5core_meta.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    items.sort(key=lambda x: x['item_id'])

    n_items = len(items)
    dataset = ImageFeatureDataset(items, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    # 初始化最终的大矩阵
    final_features = np.zeros((n_items, 2048), dtype=np.float32)
    print("开始提取特征...")
    with torch.no_grad():
        for batch_idx, (indices, batch_tensors) in enumerate(tqdm(dataloader, desc="提取特征")):
            # indices: 当前batch的原始索引列表
            # batch_tensors: 预处理后的图像tensor [batch_size, 3, 256, 256]
            valid_mask = batch_tensors.sum(dim=(1, 2, 3)) != 0
            if valid_mask.any():
                # 将有效的tensor移到GPU
                valid_tensors = batch_tensors[valid_mask].to(device)
                # GPU推理
                features = model(valid_tensors).cpu().numpy().reshape(-1, 2048)
                # 填充特征到对应位置
                valid_indices = [indices[i] for i in range(len(indices)) if valid_mask[i]]
                for idx, feat in zip(valid_indices, features):
                    final_features[idx] = feat.astype(np.float32)

    # 最终保存
    print(f"\n正在保存最终文件 {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, final_features)
    print("✅ 全量图像特征提取完成！")

if __name__ == "__main__":
    run_full_extraction()
