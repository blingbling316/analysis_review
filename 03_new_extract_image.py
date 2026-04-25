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

SAVE_INTERVAL = 50000
OUTPUT_FILE = 'image_feat.npy'
TEMP_DIR = 'image_checkpoints'
BATCH_SIZE = 64
NUM_WORKERS = 8

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
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
            return idx, img_t
        except:
            return idx, torch.zeros(3, 256, 256, dtype=torch.float32)


def run_full_extraction():
    print("Loading metadata...")
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

    final_features = np.zeros((n_items, 2048), dtype=np.float32)
    print("Extracting features...")
    with torch.no_grad():
        for batch_idx, (indices, batch_tensors) in enumerate(tqdm(dataloader, desc="Extracting Features")):

            valid_mask = batch_tensors.sum(dim=(1, 2, 3)) != 0
            if valid_mask.any():
                valid_tensors = batch_tensors[valid_mask].to(device)
                features = model(valid_tensors).cpu().numpy().reshape(-1, 2048)
                valid_indices = [indices[i] for i in range(len(indices)) if valid_mask[i]]
                for idx, feat in zip(valid_indices, features):
                    final_features[idx] = feat.astype(np.float32)

    print(f"\nSaving final file {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, final_features)
    print("Full image feature extraction completed!")

if __name__ == "__main__":
    run_full_extraction()