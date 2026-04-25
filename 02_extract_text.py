import numpy as np
import ujson as json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os


def run_text_feature_extraction():

    INPUT_META_FILE = '01_elec_5core_meta.jsonl'
    OUTPUT_FEAT_FILE = 'text_feat.npy'
    MODEL_NAME = 'all-MiniLM-L6-v2'
    BATCH_SIZE = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected device: {device.upper()}")
    if not os.path.exists(INPUT_META_FILE):
        print(f"Error: File {INPUT_META_FILE} not found, please check the path!")
        return

    # Load pre-trained model
    print(f"Loading BERT model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Read metadata
    items_raw = []
    print("Loading metadata and aligning ID order...")
    with open(INPUT_META_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading"):
            try:
                items_raw.append(json.loads(line))
            except:
                continue

    # Sort by item_id ascending to ensure feature matrix row indices match numeric IDs
    items_raw.sort(key=lambda x: x['item_id'])

    # Text cleaning and concatenation
    clean_texts = []
    for item in tqdm(items_raw, desc="Text Preprocessing"):
        title = str(item.get('title', '')).strip()
        # Process categories, may be a nested list
        cats = item.get('categories', [])
        if isinstance(cats, list):
            # Flatten the list and convert to string
            cat_str = " ".join([str(c) for c in cats])
        else:
            cat_str = str(cats)

        # Concatenate Title and Category for richer semantics
        combined_text = f"{title} {cat_str}".strip()

        # Handle empty text to avoid model errors
        if len(combined_text) < 2:
            combined_text = "electronic product"
        clean_texts.append(combined_text)

    # Perform encoding
    print(f"Converting text to embeddings (Total {len(clean_texts)} entries)...")
    # Automatically uses GPU if available
    embeddings = model.encode(
        clean_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save results
    print(f"Saving to disk: {OUTPUT_FEAT_FILE}")
    np.save(OUTPUT_FEAT_FILE, embeddings)

    print(f"Processing completed!")
    print(f"Final feature matrix shape: {embeddings.shape}")
    print(f"File path: {os.path.abspath(OUTPUT_FEAT_FILE)}")


if __name__ == "__main__":
    run_text_feature_extraction()