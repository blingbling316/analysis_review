import os
import ujson as json
import requests
import random
from tqdm import tqdm
from PIL import Image
from io import BytesIO


def download_100_samples(meta_path, output_dir='test_images'):
    # 1. 创建保存目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 2. 读取所有元数据
    items = []
    print("正在读取元数据...")
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))

    # 3. 过滤出有图片的商品并随机抽取 100 个
    items_with_images = [it for it in items if it.get('images') and len(it['images']) > 0]

    if len(items_with_images) < 100:
        samples = items_with_images
        print(f"有图片的商品不足 100 个，将下载全部 {len(samples)} 个。")
    else:
        samples = random.sample(items_with_images, 100)
        print(f"已随机抽取 100 个商品。")

    # 4. 开始下载
    success_count = 0
    for item in tqdm(samples, desc="下载进度"):
        item_id = item['item_id']
        # 尝试获取图片 URL
        img_info = item['images'][0]
        url = img_info.get('large') or img_info.get('url')

        if not url:
            continue

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # 自动识别图片格式并保存
                img = Image.open(BytesIO(response.content))
                ext = img.format.lower() if img.format else 'jpg'
                save_path = os.path.join(output_dir, f"{item_id}.{ext}")

                with open(save_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
        except Exception as e:
            # 记录失败但不中断
            continue

    print(f"\n✅ 任务完成！")
    print(f"成功下载: {success_count} 张图片")
    print(f"保存位置: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # 请确保元数据文件名正确
    download_100_samples('01_elec_5core_meta.jsonl')