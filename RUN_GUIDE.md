# 项目脚本执行指南（完整流程）

这份指南讲清楚：这些 Python 脚本怎么跑、按什么顺序跑、每一步会产出什么文件，以及常见坑如何避免。

---

## 你将得到什么

- **交互数据（5-core + 数字ID）**：`elec_5core_interactions.csv`
- **商品元数据（精简后 + 数字item_id）**：`elec_5core_meta.jsonl`
- **训练/验证/测试划分**：`data/Electronics/train.json`、`val.json`、`test.json`
- **文本特征向量**（每个商品 1 条）：`text_feat.npy`（通常是 `n_items × 384`）
- **图片特征向量**（每个商品 1 张图）：`image_feat.npy`（`n_items × 2048`）

> 说明：当前代码里“文本向量”来自商品 `title + categories`，不是逐条 review（评论）文本向量。

---

## 环境准备（Windows + PowerShell）

建议使用 Python 3.10/3.11（3.8 也大概率可用）。

### 1) 创建虚拟环境（推荐）

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) 安装依赖

你这些脚本用到的主要库：

- `numpy`
- `pandas`
- `ujson`
- `tqdm`
- `requests`
- `Pillow`
- `torch`、`torchvision`
- `sentence-transformers`

安装命令（一次性装齐）：

```bash
pip install -U pip
pip install numpy pandas ujson tqdm requests Pillow sentence-transformers
pip install torch torchvision
```

> 如果你有 NVIDIA 显卡并想用 CUDA，请安装与你 CUDA 版本匹配的 PyTorch（以 PyTorch 官网为准）。不装 CUDA 也能跑，只是会慢很多。

---

## 数据文件准备（你需要放到哪里、叫什么）

这些脚本默认用**相对路径**读取文件，通常意味着：你要在**项目根目录**（也就是这些 `.py` 文件所在目录）直接运行它们。

### 必需的原始数据（用于 5-core 预处理）

`01_rawdata_filter_5_core.py` 里写死的路径是：

- `Digital_Music.jsonl/Electronics.jsonl`
- `Digital_Music.jsonl/meta_Electronics.jsonl`

所以你需要把文件放成这样：

```text
my_project/
  Digital_Music.jsonl/
    Electronics.jsonl
    meta_Electronics.jsonl
  01_rawdata_filter_5_core.py
  ...
```

> 如果你的原始数据不在这个位置/不是这个名字，请直接改 `01_rawdata_filter_5_core.py` 最下面那行 `fast_preprocess(...)` 的两个输入路径。

---

## 完整执行流程（推荐顺序）

下面按“从原始数据 → 可训练数据 → 特征向量”的顺序来跑。

### 步骤 A（可选）：先看看元数据缺失情况（EDA）

脚本：`00_EDA.py`

用途：统计 `.jsonl` 每列的非空率。

⚠️ 注意：这个脚本里写死了一个你电脑上可能不存在的路径（`D:\...`），你需要把最后一行改成你自己的 `.jsonl` 路径再运行。

运行：

```bash
python .\00_EDA.py
```

---

### 步骤 B：5-core 过滤 + 生成干净交互/元数据（核心步骤）

脚本：`01_rawdata_filter_5_core.py`

它会做三件事：

- 扫描交互数据，找出满足 5-core 的用户/商品
- 把原始 `user_id`/`parent_asin` 映射成数字 ID（从 0 开始）
- 输出精简元数据（只保留 `title/images/categories` 等字段）

运行：

```bash
python .\01_rawdata_filter_5_core.py
```

产出（默认写到项目根目录）：

- `elec_5core_interactions.csv`
- `elec_5core_meta.jsonl`

---

### 步骤 C：按用户切分 train/val/test

脚本：`00_data_split.py`

它会读取：

- `elec_5core_interactions.csv`

并写出到：

- `data/Electronics/train.json`
- `data/Electronics/val.json`
- `data/Electronics/test.json`

运行：

```bash
python .\00_data_split.py
```

---

### 步骤 D：提取“商品文本向量”（title + categories）

脚本：`02_extract_text.py`

它会读取：

- `elec_5core_meta.jsonl`

输出：

- `text_feat.npy`

运行：

```bash
python .\02_extract_text.py
```

提示：

- 机器没 GPU 也能跑，但会慢。
- 文本向量与 `item_id` 一一对应：`text_feat[i]` 就是 `item_id=i` 的商品向量。

---

### 步骤 E（推荐）：检查文本向量 shape 是否正确

脚本：`00_check_shape.py`

读取：

- `text_feat.npy`

运行：

```bash
python .\00_check_shape.py
```

你会看到类似：

- `n_items`（商品数）
- `dim`（向量维度，`all-MiniLM-L6-v2` 通常是 384）

---

### 步骤 F（可选）：随机下载 100 张商品图做肉眼检查

脚本：`00_download_image100.py`

⚠️ 注意：这个脚本默认读取 `01_elec_5core_meta.jsonl`，但你在步骤 B 产出的是 `elec_5core_meta.jsonl`。

你有两种方式修：

- **方式 1（推荐）**：把脚本最后一行改成：
  - `download_100_samples('elec_5core_meta.jsonl')`
- **方式 2**：把 `elec_5core_meta.jsonl` 复制/改名成 `01_elec_5core_meta.jsonl`

运行：

```bash
python .\00_download_image100.py
```

输出目录：

- `test_images/`

---

### 步骤 G（可选）：先小规模测速图片特征提取耗时

脚本：`03_test_extract_image.py`

运行（默认测试前 5000 个商品）：

```bash
python .\03_test_extract_image.py
```

它会输出：

- 平均每个样本耗时
- 全量大概需要多久

> 这个脚本主要用于估时/验证，不会产出最终的 `image_feat.npy`。

---

### 步骤 H：全量提取“商品图片向量”（ResNet50, 2048维）

脚本：`03_extract_image.py`

读取：

- `elec_5core_meta.jsonl`

输出：

- `image_feat.npy`

运行：

```bash
python .\03_extract_image.py
```

重要提示（很关键）：

- **耗时很长**：需要下载大量图片，受网速/对方服务器限制影响很大。
- **需要较大内存**：脚本会一次性初始化 `n_items × 2048` 的矩阵，商品数很大时会吃内存。
- **失败回退**：没图或下载失败会用全 0 向量填充该 `item_id` 的行。
- **每个商品只用 1 张图**：取 `images[0]`（第一张）。

---

## 如何用这些向量做“相似商品检索”（你后续要做的事）

当前脚本只负责把特征提出来；“融合 + 相似检索”通常在另一个脚本里做。

最常见做法：

1. 读入 `text_feat.npy` 和 `image_feat.npy`（两者都以 `item_id` 作为行索引）
2. 各自做 L2 归一化
3. 加权拼接成融合向量
4. 用余弦相似度做 Top-K 最近邻搜索

---

## 常见问题排查

- **找不到文件**：大概率是你不是在项目根目录运行，或文件名/路径和脚本写的不一致。
- **图片下载很慢/失败多**：网络、超时、对方限流都可能导致；可以先跑 `03_test_extract_image.py` 估时。
- **CPU 太慢**：文本和图片特征都能用 GPU 加速（尤其是图片 ResNet50）。

