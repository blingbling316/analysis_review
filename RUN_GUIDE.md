# 多模态商品推荐项目完整流程（从零到可复现）

本文件基于当前仓库中的实际脚本与默认参数整理，目标是：

- 用商品文本 + 商品图片做多模态表示；
- 通过跨模态对齐和 KNN 建图构造商品图；
- 训练 GraphSAGE + BPR 推荐模型；
- 做诊断与（条件满足时）推理。

---

## 1. 项目结构与职责（按阶段）

- `01_rawdata_filter_5_core.py`：从原始 Amazon 风格数据做 5-core 过滤 + ID 映射，产出交互和精简 meta。
- `00_data_split.py`：按用户内部交互切分 train/val/test（json 字典格式）。
- `02_extract_text.py`：从 `title + categories` 提取文本向量（SentenceTransformer）。
- `03_new_extract_image.py`：从商品图 URL 抽取 ResNet50 图像向量。
- `04_alignment_02.py`：跨模态对齐（InfoNCE 双塔投影），导出对齐后的 text/image 向量。
- `05_build_joint_knn_from_aligned.py`：拼接对齐向量并用 FAISS 建联合 KNN 图。
- `07_train_bpr_mhm_02.py`：在商品图上训练 GraphSAGE + BPR。
- `debug_coldstart_pipeline.py`：全链路诊断（特征质量、图质量、评估退化等），可写日志。
- `08_infer_recommender.py`：推理脚本（注意与训练 ckpt 格式兼容性，见文末）。
- `06_gnn_model.py`：GraphSAGE 模型定义与导出工具（辅助脚本）。

---

## 2. 环境准备

推荐你当前使用的 Conda 方式（`start.md`）：

```bash
conda create -n analysis_review python=3.11 -y
conda activate analysis_review
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

补充说明：

- 若无 GPU，把第三行改成 CPU 版本安装 PyTorch。
- `torch-geometric` 在部分环境可能需要按官方说明单独安装对应 wheel。

---

## 3. 关键输入与命名约定

当前脚本默认大量使用固定文件名。仓库中已存在以下主干产物：

- `01_elec_5core_interactions.csv`
- `01_elec_5core_meta.jsonl`
- `02_text_feat.npy`
- `03_image_feat.npy`
- `04_image_feat_aligned_02.npy`
- `04_text_feat_aligned_02.npy`
- `05_joint_knn_edges_02.npz`

建议后续都沿用这套 `01/02/03/04/05 + _02` 命名，避免脚本互相找不到文件。

---

## 4. 从原始数据开始的完整执行顺序

> 若你已经有第 01~05 阶段产物，可从第 8 节直接训练或从第 9 节诊断。

### 步骤 A：5-core 预处理（交互 + 元数据）

执行：

```bash
python 01_rawdata_filter_5_core.py
```

默认输入（脚本中写死）：

- `Digital_Music.jsonl/Electronics.jsonl`
- `Digital_Music.jsonl/meta_Electronics.jsonl`

默认输出（脚本参数 `output_prefix='elec_5core'`）：

- `elec_5core_interactions.csv`
- `elec_5core_meta.jsonl`

建议：若后续要无缝跑当前主链脚本，请把输出重命名为：

- `01_elec_5core_interactions.csv`
- `01_elec_5core_meta.jsonl`

### 步骤 B：切分 train/val/test（可选）

执行：

```bash
python 00_data_split.py
```

默认读取：`elec_5core_interactions.csv`
输出：`data/Electronics/train.json`、`val.json`、`test.json`

说明：该切分目前主要用于数据分析/备用流程；主训练脚本 `07_train_bpr_mhm_02.py` 会自行再按用户切分。

### 步骤 C：文本特征提取

执行：

```bash
python 02_extract_text.py
```

默认读取：`01_elec_5core_meta.jsonl`
输出：`text_feat.npy`

建议统一命名：

```bash
# PowerShell
Copy-Item text_feat.npy 02_text_feat.npy
```

> 代码里有一个小问题：`print(f"检测到设备: {device.upper(x)}")` 应该改成 `device.upper()`。

### 步骤 D：图像特征提取

执行：

```bash
python 03_new_extract_image.py
```

默认读取：`01_elec_5core_meta.jsonl`
输出：`image_feat.npy`

建议统一命名：

```bash
Copy-Item image_feat.npy 03_image_feat.npy
```

备注：脚本遇到无图或下载失败时会填零向量（这会影响建图质量，后续可用诊断脚本检查比例）。

### 步骤 E：跨模态对齐（InfoNCE）

执行：

```bash
python 04_alignment_02.py
```

默认读取：

- `03_image_feat.npy`
- `02_text_feat.npy`
- `01_elec_5core_interactions.csv`

输出：

- `04_cross_modal_alignment_02.pt`
- `04_image_feat_aligned_02.npy`
- `04_text_feat_aligned_02.npy`

核心机制：

- 在训练阶段仅用训练划分里的 item 学 projection head；
- 推理阶段对全量 item 导出对齐向量；
- 损失函数为双向 InfoNCE。

### 步骤 F：构建联合 KNN 图

执行：

```bash
python 05_build_joint_knn_from_aligned.py
```

默认读取：

- `04_image_feat_aligned_02.npy`
- `04_text_feat_aligned_02.npy`

输出：

- `05_joint_knn_neighbors_02.npy`
- `05_joint_knn_scores_02.npy`
- `05_joint_knn_edges_02.npz`

说明：

- 使用 FAISS 内积索引 + L2 归一化（等价余弦）；
- 检索 `K+1` 后去掉自身，理论上不应有 self-loop（若仍出现，建议用诊断脚本复查）。

---

## 5. 训练主模型（GraphSAGE + BPR）

执行：

```bash
python 07_train_bpr_mhm_02.py
```

默认读取：

- `04_image_feat_aligned_02.npy`
- `04_text_feat_aligned_02.npy`
- `05_joint_knn_edges_02.npz`
- `01_elec_5core_interactions.csv`

输出目录：`outputs_isolated/`

主要产物：

- `best_model.pth`（仅 `model.state_dict()`）
- `checkpoint_epoch_*.pt`（包含优化器、曲线、best_auc 等）
- `user_emb_cache.pt`（验证用户向量缓存）

训练逻辑要点：

- 先按用户切分 train/val；
- 只保留 train-item 子图训练；
- 用边采样构造 `(u, pos, neg)`，优化 BPR 损失；
- 每个 epoch 做 AUC + Recall@K 评估。

---

## 6. 诊断与质量检查（强烈建议每轮都跑）

执行：

```bash
python debug_coldstart_pipeline.py --log-file debug_coldstart_pipeline.log
```

会检查：

- 交互基本统计、重复行、ID 范围；
- 特征矩阵 NaN/Inf/零行；
- 行数对齐；
- KNN 非法边、自环、分数分布、双向边比例；
- 当前 07 评估协议下的 `global_mean fallback` 比例；
- 样本商品近邻可解释性。

如果你看到：

- `global_mean_fallback_users` 非常高（尤其接近 100%），说明当前验证协议会严重弱化评估可信度；
- `aligned_text_feat` 缺失，说明流程没有完整跑通（仅图像对齐不够）。

---

## 7. 推理（当前状态下的兼容性说明）

脚本：`08_infer_recommender.py`

它要求 `--ckpt` 内含：

- `gnn_state_dict`
- `rec_state_dict`（含 `user_embedding.weight`）

但当前 `07_train_bpr_mhm_02.py` 保存的 `best_model.pth` 只有单一 `model.state_dict()`，**二者格式不兼容**，因此直接推理会失败。

可选方案：

1. **改训练保存格式**：在训练脚本里按 `08` 所需键名保存完整字典；
2. **改推理脚本**：适配 `07` 的 checkpoint（仅 item encoder），并重新定义用户向量来源；
3. **仅做 item-item 检索**：直接基于 `debug` 或 `05` 产物做相似商品推荐。

---

## 8. 一条可直接复现的命令链（当前项目命名）

```bash
python 02_extract_text.py
Copy-Item text_feat.npy 02_text_feat.npy

python 03_new_extract_image.py
Copy-Item image_feat.npy 03_image_feat.npy

python 04_alignment_02.py
python 05_build_joint_knn_from_aligned.py
python 07_train_bpr_mhm_02.py

python debug_coldstart_pipeline.py --log-file debug_coldstart_pipeline.log
```

---

## 9. 常见问题与改进建议

- **评估退化**：按用户切分导致验证用户在训练集中无历史，07 会大量退化到 `global_mean`；建议改成时间切分或 per-user 留一。
- **命名不一致**：`text_feat.npy/image_feat.npy` 与 `02_/03_` 前缀混用，建议统一命名或在脚本顶部常量统一维护。
- **图像零向量**：缺图商品建议后续加入“文本兜底”或“均值向量兜底”，减少 KNN 噪声。
- **推理链路断点**：先统一 07 与 08 的 checkpoint 结构，再做线上化。

---

## 10. 文件对应速查表

- 原始交互/元数据：`Digital_Music.jsonl/*.jsonl`
- 5-core：`01_elec_5core_interactions.csv`、`01_elec_5core_meta.jsonl`
- 原始特征：`02_text_feat.npy`、`03_image_feat.npy`
- 对齐特征：`04_text_feat_aligned_02.npy`、`04_image_feat_aligned_02.npy`
- 图结构：`05_joint_knn_edges_02.npz`（及 neighbors/scores）
- 训练输出：`outputs_isolated/best_model.pth`、`checkpoint_epoch_*.pt`
- 诊断日志：`debug_coldstart_pipeline.log`

如果你希望，我可以在下一步直接把 `07_train_bpr_mhm_02.py` 与 `08_infer_recommender.py` 的 checkpoint 格式做成完全打通（训练后可直接 `recommend/similar`）。
