# MoLAQ 开发日志

## 2026-05-13

### Phase 0 端到端流水线跑通（8样本冒烟测试）🎉

**完成内容：**
- 修复 `model.visual()` 返回 `tuple` 导致的 `AttributeError`（act_norm 分支）
- 修复三处跨设备计算错误（`smooth.py` / `weighted_hessian.py` / `saliency_scaling.py`），所有张量统一在 `W.device`（cuda:0）上计算
- 修复 Flickr30k 数据加载：从 `rglob jpg/png` 改为 `load_dataset parquet` + `resolve_image_from_field`，使用真实 caption 构建 multimodal prompt
- 修复 `run_molaq.py` LLM 层名前缀（`model.language_model.layers`）

**冒烟测试结果（8 样本校准，Qwen3-VL-2B，全 A+B+C，CPU 计算）：**
- 196 个 LLM Linear 层全部量化完成，无崩溃
- κ(H_GPTQ) 普遍 10¹²~10¹⁸，κ(H_MoLAQ) 稳定 125~1000，改善 6~9 个数量级 ✅
- 量化误差 mean < 0.02（全部 196 层），max 绝大多数 < 0.3 ✅
- 1 层警告（layers.27.mlp.down_proj，max=0.63），需正式实验用 128 样本复查
- Knapsack 分配：INT4=159/196，achieved=4.506b ≈ budget 4.5b ✅
- 量化模型保存到 `/home/lml/models/Qwen3-VL-2B-MoLAQ-smoke`

**Sanity generation 验证（2026-05-14）：**
- 加载 smoke 模型，输入 MathVision 数学题图片，输出正确识别等差数列车厢题目 ✅
- Tokenizer Mistral regex warning 为误报，与 Qwen3-VL 无关，可忽略

---

## 2026-05-14

### GPU 全量计算迁移 ✅

**问题：** 量化计算（Hessian/GPTQ/smooth/knapsack）全部压在 CPU，24 核满载，GPU 只占用 5.4GB/16GB。

**根因：**
- `modal_stats.py` hook 里 `storage.append(x.cpu())`，激活强制搬 CPU
- `run_molaq.py` 里 `W = module.weight.data.detach().cpu().float()`，权重强制拉 CPU

**修复（commit 6e125c9）：**
- `modal_stats.py`：删除 hook 中所有 `.cpu()`，所有 storage/mask/p_sal 全部留在 `cuda`；`compute_stats_for_layer` 在 `X.device` 上创建权重张量
- `run_molaq.py`：`W = module.weight.data.detach().to(device=dev, dtype=torch.float32)`；逐层结尾加 `del all_stats[layer_name] + torch.cuda.empty_cache()` 防 196 层激活堆积
- `saliency_scaling.py`：修复 docstring `\i` SyntaxWarning

### 正式 GPU 运行结果（8 样本，全 A+B+C，SALIENCY_MODE=act_norm）

**硬件：** RTX 5080 Laptop，16GB VRAM  
**GPU 峰值占用：** ~9.9GB / 16GB ✅（较修复前 5.4GB 显著提升，CPU 利用率回落正常）

**κ 条件数统计（全 196 层）：**
- κ(H_GPTQ) 范围：~10¹¹ ~ 10¹⁵（极端层达 1.34×10¹⁶）
- κ(H_MoLAQ) 范围：~125 ~ 1001（稳定有界）
- 条件数改善倍数：普遍 6~9 个数量级，C 模块（smooth）+ A 模块（加权 Hessian）联合作用有效 ✅

**量化误差统计（全 196 层）：**
- mean error：0.003~0.010，浅层偏低、深层略高（正常趋势）✅
- max error：绝大多数 < 0.25，少数深层（layer 19~27）达 0.28~0.37
- ⚠️ 异常层：`layers.27.mlp.down_proj`，mean=0.0157，max=0.6289，超过阈值 0.5
  - 原因分析：该层 down_proj [2048×6144] 输出幅度在最后一层骤变，是 LLM 最后几层典型的权重分布异常（离群值集中）
  - 处理方案：正式实验（128 样本）复查；若仍异常，对该层强制分配 INT8

**Knapsack bit 分配：**
- budget=4.50b，achieved=4.506b，INT4=159/196，INT8=37/196 ✅
- 配置已保存：`/home/lml/models/Qwen3-VL-2B-MoLAQ-smoke/molaq_bits.json`

**4B 模型显存预估：**
- 模型权重 ~8.5GB + 激活矩阵（8样本）~4-5GB，峰值约 13-14GB
- 128 样本正式校准时建议 `--n_samples 64` 或 `--max_seq_len 1024` 控制显存

### 低显存分块量化脚本（scripts/run_molaq_chunked.py）✅

**动机：** 单次对 196 层全部挂 hook 收集激活，n_samples=64/128 时显存随样本数线性增长，16GB 显卡上直接 OOM。

**方案：**
- 新增 `scripts/run_molaq_chunked.py`，保持 A/B/C 算法完全不变，只改变调度：
  - 按 `chunk_size`（默认 4）把 196 个 Linear 层分块
  - 对每一块：
    1. 仅对该块的层挂 hook，跑一遍全校准集，调用 `collect_modal_stats` 收集统计量
    2. 立即对该块的层执行 Stage 2/3/4（C+A+B）量化，并写回权重
    3. 删除该块的 `all_stats_chunk` 并 `torch.cuda.empty_cache()`
  - 下一块继续，如此直到全部 196 层完成

**接口：**
- `--chunk_size`：每次同时收集并量化的层数，默认 4，显存 ~ O(chunk_size)
- `--sanity`：是否启用每层 κ(H) 的 sanity check（默认关闭以节省显存与时间）

**预期效果：**
- 8 样本 smoke：峰值显存应明显低于旧脚本的 ~9.9GB
- 64/128 样本：不再在 Stage 0+1 OOM，可在 16GB 显存卡上完成正式校准

**下一步：**
1. 用 `scripts/run_molaq_chunked.py` 在 8 样本上做 sanity，对比显存占用和误差是否一致
2. 成功后，将其作为 2B/4B 正式 MoLAQ 校准脚本，原脚本保留做对照

### 128 样本正式校准（chunk_size=4, 启用 sanity）✅

**运行配置：**
- 脚本：`scripts/run_molaq_chunked.py`
- 校准集：Flickr30k parquet，`n_samples=128`，`max_seq_len=2048`
- 模型：Qwen3-VL-2B-Instruct（LLM 端 196 个 Linear 层）
- 参数：`--enable_A --enable_B --enable_C --budget_bits 4.5 --chunk_size 4 --sanity`

**显存曲线：**
- 专用 GPU 显存在各 chunk 内缓慢上升至 ~8.7GB，然后在 chunk 结束时回落到 ~6GB 左右，整体在 6–9GB 区间波动（RTX 5080 Laptop, 16GB VRAM）。
- 说明分块调度和 `del all_stats_chunk[...] + torch.cuda.empty_cache()` 生效，每个 chunk 的激活和 Hessian 统计都能被释放。

**量化误差统计（128 样本）：**
- 前半层：mean ≈ 0.003–0.007，max 主流在 0.05–0.15 之间，与 8 样本一致或略高；
- 深层（layer 14 之后）：mean 上升至 0.007–0.011，max 主流在 0.2–0.35 区间；
- 异常层：`layers.27.mlp.down_proj`，mean≈0.0158, max≈0.672（稳定复现），后续考虑强制 INT8。

**κ(H) 条件数（128 样本）：**
- κ(H_GPTQ) 范围约 10⁵–10¹⁴；
- κ(H_MoLAQ) 全部落在 ~170–510 区间（个别极端层 ≈ 770），与 8 样本同一数量级；
- A+C 模块在所有层上将 κ 从 GPTQ 的 10¹¹–10¹⁵ 降到 10²–10³，改善 5–9 个数量级。

**Knapsack bit 分配：**
- budget=4.50b，achieved=4.506b，INT4=159/196，INT8=37/196，与 8 样本配置保持一致。

**结论：**
- 基于 128 样本的 MoLAQ Hessian 统计已经稳定：κ(H_MoLAQ) 在所有层上收敛于 O(10²–10³)，量化误差均值 <0.02，问题层可通过后处理强制 INT8。
- 分块脚本在 16GB GPU 上成功完成 128 样本 + sanity 的全量量化，为后续多模态基准评测提供了可靠的 2B MoLAQ 模型快照 `/home/lml/models/Qwen3-VL-2B-MoLAQ-128-chunked`。
