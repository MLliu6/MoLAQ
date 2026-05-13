# MoLAQ: Modality & Layer-Aware Mixed-Precision Quantization for VLMs

> 针对 Qwen3-VL 系列模型的模态感知、层级混合精度量化算法

## 算法概述

MoLAQ 包含三个核心创新点，依次作用于量化误差的不同环节：

| 创新点 | 模块 | 作用 |
|--------|------|------|
| **C**: 模态感知激活预平滑 | `molaq/core/smooth.py` | 消除显著视觉 token 通道间幅度不均匀，改善 Hessian 条件数 |
| **A**: 模态加权 Hessian + GPTQ | `molaq/core/weighted_hessian.py` | 按各组 token 对量化误差的贡献加权重建，替换等权 Hessian |
| **B**: 显著 token 驱动缩放搜索 | `molaq/core/saliency_scaling.py` | 用显著性加权统计量替代 AWQ 全 token 均值，保护关键通道 |

执行顺序固定：`Stage 0 (统计) → Stage 2 (C) → Stage 3 (A) → Stage 4 (B) → knapsack 分配`

## 代码库架构

```
MoLAQ/
├── README.md
├── log.md                          # 每日进展日志
├── requirements.txt
├── molaq/
│   ├── __init__.py
│   ├── stats/
│   │   ├── __init__.py
│   │   └── modal_stats.py          # §3.2–3.3: Token 分组、激活收集、共享统计量
│   ├── core/
│   │   ├── __init__.py
│   │   ├── smooth.py               # §3.4: 创新点 C
│   │   ├── weighted_hessian.py     # §3.5: 创新点 A
│   │   └── saliency_scaling.py     # §3.6: 创新点 B
│   └── assign/
│       ├── __init__.py
│       └── knapsack.py             # §3.7: 混合精度贪心分配
├── scripts/
│   └── run_molaq.py                # 流水线主调脚本
├── tests/
│   ├── test_smooth.py
│   ├── test_knapsack.py
│   └── test_vit_attention.py
└── configs/
    └── .gitkeep
```

## 安装

```bash
source ~/vllm/bin/activate
pip install -r requirements.txt
```

## 快速上手

```bash
source ~/vllm/bin/activate
cd ~/MoLAQ

# Step 1: 运行 ViT Attention 可获取性检查（必须先执行）
python tests/test_vit_attention.py

# Step 2: 跑单元测试
pytest -q tests/

# Step 3: 运行 MoLAQ 完整流水线
python scripts/run_molaq.py \
    --model /home/lml/models/Qwen3-VL-2B-Instruct \
    --calib_data /mnt/e/BISHE_START/Datasets/flickr30k/data \
    --output /home/lml/models/Qwen3-VL-2B-MoLAQ \
    --enable_A --enable_B --enable_C \
    --budget_bits 4.5
```

## 消融实验配置

| 配置 | C | A | B | 说明 |
|------|:---:|:---:|:---:|------|
| GPTQ-baseline | ✗ | ✗ | ✗ | 原始等权 Hessian |
| +C only | ✓ | ✗ | ✗ | 仅预平滑 |
| +A only | ✗ | ✓ | ✗ | 仅加权 Hessian |
| +B only | ✗ | ✗ | ✓ | 仅显著 scaling |
| +A+C | ✓ | ✓ | ✗ | 完整重建路径 |
| **MoLAQ** | ✓ | ✓ | ✓ | **完整方法** |

## 环境要求

- OS: WSL2
- 虚拟环境: `~/vllm`
- GPU: CUDA capable (推荐 ≥ 16GB VRAM for 2B)
- 模型路径: `/home/lml/models/Qwen3-VL-2B-Instruct`
- 校准数据: `/mnt/e/BISHE_START/Datasets/flickr30k/data`
