# MoLAQ 开发日志

## 2026-05-13

- 项目初始化：在 `/home/lml/MoLAQ` 下建立完整目录骨架。
- 创建核心算法包 `molaq/`，包含：
  - `sensitivity/hessian.py`：Hessian对角近似计算 + 激活收集hook
  - `sensitivity/saliency.py`：ViT注意力图top-K显著视觉token提取
  - `sensitivity/modal_gradient.py`：σ_l = tr(H_l)·(||g_lang||_1 + α||g_vis_sal||_1) 实现
  - `allocation/knapsack.py`：0-1背包贪心求解器（min Σσ_l·Δ_l(b_l)，s.t. bit预算约束）
  - `allocation/budget.py`：显存/bit预算估算工具
- 创建 `tests/test_knapsack.py` sanity check。
- 建立 git 仓库，完成初始提交。
- **下一步**：补全 GPTQ baseline 评测数据，建立 FP16/AWQ/GPTQ/FP8 四组结果表（Table 1雏形）。
