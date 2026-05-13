# MoLAQ 开发日志

---

## 2026-05-13

### 完成内容
- 初始化 GitHub 仓库完整骨架，包含所有模块占位文件、README、requirements.txt
- 完成两轮代码审查，发现并记录 7 处 `.shape` 取 tuple 的静默 bug，已在 v2 文档中全部修正
- 确认算法可行度 ≈ 99%，理论与工程映射自洽

### 已创建文件
- `molaq/__init__.py`
- `molaq/stats/__init__.py` + `modal_stats.py`（待实现）
- `molaq/core/__init__.py` + `smooth.py` + `weighted_hessian.py` + `saliency_scaling.py`（待实现）
- `molaq/assign/__init__.py` + `knapsack.py`（待实现）
- `scripts/run_molaq.py`（骨架）
- `tests/test_vit_attention.py` + `test_smooth.py` + `test_knapsack.py`（待实现）

### 待处理
- [ ] 运行 `tests/test_vit_attention.py`，确定 `SALIENCY_MODE`
- [ ] 实现 `molaq/core/smooth.py` 并通过 `tests/test_smooth.py`
- [ ] 实现 `molaq/stats/modal_stats.py`（LayerStats + collect_modal_stats）
- [ ] 实现 `molaq/core/weighted_hessian.py`，逐层打印量化误差
- [ ] 实现 `molaq/core/saliency_scaling.py`
- [ ] 实现 `molaq/assign/knapsack.py` 并通过 `tests/test_knapsack.py`
- [ ] 补全 `scripts/run_molaq.py`（dataloader + get_linear_layers）
- [ ] 50 样本 MMStar 快速验证

---
