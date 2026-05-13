# 3. MoLAQ: Modality & Layer-Aware Mixed-Precision Quantization

## 3.1 Preliminary: PTQ Error Decomposition

设 VLM 由 ViT Encoder $\mathcal{E}$ 和 LLM Decoder $\mathcal{D}$ 组成，
共 $L$ 个可量化线性层 $\{W_l\}_{l=1}^{L}$，
$W_l \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$。
给定多模态校准集 $\mathcal{C}$（128 个图文对），
令 $X_l \in \mathbb{R}^{N \times d_\text{in}}$ 为层 $l$
在校准集上收集的输入激活（$N$ 为所有样本的 token 总数）。

层 $l$ 的量化误差分解为：

$$
\mathcal{L}_l =
\underbrace{\|(W_l - \hat{W}_l)\tilde{X}_l\|_F^2}_{
  \text{权重重建误差，由 C+A 控制}}
+
\underbrace{\|\hat{W}_l^{\text{scale}} X_l - W_l X_l\|_F^2}_{
  \text{缩放搜索残差，由 B 控制}}
$$

其中 $\tilde{X}_l$ 为经创新点 C 预平滑后的激活，
$\hat{W}_l^{\text{scale}}$ 为经创新点 B 缩放搜索后的量化权重。
MoLAQ 的三个创新点依次作用于上述误差分解的不同环节，
共享一次 forward pass 收集的激活统计量
（$e_m$、$p_i$、$\bar{x}_j$），
构成固定执行顺序的四阶段流水线（图 1）：

$$
\text{Stage 0 (统计)} \;\to\;
\text{Stage 1 (共享统计量)} \;\to\;
\underbrace{\text{Stage 2 (C)}}_{\text{预平滑}} \;\to\;
\underbrace{\text{Stage 3 (A)}}_{\text{加权 Hessian}} \;\to\;
\underbrace{\text{Stage 4 (B)}}_{\text{显著 scaling}}
$$

---

## 3.2 Stage 0: Calibration Forward Pass 与 Token 分组

对校准集 $\mathcal{C}$ 执行完整 forward pass，收集：

- 每层输入激活 $X_l$（在 `nn.Linear` 输入端注册 forward hook 截取）；
- ViT 最后一层的显著性分数 $p_i$（定义见下）。

**视觉 Token 显著性分数 $p_i$ 的计算（含备选方案）**：

优先方案（若 ViT 支持 `output_attentions=True` 且存在 CLS token）：

$$
p_i = \frac{1}{N_h} \sum_{h=1}^{N_h} A_{h,\,0,\,i}
\quad \text{（CLS token 对 patch } i \text{ 的各头平均注意力）}
$$

**备选方案 1**（若无 CLS token，如 SigLIP 架构）：
用每个 patch 被所有其他 patch 关注的总量（"被注意度"）：

$$
p_i = \frac{1}{N_h N_p} \sum_{h=1}^{N_h} \sum_{k=1}^{N_p} A_{h,\,k,\,i}
$$

**备选方案 2**（若 ViT 注意力权重完全不可获取）：
用 ViT 最后一层输出激活的 L2 范数作为显著性代理：

$$
p_i = \frac{\|z_i^{\text{ViT}}\|_2}{\sum_{k=1}^{N_p} \|z_k^{\text{ViT}}\|_2}
$$

三种方案下，$p_i \in (0, 1)$，$\sum_i p_i = 1$，
后续所有公式对 $p_i$ 的使用方式完全相同，
仅计算来源不同。实现前须运行附录 A 的检查脚本确认实际可用接口。

**Token 分组**：
利用 Qwen3-VL 的 `image_grid_thw` 元信息直接定位视觉 token 位置范围，
将 $N$ 个 token 划分为三组：

$$
\mathcal{T}_\text{lang}, \quad
\mathcal{T}_\text{sal} = \{i \in \mathcal{T}_\text{vis} : p_i \geq p_{(K)}\}, \quad
\mathcal{T}_\text{bg} = \mathcal{T}_\text{vis} \setminus \mathcal{T}_\text{sal}
$$

其中 $p_{(K)}$ 为第 $K$ 大显著性分数，
$K = \lceil 0.2 \cdot N_p \rceil$（top-20%）。

---

## 3.3 Stage 1: 共享统计量计算

所有统计量在**原始激活 $X_l$** 上一次性计算，
后续预平滑操作（Stage 2）不影响这些数值。

**各组激活能量**：

$$
e_m^{(l)} = \frac{1}{N_m \cdot d_\text{in}}
\sum_{i \in \mathcal{T}_m} \|x_i^{(l)}\|_2^2,
\quad m \in \{\text{lang},\, \text{sal},\, \text{bg}\}
$$

**命题 3.1（激活能量正比于量化误差期望贡献）**：
设均匀量化器误差元素独立，方差为 $\Delta^2/12$。
对单 token $x_i$，其输出空间的期望量化误差为：

$$
\mathbb{E}\bigl[\|(W - \hat{W})x_i\|_2^2\bigr]
= \frac{\Delta^2 d_\text{out}}{12} \cdot \|x_i\|_2^2
$$

*证明*：$\mathbb{E}[\|(W-\hat{W})x_i\|_2^2]
= \mathrm{tr}(\mathbb{E}[EE^T] x_i x_i^T)$，
由误差元素 i.i.d.，$\mathbb{E}[EE^T] = \frac{\Delta^2}{12} d_\text{out} I$，
代入即得。$\square$

因此子组 $m$ 的期望误差贡献 $\mathcal{E}_m \propto e_m^{(l)}$，
加权系数应正比于激活能量。

**A 模块的 Hessian 权重系数**
（归一化使 $\sum_m \alpha_m = 3$，保持与原始 GPTQ Hessian 同量级）：

$$
\alpha_m^{(l)} = \frac{3\, e_m^{(l)}}{
  e_\text{lang}^{(l)} + e_\text{sal}^{(l)} + e_\text{bg}^{(l)}}
$$

**B/C 模块的激活幅度权重系数**
（归一化使最大组系数为 1，用于激活幅度加权）：

$$
a_m^{(l)} = \frac{e_m^{(l)}}{
  \max(e_\text{lang}^{(l)},\, e_\text{sal}^{(l)},\, e_\text{bg}^{(l)})}
$$

> **量纲说明**：$\alpha_m$ 归一化到 3 以匹配 Hessian 矩阵的迹量级；
> $a_m$ 归一化到 1 以作为相对幅度权重，
> 加权均值公式中的分母 $\sum_i w_i$ 自动完成最终归一化。
> 两套系数均来源于"该组 token 对量化误差的相对贡献"，
> 物理含义一致，仅归一化约束不同。

**显著性加权激活幅度**（B 和 C 共用）：

$$
\bar{x}_j^{(l)} =
\frac{\displaystyle\sum_{i=1}^{N} w_i^{(l)}\, |X_{i,j}^{(l)}|}
     {\displaystyle\sum_{i=1}^{N} w_i^{(l)}},
\qquad
w_i^{(l)} =
\begin{cases}
  a_\text{lang}^{(l)} & i \in \mathcal{T}_\text{lang} \\
  a_\text{sal}^{(l)} \cdot p_i & i \in \mathcal{T}_\text{sal} \\
  a_\text{bg}^{(l)}  & i \in \mathcal{T}_\text{bg}
\end{cases}
$$

---

## 3.4 Stage 2 [创新点 C]: 模态感知激活预平滑

**动机**：视觉显著 token 的激活在某些通道存在显著离群值，
导致子组 Gram 矩阵 $G_\text{sal}$ 对角元素方差极大，
后续加权 Hessian（Stage 3）的条件数恶化，Cholesky 分解数值不稳定。
Stage 2 先消除通道间幅度不均匀（通道维度校正），
Stage 3 再校正 token 组间重要性差异（token 维度校正），
两者在操作对象上严格正交，互不干扰。

**逐通道自适应迁移强度**：

$$
\rho_j^{(l)} = \frac{\bar{x}_j^{(l)}}{\bar{x}_j^{(l)} + \max_k |W_{l,\,k,\,j}|}
\;\in\; (0,\, 1)
$$

$\rho_j \to 1$：该通道激活远大于权重，离群值激进迁移到权重侧；
$\rho_j \to 0$：权重主导，迁移保守。

**缩放因子**（由原始 $X_l$ 和 $W_l$ 计算，不依赖 Stage 3 的结果）：

$$
s_j^{(l)} =
\Bigl[\bar{x}_j^{(l)}\Bigr]^{\rho_j^{(l)}}
\cdot
\Bigl[\max_k |W_{l,\,k,\,j}|\Bigr]^{-(1 - \rho_j^{(l)})}
$$

**等价变换**（输出数值完全不变）：

$$
\tilde{X}_l = X_l \cdot \mathrm{diag}(s^{(l)})^{-1},
\qquad
\tilde{W}_l = \mathrm{diag}(s^{(l)}) \cdot W_l
$$

**命题 3.2（条件数改善）**：
平滑变换后，$\tilde{G}_\text{sal}$ 的对角元素方差满足：

$$
\mathrm{Var}\!\bigl(\mathrm{diag}(\tilde{G}_\text{sal})\bigr)
\;\leq\;
\left(\frac{\max_j s_j}{\min_j s_j}\right)^{-2}
\cdot
\mathrm{Var}\!\bigl(\mathrm{diag}(G_\text{sal})\bigr)
$$

实验验证见附录 A（逐层条件数对比图）。

---

## 3.5 Stage 3 [创新点 A]: 模态感知加权 Hessian 量化重建

**动机**：原始 GPTQ 的 $H_l = \frac{2}{N} X_l^T X_l$
对所有 token 等权，忽视各组对量化误差的差异贡献。
由命题 3.1，$\alpha_m \propto e_m$ 是有理论依据的自适应系数，
而非经验超参。

**平滑后的子组归一化 Gram 矩阵**：

$$
\tilde{G}_m^{(l)} = \frac{1}{N_m}
\sum_{i \in \mathcal{T}_m} \tilde{x}_i^{(l)}\, {\tilde{x}_i^{(l)}}^T
$$

**MoLAQ 加权 Hessian**
（作用于平滑激活 $\tilde{X}_l$；
权重系数 $\alpha_m^{(l)}$ 由**原始** $X_l$ 的 $e_m$ 决定，
保证 C 与 A 的独立性）：

$$
\boxed{
H_l^\text{MoLAQ} = \frac{2}{3}\Bigl(
  \alpha_\text{lang}^{(l)}\, \tilde{G}_\text{lang}^{(l)}
+ \alpha_\text{sal}^{(l)}\,  \tilde{G}_\text{sal}^{(l)}
+ \alpha_\text{bg}^{(l)}\,   \tilde{G}_\text{bg}^{(l)}
\Bigr)
}
$$

**正定性保证**：三组 Gram 矩阵均半正定，正系数线性组合亦然。
在校准集充分（$N \gg d_\text{in}$）时严格正定。
退化情形施加对角正则化：

$$
H_l^\text{MoLAQ} \;\leftarrow\;
H_l^\text{MoLAQ} + \lambda\, \mathrm{tr}(H_l^\text{MoLAQ}) \cdot I,
\quad \lambda = 10^{-3}
$$

**与 GPTQ 的接口关系**：
仅替换 Hessian 收集阶段，Cholesky 分解与逐列权重更新循环完全复用。

**量化求解与权重恢复**：
将 $H_l^\text{MoLAQ}$ 代入 GPTQ Cholesky 逐列更新，
对 $\tilde{W}_l$ 执行 INT4 量化，得 $\hat{\tilde{W}}_l$。
部署时通过等价变换的逆恢复：

$$
\hat{W}_l^{(A)} = \mathrm{diag}(s^{(l)})^{-1} \cdot \hat{\tilde{W}}_l
$$

---

## 3.6 Stage 4 [创新点 B]: 显著 Token 驱动的缩放搜索

**动机**：AWQ 的搜索统计量
$\bar{x}_j^\text{AWQ} = \mathrm{mean}(|X_{:,j}|)$ 对全 token 等权，
大量低激活的背景视觉 token 稀释了统计量，
导致对语言 token 和显著视觉 token 关键通道的缩放保护不足。
量化步长与激活强度匹配的理论要求（见 §3.1 中
$\mathcal{E}_j \approx \frac{\Delta_j^2}{12} \mathbb{E}[X_{:,j}^2]$）
指向以 $\bar{x}_j^{(l)}$（Stage 1 已计算）替代全 token 均值。

**A 与 B 的串联关系（必须明确）**：

在完整 MoLAQ 流水线（$+$A$+$B$+$C）中，
**B 的 AWQ scaling 搜索以 A 的量化输出 $\hat{W}_l^{(A)}$ 为起点**，
对其执行 per-channel 显著性加权缩放修正，
即 C $\to$ A $\to$ B 为固定串联关系，B 是 A+C 结果上的后处理。

在消融实验的独立配置中：
- **$+$B only**：跳过 A/C，对原始浮点权重 $W_l$
  直接执行显著性加权 AWQ 搜索（即 $\hat{W}_l^{(A)}$ 替换为 $W_l$）；
- **$+$A only**：跳过 B/C，$\hat{W}_l^{(A)}$ 直接作为最终量化权重；
- **$+$A$+$C**：跳过 B，$\hat{W}_l^{(A)}$ 直接输出。

以上三种配置在实现上对应**不同代码路径**，
须在主调脚本中通过 `--enable_A`、`--enable_B`、`--enable_C`
三个开关分支控制，而非在单一流水线中内联。

**缩放因子搜索**：

$$
s_j^{*,(l)} = \arg\min_{s_j}
\Bigl\|
  Q\!\bigl(\hat{W}_l^{(A)} \cdot \mathrm{diag}(s)\bigr)
  \cdot \mathrm{diag}(s)^{-1} X_l
  - \hat{W}_l^{(A)} X_l
\Bigr\|_F^2
$$

用显著性加权均值 $\bar{x}_j^{(l)}$ 作为搜索统计量：

$$
s_j^{*,(l)} \approx \bigl[\bar{x}_j^{(l)}\bigr]^{t^*},
\qquad
t^* = \arg\min_{t \in [0,1]} \mathcal{E}(t)
\quad \text{（20 步 grid search）}
$$

---

## 3.7 混合精度分配

完成 A/B/C 逐层执行后，计算各层的**比特切换边际代价**：

$$
\delta_l = \sigma_l \cdot \bigl[\Delta_l(8) - \Delta_l(4)\bigr]
$$

其中层级敏感度 $\sigma_l = \mathrm{tr}(H_l^\text{MoLAQ})$，
$\Delta_l(b)$ 为层 $l$ 在比特数 $b$ 下的量化误差代理（RTN 误差，校准集估计），
两者在两个具体 bit 数（$b=4$ 和 $b=8$）下分别计算，
**$\Delta_l$ 不是决策变量 $b_l$ 的抽象函数**。

**贪心背包求解**：
$\delta_l$ 最小的层，降至 INT4 损失最小，优先分配 INT4；
$\delta_l$ 最大的层，对精度最敏感，保留 INT8。
在给定平均 bit 预算 $\bar{b}$（如 4.5 bits/param）约束下：

$$
\text{按 } \delta_l \text{ 升序排列各层，依次分配 } b_l = 4，
\text{直到 } \frac{\sum_l |W_l| \cdot b_l}{\sum_l |W_l|} \leq \bar{b}
$$

输出 `configs/molaq_bits_qwen3vl2b.json`（`{layer_name: bit_width}`），
供 vLLM 的 `awq_marlin`（INT4）和 `w8a8-fp8`（INT8）kernel 加载。

---

## 3.8 消融实验设计

六组配置全部在 **W4A16** 精度下运行，
在 Qwen3-VL-2B/4B 上评测 MMStar、MMMU、TextVQA：

| 配置 | C | A | B | 代码路径说明 |
|------|:---:|:---:|:---:|------|
| GPTQ-baseline | ✗ | ✗ | ✗ | 原始等权 Hessian，无 scaling |
| $+$C only | ✓ | ✗ | ✗ | 仅模态感知预平滑，后接等权 Hessian GPTQ |
| $+$A only | ✗ | ✓ | ✗ | 加权 Hessian，无预平滑，无显著 scaling |
| $+$B only | ✗ | ✗ | ✓ | 对原始浮点 $W_l$ 执行显著性加权 AWQ |
| $+$A$+$C | ✓ | ✓ | ✗ | 预平滑 + 加权 Hessian（完整重建路径） |
| MoLAQ ($+$A$+$B$+$C) | ✓ | ✓ | ✓ | 完整方法，C→A→B 串联 |

> 消融逻辑：
> $+$C only 与 $+$A only 的差值 = C 对 Hessian 条件数改善的独立贡献；
> $+$A$+$C 与 MoLAQ 的差值 = B 的 post-hoc scaling 修正增益；
> $+$B only 与 MoLAQ 的差值 = A+C 的权重重建改善必要性；
> GPTQ-baseline 与 $+$A only 的差值 = 加权 Hessian 相对等权 Hessian 的核心贡献。

---

## 附录 A: 实现前验证项

### A.1 ViT Attention 可获取性检查

```python
# 目的：确认 Qwen3-VL ViT 的 attention 接口，
# 决定 p_i 使用优先方案还是备选方案
source ~/vllm/bin/activate
python - << 'EOF'
from transformers import Qwen3VLForConditionalGeneration
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/lml/models/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cpu"
)

# 尝试获取 ViT attention
try:
    # Qwen3-VL visual 模块的调用方式
    pixel_values = torch.zeros(1, 3, 336, 336, dtype=torch.bfloat16)
    out = model.visual(pixel_values, output_attentions=True)
    attns = out.attentions  # list of [batch, heads, seq, seq]
    print(f"[OK] ViT attention available")
    print(f"     num_layers: {len(attns)}")
    print(f"     last layer shape: {attns[-1].shape}")
    # 检查 CLS token（如果存在，attn[:, :, 0, :] 是 CLS->patch 注意力）
    seq_len = attns[-1].shape[-1]
    print(f"     seq_len: {seq_len} ({'CLS+patches' if seq_len == 577 else 'patches only, no CLS'})")
except Exception as e:
    print(f"[FALLBACK] ViT attention not available: {e}")
    print("  -> Will use activation L2 norm as saliency proxy")
EOF
```

根据输出结果选择 §3.2 中的对应 $p_i$ 计算方案。

### A.2 层名格式检查

```python
source ~/vllm/bin/activate
python - << 'EOF'
from transformers import Qwen3VLForConditionalGeneration
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/lml/models/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cpu"
)
# ViT 侧 Linear 层
vit_linears = [(n, m.weight.shape) for n, m in model.named_modules()
               if "visual" in n and isinstance(m, torch.nn.Linear)]
print("=== ViT Linear layers (first 8) ===")
for n, s in vit_linears[:8]:
    print(f"  {n}: {s}")

# LLM 侧 Linear 层
llm_linears = [(n, m.weight.shape) for n, m in model.named_modules()
               if "model.layers" in n and isinstance(m, torch.nn.Linear)]
print(f"\n=== LLM Linear layers (first 5 / total {len(llm_linears)}) ===")
for n, s in llm_linears[:5]:
    print(f"  {n}: {s}")
EOF
```

输出结果将确定 `image_grid_thw` token 边界解析逻辑
以及 hook 注册时的层名过滤规则，
供 `modal_stats.py` 的实现直接使用。