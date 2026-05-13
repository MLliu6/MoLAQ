# MoLAQ 代码实现细则文档（修订终稿 v2）
# 面向算法研究员 · 对应理论指南 §3.1–§3.8


---


## 0. 总体说明


### 0.1 模块映射


| 理论章节 | 模块文件 | 核心职责 |
|----------|----------|----------|
| §3.2–3.3 | `molaq/stats/modal_stats.py` | Token 分组、激活收集、所有共享统计量 |
| §3.4 | `molaq/core/smooth.py` | 创新点 C：模态感知激活预平滑 |
| §3.5 | `molaq/core/weighted_hessian.py` | 创新点 A：加权 Hessian + GPTQ 求解 |
| §3.6 | `molaq/core/saliency_scaling.py` | 创新点 B：显著 token 驱动缩放搜索 |
| §3.7 | `molaq/assign/knapsack.py` | 混合精度贪心分配 |
| 主调 | `scripts/run_molaq.py` | 流水线编排 + `--enable_A/B/C` 开关 |


### 0.2 执行顺序（固定，不可调换）
校准集 forward pass（modal_stats.py）
↓ 产出：X_l, p_i, e_m, α_m, a_m, x̄_j（所有层，一次性）
Stage 2：smooth.py（创新点 C）
↓ 产出：X̃_l, W̃_l, s_j（逐层）
Stage 3：weighted_hessian.py（创新点 A）
↓ 产出：Ŵ_l^(A)（逐层量化权重）
Stage 4：saliency_scaling.py（创新点 B）
↓ 产出：Ŵ_l^(final)（逐层量化权重，含缩放修正）
knapsack.py
↓ 产出：configs/molaq_bits_qwen3vl2b.json


text


### 0.3 消融配置的代码路径


| 配置 | enable_C | enable_A | enable_B | B 的输入权重 |
|------|:--------:|:--------:|:--------:|-------------|
| GPTQ-baseline | ✗ | ✗ | ✗ | — |
| +C only | ✓ | ✗ | ✗ | — |
| +A only | ✗ | ✓ | ✗ | — |
| +B only | ✗ | ✗ | ✓ | 原始浮点 $W_l$ |
| +A+C | ✓ | ✓ | ✗ | — |
| MoLAQ (+A+B+C) | ✓ | ✓ | ✓ | $\\hat{W}_l^{(A)}$（A 的量化结果）|


**关键**：B 的输入权重在不同配置下不同——
`+B only` 时输入浮点 $W_l$，`MoLAQ` 时输入 $\\hat{W}_l^{(A)}$。
主调脚本须根据 flag 组合显式切换，不能在单一流水线中内联。


---


## 1. 实现前验证：ViT Attention 可获取性


**在编写任何算法代码之前，必须先运行此检查。**
结果决定 `modal_stats.py` 中 $p_i$ 的计算分支。


```python
# 目的：确认 Qwen3-VL ViT 的 attention 接口与输入格式
# 注意：必须经 processor 预处理后再调用 visual 模块，
#       不能直接传 raw pixel tensor（会在 patch embedding 层报维度错误）
source ~/vllm/bin/activate
python - << 'EOF'
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_PATH = "/home/lml/models/Qwen3-VL-2B-Instruct"
TEST_IMAGE  = "/mnt/e/BISHE_START/Datasets/MathVision/images/images/1.jpg"


processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
)


img = Image.open(TEST_IMAGE).convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text",  "text": "test"},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True,
    return_tensors="pt", add_generation_prompt=False,
)
pixel_values   = inputs["pixel_values"].to(torch.bfloat16)
image_grid_thw = inputs.get("image_grid_thw")


print(f"pixel_values shape : {pixel_values.shape}")
print(f"image_grid_thw     : {image_grid_thw}")


import inspect
sig = inspect.signature(model.visual.forward)
print(f"visual.forward params: {list(sig.parameters.keys())}")


try:
    with torch.no_grad():
        vit_out = model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            output_attentions=True,
        )
    attns   = vit_out.attentions           # list[Tensor]，每层一个
    seq_len = attns[-1].shape[-1]          # 最后一层的序列长度
    # ← 修正1（Bug-3）：取 image_grid_thw 得到 [t,h,w]，再逐元素相乘
    thw       = image_grid_thw[0]                  # Tensor[1]
    n_patches = int(thw[0] * thw[1] * thw[2])       # t × h × w[2][3]
    has_cls   = (seq_len == n_patches + 1)


    print(f"[OK] attention available, num_layers={len(attns)}")
    print(f"     last layer shape: {attns[-1].shape}")
    print(f"     n_patches={n_patches}, seq_len={seq_len}")
    print(f"     has CLS token: {has_cls}")
    print("=> 使用优先方案（CLS attention）" if has_cls
          else "=> 使用备选方案 1（被注意度 row_sum）")
except Exception as e:
    print(f"[FALLBACK] attention not available: {e}")
    print("=> 使用备选方案 2（ViT 输出激活 L2 范数）")
EOF
```


根据输出在 `modal_stats.py` 顶部设置：


```python
# modal_stats.py 顶部——根据上述脚本输出选择一项
SALIENCY_MODE = "cls_attn"   # 优先方案：有 CLS token
# SALIENCY_MODE = "row_sum"  # 备选方案 1：无 CLS，用被注意度
# SALIENCY_MODE = "act_norm" # 备选方案 2：无 attention
```


---


## 2. `molaq/stats/modal_stats.py`


### 2.1 职责


- 注册 forward hook 收集每层 $X_l$
- 提取 ViT 最后一层 attention（或激活），计算显著性分数 $p_i$
- 利用 `image_grid_thw` 完成 token 分组
- 计算并返回所有共享统计量：$e_m$、$\\alpha_m$、$a_m$、$\\bar{x}_j$


### 2.2 输入输出规格


```python
def collect_modal_stats(
    model: Qwen3VLForConditionalGeneration,
    dataloader: DataLoader,          # batch_size=1，含 pixel_values + image_grid_thw
    target_layer_names: List[str],   # 需要统计的层名列表（Linear 层）
    top_k_ratio: float = 0.2,        # 显著 token 比例
    saliency_mode: str = "cls_attn", # "cls_attn" | "row_sum" | "act_norm"
    device: str = "cuda",
) -> Dict[str, "LayerStats"]:
    """
    Returns:
        stats[layer_name] = LayerStats(
            X_raw       : Tensor [N, d_in]   原始激活（所有校准样本拼接）
            e_lang      : float              语言组激活能量
            e_sal       : float              显著视觉组激活能量
            e_bg        : float              背景视觉组激活能量
            alpha_lang  : float              A 模块 Hessian 权重（归一化到 3）
            alpha_sal   : float
            alpha_bg    : float
            a_lang      : float              B/C 模块幅度权重（归一化到 1）
            a_sal       : float
            a_bg        : float
            x_bar       : Tensor [d_in]      显著性加权激活幅度
            lang_mask   : Tensor [N] bool    语言 token 掩码
            sal_mask    : Tensor [N] bool    显著视觉 token 掩码
            bg_mask     : Tensor [N] bool    背景视觉 token 掩码
        )
    """
```


### 2.3 实现要点


**Hook 注册**：


```python
storage = {}   # {layer_name: List[Tensor]}


def make_hook(name):
    def hook(module, input, output):
        x = input.detach().float()       # 统一转 float32
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1]) # [B*seq, d_in]
        storage.setdefault(name, []).append(x.cpu())
    return hook


handles = []
for name, module in model.named_modules():
    if name in target_layer_names:
        handles.append(module.register_forward_hook(make_hook(name)))
```


**Token 分组（使用 image_grid_thw）**：


```python
def get_token_masks(
    input_ids:      Tensor,   # [1, seq_len]
    image_grid_thw: Tensor,   # [num_images, 3]，每行 (t, h, w)
    image_token_id: int,      # processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
        vis_mask  : [seq_len] bool
        lang_mask : [seq_len] bool
    """
    vis_mask  = (input_ids == image_token_id)
    lang_mask = ~vis_mask
    return vis_mask, lang_mask
```


**$p_i$ 的三种实现**：


```python
def compute_saliency(
    model, pixel_values, image_grid_thw,
    mode: str = "cls_attn",
) -> Tensor:
    """Returns: p [N_patches]，满足 sum(p) == 1"""
    if mode == "cls_attn":
        with torch.no_grad():
            out = model.visual(pixel_values, grid_thw=image_grid_thw,
                               output_attentions=True)
        last_attn = out.attentions[-1]          # [1, N_heads, seq, seq]
        cls_attn  = last_attn[0, :, 0, 1:]     # [N_heads, N_patches]
        p = cls_attn.mean(dim=0)


    elif mode == "row_sum":
        with torch.no_grad():
            out = model.visual(pixel_values, grid_thw=image_grid_thw,
                               output_attentions=True)
        last_attn = out.attentions[-1]          # [1, N_heads, N_p, N_p]
        p = last_attn.mean(dim=0).mean(dim=0)  # [N_p]


    elif mode == "act_norm":
        with torch.no_grad():
            out = model.visual(pixel_values, grid_thw=image_grid_thw)
        p = out.last_hidden_state.float().norm(dim=-1)  # [N_patches]


    p = p.float()
    p = p / (p.sum() + 1e-8)
    return p
```


**统计量计算**：


```python
def compute_stats_for_layer(
    X:        Tensor,   # [N, d_in]，float32
    lang_mask: Tensor,  # [N] bool
    sal_mask:  Tensor,  # [N] bool
    bg_mask:   Tensor,  # [N] bool
    p_sal:     Tensor,  # [N_sal]，显著 token 对应的 p_i（已归一化，长度=sal_mask.sum()）
) -> "LayerStats":


    def energy(mask):
        if mask.sum() == 0:
            return 1e-8
        return X[mask].pow(2).mean().item()   # 标量


    e = {
        "lang": energy(lang_mask),
        "sal":  energy(sal_mask),
        "bg":   energy(bg_mask),
    }
    e_sum = sum(e.values())


    alpha = {m: 3.0 * e[m] / e_sum for m in e}


    e_max = max(e.values())
    a = {m: e[m] / e_max for m in e}


    # 显著性加权激活幅度 x̄_j，shape [d_in]
    # ← 修正3（Logic-1）：N 取行数标量，不能用整个 X.shape
    N = X.shape[0]
    w = torch.zeros(N, dtype=torch.float32)
    w[lang_mask] = a["lang"]
    w[bg_mask]   = a["bg"]
    sal_indices  = sal_mask.nonzero(as_tuple=True)
    w[sal_indices] = a["sal"] * p_sal.to(w.device)


    w_sum = w.sum() + 1e-8
    x_bar = (w.unsqueeze(1) * X.abs()).sum(dim=0) / w_sum  # [d_in]


    return LayerStats(
        X_raw=X,
        e_lang=e["lang"], e_sal=e["sal"], e_bg=e["bg"],
        alpha_lang=alpha["lang"], alpha_sal=alpha["sal"], alpha_bg=alpha["bg"],
        a_lang=a["lang"], a_sal=a["sal"], a_bg=a["bg"],
        x_bar=x_bar,
        lang_mask=lang_mask, sal_mask=sal_mask, bg_mask=bg_mask,
    )
```


**数值注意事项**：
- 所有激活统一转 `float32`，避免 `bfloat16` 在 Gram 累加时精度损失
- `energy()` 的 `1e-8` 保护必须加，ViT 浅层背景 token 激活可能全零
- `p_sal` 长度必须严格等于 `sal_mask.sum()`，
  与 `sal_indices` 一一对应，传参前须验证


---


## 3. `molaq/core/smooth.py`


### 3.1 职责


创新点 C：逐通道等价缩放，消除视觉显著 token 激活的通道间幅度不均匀。


### 3.2 输入输出规格


```python
def compute_smooth_scale(x_bar: Tensor, W: Tensor) -> Tensor:
    """
    Args:
        x_bar : [d_in]，显著性加权激活幅度
        W     : [d_out, d_in]，原始浮点权重
    Returns:
        s     : [d_in]，逐通道缩放因子
    公式：
        ρ_j = x̄_j / (x̄_j + max_k|W_{k,j}|)
        s_j = x̄_j^{ρ_j} · max_k|W_{k,j}|^{-(1-ρ_j)}
    """


def apply_smooth(X: Tensor, W: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Returns: (X̃, W̃)
        X̃ = X · diag(s)^{-1}   [N, d_in]
        W̃ = diag(s) · W         [d_out, d_in]
    等价性：W̃ · X̃^T = W · X^T（单元测试验证）
    """
```


### 3.3 实现


```python
def compute_smooth_scale(x_bar: Tensor, W: Tensor) -> Tensor:
    w_max = W.abs().max(dim=0).values    # [d_in]
    x_bar = x_bar.clamp(min=1e-8)
    w_max = w_max.clamp(min=1e-8)
    rho   = x_bar / (x_bar + w_max)     # [d_in]，∈ (0, 1)
    # log-exp 计算，避免极端值下溢/上溢
    log_s = rho * x_bar.log() - (1 - rho) * w_max.log()
    s     = log_s.exp().clamp(max=1e4)  # [d_in]
    return s



def apply_smooth(X: Tensor, W: Tensor, s: Tensor):
    # W * s.unsqueeze(0)：对 W[i, j] *= s[j]，等价于 diag(s) · W 的行操作
    X_tilde = X / s.unsqueeze(0)        # [N, d_in]
    W_tilde = W * s.unsqueeze(0)        # [d_out, d_in]
    return X_tilde, W_tilde
```


**单元测试（必须通过再继续）**：


```python
# tests/test_smooth.py
def test_smooth_equivalence():
    torch.manual_seed(42)
    N, d_in, d_out = 64, 128, 256
    X = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in)
    x_bar = X.abs().mean(dim=0)


    s = compute_smooth_scale(x_bar, W)
    X_t, W_t = apply_smooth(X, W, s)


    out_orig   = X @ W.T
    out_smooth = X_t @ W_t.T
    max_err = (out_orig - out_smooth).abs().max().item()
    assert max_err < 1e-3, f"Smooth equivalence failed: max_err={max_err}"
    print(f"test_smooth_equivalence PASSED (max_err={max_err:.2e})")
```


---


## 4. `molaq/core/weighted_hessian.py`


### 4.1 职责


创新点 A：用模态加权 Hessian 替换 GPTQ 等权 Hessian，
执行 Cholesky 逐列量化，输出 $\\hat{W}_l^{(A)}$。


### 4.2 输入输出规格


```python
def compute_modal_hessian(
    X_tilde:    Tensor,      # [N, d_in]，平滑后激活
    alpha_lang: float,
    alpha_sal:  float,
    alpha_bg:   float,
    lang_mask:  Tensor,      # [N] bool
    sal_mask:   Tensor,      # [N] bool
    bg_mask:    Tensor,      # [N] bool
    damp:       float = 1e-3,
) -> Tensor:                 # [d_in, d_in]


def gptq_quantize(
    W_tilde:    Tensor,      # [d_out, d_in]
    H:          Tensor,      # [d_in, d_in]
    bits:       int   = 4,
    group_size: int   = 128,
    sym:        bool  = True,
) -> Tensor:                 # [d_out, d_in]


def quantize_layer_A(
    stats:    "LayerStats",
    W:        Tensor,        # [d_out, d_in]，原始浮点权重
    s:        Tensor,        # [d_in]，smooth 缩放因子
    enable_C: bool = True,
    bits:     int  = 4,
    group_size: int = 128,
) -> Tuple[Tensor, Tensor]:  # (Ŵ^(A), s)
```


### 4.3 Hessian 计算（Bug-1 修正：正则化公式统一）


```python
def compute_modal_hessian(X_tilde, alpha_lang, alpha_sal, alpha_bg,
                           lang_mask, sal_mask, bg_mask, damp=1e-3):
    d_in = X_tilde.shape[1]


    def gram(mask):
        if mask.sum() == 0:
            return torch.zeros(d_in, d_in, dtype=torch.float32)
        x_m = X_tilde[mask].float()
        # ← 修正2（Logic-1）：N_m 取行数标量
        N_m = x_m.shape[0]
        return x_m.T @ x_m / N_m          # [d_in, d_in]


    G_lang = gram(lang_mask)
    G_sal  = gram(sal_mask)
    G_bg   = gram(bg_mask)


    H = (2.0 / 3.0) * (
        alpha_lang * G_lang +
        alpha_sal  * G_sal  +
        alpha_bg   * G_bg
    )


    # Bug-1 修正：正则化公式与理论文档 §3.5 保持一致，使用 λ·tr(H)·I
    # 注：此处有意不用 tr(H)/d_in 的 per-元素版本，
    #     以保证代码与论文公式 H ← H + λ·tr(H)·I 完全对应，
    #     λ=1e-3 足以保证正定性而不过度压缩敏感层的 Hessian 比例。
    trace_H = H.trace().item()
    H = H + damp * trace_H * torch.eye(d_in, dtype=H.dtype, device=H.device)


    return H
```


### 4.4 GPTQ Cholesky 逐列更新（Bug-2 修正）


```python
def gptq_quantize(W_tilde, H, bits=4, group_size=128, sym=True):
    d_out, d_in = W_tilde.shape
    W_q = W_tilde.clone().float()


    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        # 正定性恢复：取标量维度
        d = H.shape[0]
        H = H + 1e-2 * H.trace() / d * torch.eye(d, dtype=H.dtype, device=H.device)
        L = torch.linalg.cholesky(H)


    H_inv = torch.cholesky_inverse(L)    # [d_in, d_in]


    for i in range(d_in):
        w_col   = W_q[:, i]              # [d_out]
        g_idx   = i // group_size
        g_start = g_idx * group_size
        g_end   = min(g_start + group_size, d_in)
        w_group = W_q[:, g_start:g_end]  # [d_out, group_size]


        scale = w_group.abs().max(dim=1, keepdim=True).values \\
                / (2**(bits-1) - 1)
        scale = scale.clamp(min=1e-8)


        if sym:
            w_col_q = (w_col / scale.squeeze()).round().clamp(
                -(2**(bits-1)), 2**(bits-1)-1
            ) * scale.squeeze()
        else:
            raise NotImplementedError("asymmetric quant not implemented")


        quant_err = w_col_q - w_col                  # [d_out]
        h_inv_row = H_inv[i, i+1:]                   # [d_in - i - 1]
        h_inv_ii  = H_inv[i, i].clamp(min=1e-8)


        W_q[:, i]    = w_col_q
        W_q[:, i+1:] -= torch.outer(quant_err, h_inv_row / h_inv_ii)


    return W_q
```


### 4.5 权重恢复（Logic-1 修正）


```python
def quantize_layer_A(stats, W, s, enable_C=True, bits=4, group_size=128):
    if enable_C:
        X_tilde, W_tilde = apply_smooth(stats.X_raw, W, s)
    else:
        X_tilde  = stats.X_raw.float()
        W_tilde  = W.float()
        # identity 缩放向量：d_in 维全一向量
        s = torch.ones(W.shape[1], dtype=torch.float32, device=W.device)


)


    H = compute_modal_hessian(
        X_tilde,
        stats.alpha_lang, stats.alpha_sal, stats.alpha_bg,
        stats.lang_mask, stats.sal_mask, stats.bg_mask,
    )


    W_tilde_q = gptq_quantize(W_tilde, H, bits=bits, group_size=group_size)


    # 恢复到原始空间：Ŵ^(A) = diag(s)^{-1} · Ŵ_tilde
    W_hat_A = W_tilde_q / s.unsqueeze(0)   # [d_out, d_in]


    return W_hat_A, s
```


**Sanity check（量化后必须运行）**：


```python
err = (W_hat_A - W.float()).abs()
print(f"Layer quant error: mean={err.mean():.4f}, max={err.max():.4f}")
# 正常范围：mean < 0.02，max < 0.5
# 若 mean > 0.1，检查正则化强度（damp 是否过小）


# 条件数对比（用于附录 A 实验数据收集）
# ← 修正5：X_raw.shape 取行数标量
H_gptq = (2.0 / stats.X_raw.shape[0]) * stats.X_raw.T @ stats.X_raw
kappa_gptq  = torch.linalg.cond(H_gptq.float()).item()
kappa_molaq = torch.linalg.cond(H.float()).item()
print(f"κ(H_GPTQ)={kappa_gptq:.1f}, κ(H_MoLAQ)={kappa_molaq:.1f}")
# 预期：κ(H_MoLAQ) < κ(H_GPTQ)（若反向，检查 C 的 smooth_scale 是否生效）
```


---


## 5. `molaq/core/saliency_scaling.py`


### 5.1 职责


创新点 B：用显著性加权统计量 $\\bar{x}_j$ 替代 AWQ 全 token 均值，
搜索最优 per-channel 缩放因子，执行 W4 量化。


### 5.2 输入输出规格


```python
def saliency_awq_quantize(
    W_input:    Tensor,   # [d_out, d_in]
                          # +B only → 原始浮点 W_l
                          # MoLAQ  → Ŵ_l^(A)
    X_orig:     Tensor,   # [N, d_in]，原始激活（始终用原始 X，不用 X̃）
    x_bar:      Tensor,   # [d_in]，显著性加权幅度（来自 modal_stats）
    bits:       int = 4,
    group_size: int = 128,
    n_grid:     int = 20,
) -> Tensor:              # [d_out, d_in]，Ŵ^(final)
```


### 5.3 实现


```python
def saliency_awq_quantize(W_input, X_orig, x_bar, bits=4,
                           group_size=128, n_grid=20):
    W_fp     = W_input.float()
    X_fp     = X_orig.float()
    x_bar_fp = x_bar.float().clamp(min=1e-8)


    # 若 N 较大（>4096），子采样降显存压力
    if X_fp.shape[0] > 4096:
        X_fp = X_fp[::4]


    best_err = float("inf")
    best_W_q = None


    for t in torch.linspace(0, 1, n_grid):
        s = x_bar_fp.pow(t.item()).clamp(min=1e-8)  # [d_in]


        W_scaled   = W_fp * s.unsqueeze(0)           # W · diag(s)
        W_scaled_q = rtn_quantize(W_scaled, bits=bits, group_size=group_size)
        W_q        = W_scaled_q / s.unsqueeze(0)     # 逆变换


        # P2-1 修正：误差参考点为 W_input（B 的参考基准），
        # 即衡量 B 相对于 A 输出的增量修正收益，而非相对于原始 W_l
        err = ((W_q @ X_fp.T) - (W_fp @ X_fp.T)).pow(2).mean().item()


        if err < best_err:
            best_err = err
            best_W_q = W_q


    return best_W_q



def rtn_quantize(W: Tensor, bits: int, group_size: int) -> Tensor:
    """RTN 量化，用于 AWQ grid search 的快速误差估计"""
    d_out, d_in = W.shape
    W_q = W.clone()
    for g in range(0, d_in, group_size):
        wg    = W[:, g:g+group_size]
        scale = wg.abs().max(dim=1, keepdim=True).values \\
                / (2**(bits-1) - 1)
        scale = scale.clamp(min=1e-8)
        W_q[:, g:g+group_size] = (
            (wg / scale).round()
            .clamp(-(2**(bits-1)), 2**(bits-1)-1)
            * scale
        )
    return W_q
```


---


## 6. `molaq/assign/knapsack.py`


### 6.1 职责


混合精度贪心分配：基于边际代价 $\\delta_l$ 排序，
在平均 bit 预算约束下决定每层用 INT4 还是 INT8。


### 6.2 实现


```python
def greedy_bit_allocation(
    layer_names:     List[str],
    param_counts:    Dict[str, int],   # {layer_name: 参数量（元素个数）}
    hessian_trace:   Dict[str, float], # {layer_name: tr(H^MoLAQ)}，即 σ_l
    delta_4:         Dict[str, float], # {layer_name: Δ_l(4)，RTN 误差}
    delta_8:         Dict[str, float], # {layer_name: Δ_l(8)，RTN 误差}
    budget_avg_bits: float = 4.5,
) -> Dict[str, int]:
    """
    Returns: {layer_name: bit_width}，bit_width ∈ {4, 8}


    核心公式：
        δ_l = σ_l · (Δ_l(8) - Δ_l(4))
        δ_l 越小 → 降到 INT4 精度损失越小 → 优先分配 INT4
        δ_l 越大 → 层对精度最敏感 → 保留 INT8


    注意：Δ_l(4) 和 Δ_l(8) 是在具体 bit 数下分别估计的标量，
    不是决策变量 b_l 的抽象函数。
    """
    total_params = sum(param_counts[l] for l in layer_names)
    budget_total = budget_avg_bits * total_params


    delta_margin = {
        l: hessian_trace[l] * (delta_8[l] - delta_4[l])
        for l in layer_names
    }


    assignment    = {l: 8 for l in layer_names}   # 初始全 INT8
    current_bits  = float(total_params * 8)


    # δ_l 升序：优先将损失最小的层翻转为 INT4
    for l in sorted(layer_names, key=lambda l: delta_margin[l]):
        bits_saved = param_counts[l] * 4             # INT8→INT4 节省
        if current_bits - bits_saved < budget_total:
            break                                    # 翻转后低于预算下界，停止
        assignment[l]  = 4
        current_bits  -= bits_saved


    achieved = current_bits / total_params
    n_int4   = sum(1 for b in assignment.values() if b == 4)
    print(f"[knapsack] budget={budget_avg_bits:.2f}b | "
          f"achieved={achieved:.3f}b | INT4={n_int4}/{len(layer_names)}")
    return assignment



def estimate_delta(W: Tensor, bits: int, group_size: int = 128) -> float:
    """用 RTN 误差估计 Δ_l(b)，不需要校准集前向传播"""
    W_q = rtn_quantize(W.float(), bits=bits, group_size=group_size)
    return (W_q - W.float()).pow(2).mean().item()
```


---


## 7. `scripts/run_molaq.py`（主调脚本骨架）


```python
#!/usr/bin/env python
"""
MoLAQ 主调脚本
用法：
  python scripts/run_molaq.py \\
      --model /home/lml/models/Qwen3-VL-2B-Instruct \\
      --calib_data /mnt/e/BISHE_START/Datasets/flickr30k/data \\
      --output /home/lml/models/Qwen3-VL-2B-MoLAQ \\
      --enable_A --enable_B --enable_C \\
      --budget_bits 4.5
"""
import argparse, json, os, torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from molaq.stats.modal_stats      import collect_modal_stats
from molaq.core.smooth            import compute_smooth_scale, apply_smooth
from molaq.core.weighted_hessian  import quantize_layer_A
from molaq.core.saliency_scaling  import saliency_awq_quantize
from molaq.assign.knapsack        import greedy_bit_allocation, estimate_delta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--calib_data",   required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--enable_A",     action="store_true")
    p.add_argument("--enable_B",     action="store_true")
    p.add_argument("--enable_C",     action="store_true")
    p.add_argument("--budget_bits",  type=float, default=4.5)
    p.add_argument("--bits",         type=int,   default=4)
    p.add_argument("--group_size",   type=int,   default=128)
    p.add_argument("--n_samples",    type=int,   default=128)
    return p.parse_args()



def main():
    args = parse_args()


    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True
    )


    # Stage 0+1：一次 forward pass 收集所有统计量
    all_stats = collect_modal_stats(model, ...)  # 传入 dataloader


    hessian_trace, delta_4_dict, delta_8_dict, param_counts = {}, {}, {}, {}


    for layer_name, module in get_linear_layers(model):
        stats = all_stats[layer_name]
        W     = module.weight.data.float()


        # Stage 2（C）+ Stage 3（A）
        if args.enable_C or args.enable_A:
            # ← 修正4（Logic-2）：三元表达式直接返回向量，不能在外面套[2]
            smooth_s = (compute_smooth_scale(stats.x_bar, W)
                        if args.enable_C
                        else torch.ones(W.shape[1], dtype=torch.float32, device=W.device))
            W_hat_A, smooth_s = quantize_layer_A(
                stats, W, smooth_s,
                enable_C=args.enable_C,
                bits=args.bits, group_size=args.group_size,
            )
        else:
            W_hat_A = W


        # Stage 4（B）
        if args.enable_B:
            # B 的输入权重：enable_A 时用量化结果，否则用浮点原始权重
            W_input_B = W_hat_A if args.enable_A else W
            W_final   = saliency_awq_quantize(
                W_input_B, stats.X_raw, stats.x_bar,
                bits=args.bits, group_size=args.group_size,
            )
        else:
            W_final = W_hat_A


        module.weight.data = W_final.to(module.weight.dtype)


        # 收集混合精度分配所需统计量
        from molaq.core.weighted_hessian import compute_modal_hessian
        H = compute_modal_hessian(
            stats.X_raw.float(),
            stats.alpha_lang, stats.alpha_sal, stats.alpha_bg,
            stats.lang_mask, stats.sal_mask, stats.bg_mask,
        )
        hessian_trace[layer_name] = H.trace().item()
        delta_4_dict[layer_name]  = estimate_delta(W, bits=4,
                                        group_size=args.group_size)
        delta_8_dict[layer_name]  = estimate_delta(W, bits=8,
                                        group_size=args.group_size)
        param_counts[layer_name]  = W.numel()


    # 混合精度分配（可选，budget_bits < 8 时生效）
    if args.budget_bits < 8.0:
        layer_names = list(all_stats.keys())
        assignment  = greedy_bit_allocation(
            layer_names, param_counts,
            hessian_trace, delta_4_dict, delta_8_dict,
            budget_avg_bits=args.budget_bits,
        )
        os.makedirs(args.output, exist_ok=True)
        with open(f"{args.output}/molaq_bits.json", "w") as f:
            json.dump(assignment, f, indent=2)


    # 保存
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, save_compressed=True)
    processor.save_pretrained(args.output)
    print(f"[MoLAQ] saved to {args.output}")



if __name__ == "__main__":
    main()
```


---


## 8. 测试与 Sanity Check 清单


实现完成后按顺序运行，全部通过再进行 MMStar 评测：


| 优先级 | 检查项 | 脚本位置 | 预期结果 |
|:------:|--------|----------|----------|
| P0 | ViT attention 可获取性 | §1 检查脚本 | 确定 `SALIENCY_MODE` |
| P0 | Smooth 等价变换 | `tests/test_smooth.py` | max_err < 1e-3 |
| P0 | 各层量化误差范围 | 主调脚本内置打印 | mean < 0.02，max < 0.5 |
| P1 | 条件数改善验证 | 主调脚本内置打印 | κ(H_MoLAQ) < κ(H_GPTQ) |
| P1 | Knapsack 分配方向 | `tests/test_knapsack.py` | 高 δ_l 层保留 INT8 |
| P1 | Sanity 推理 | 复用现有 sanity check | 生成结果语义正确 |
| P2 | MMStar 快速验证（50 样本）| VLMEvalKit | 对比 GPTQ-baseline 无显著下降 |



现在上面的内容应该可以是最终版了。我需要的是“在本对话中，像一名顶尖算法团队中研究LLM/VLM量化算法的顶尖算法研究员一样，遵循下面的指南和空间指令中的要求来开展MoLAQ量化算法的代码编写及验证工作。
在WSL2的(vllm) lml@LAPTOP-APTRGSIG:~/MoLAQ中进行代码库的保存，你要高效规划代码库的主要文件以及文件夹架构，并且要有README.md和一个log.md（记录每日的核心进展和操作），并且在规划后将本地的代码库上传到 [https://github.com/MLliu6/MoLAQ](https://github.com/MLliu6/MoLAQ) 代码库中。”的一个system prompt，给出完整的prompt使该需求可以正确传递（prompt是要包含上面的所有内容的，除此之外再加上你对于程序员的具体要求）。