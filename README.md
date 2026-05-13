# MoLAQ: Modality & Layer-Aware Mixed-Precision Quantization for VLMs

## Overview

MoLAQ is a post-training quantization (PTQ) framework targeting Vision-Language Models (VLMs),
specifically Qwen3-VL series. It proposes a **cross-modal, layer-aware sensitivity-driven
mixed-precision bit allocation** under hardware resource constraints, deployable directly
on vLLM with INT4/INT8/FP8 kernels — no new number formats.

## Core Contribution

Unlike MBQ (coarse modal weighting) and Q-VLM (entropy proxy), MoLAQ introduces:

1. **Fine-grained visual saliency**: separates salient visual patch tokens (ViT attention top-K)
   from background tokens — a sub-modal sensitivity distinction not done before.
2. **Unified ViT+LLM optimization**: both encoder and decoder layers participate in the
   constrained bit-assignment problem, capturing cross-module error propagation.
3. **Principled 0-1 Knapsack allocation**: given a memory/bit budget B_total, solves
   min Σ σ_l · Δ_l(b_l) s.t. Σ |W_l|·b_l ≤ B_total, via greedy or DP.

## Environment

- WSL2, Ubuntu, Python 3.10+
- vLLM + llm-compressor + VLMEvalKit
- Target model: Qwen3-VL-2B/4B-Instruct

## Phases

| Phase | Goal |
|-------|------|
| 0 | Environment & baseline inference |
| 1 | FP16 / AWQ / GPTQ / FP8 baselines + eval |
| 2 | MoLAQ sensitivity estimation module |
| 3 | Bit allocation + quantization execution + vLLM export |
| 4 | Full experiments, ablation, paper-ready tables |

## Quick Start

```bash
source ~/vllm/bin/activate
cd /home/lml/MoLAQ

# Run sensitivity estimation
python scripts/run_molaq_sensitivity.py \
    --model /path/to/Qwen3-VL-2B \
    --calib_data flickr30k \
    --n_samples 128 \
    --output configs/sensitivity_qwen3vl2b.json

# Run bit allocation
python scripts/run_molaq_assign_bits.py \
    --sensitivity configs/sensitivity_qwen3vl2b.json \
    --budget_bits 4.5 \
    --output configs/molaq_bits_qwen3vl2b.json
```

## Citation

> [To be filled upon paper submission]

## License

MIT
