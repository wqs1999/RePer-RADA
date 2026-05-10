# RPTF: Relative-Periodic Temporal Fusion

Implementation of the paper:
**"Time Series Forecasting with Relative-Periodic Temporal Encoding and Resolution-aware Dynamic Aggregation via Large Language Models"**

---

## Overview

RPTF is a large language model-based time series forecasting framework that integrates three key innovations:

1. **Relative-Periodic Joint Temporal Encoding** (Section 3.2)
2. **Multi-Resolution Decomposition** (Section 3.3)
3. **Resolution-Aware Dynamic Weight Aggregation** (Section 3.4)

---

## Key Modules

### 1. Relative-Periodic Joint Temporal Encoding
- **File**: `models/RPTF.py`, class `RelativePeriodicEncoding`
- Implements Equations (1)-(7) from the paper
- **Relative Encoding**: Learnable embedding for time step offsets from a reference index
- **Periodic Encoding**: Sinusoidal basis functions with learnable embeddings, fused via MLP-based weights
- **Joint Encoding**: Element-wise sum of relative and periodic components

### 2. Multi-Resolution Decomposition
- **File**: `models/RPTF.py`, method `nonoverlap_avg_pool`
- Implements Equations (8)-(12)
- **Non-overlapping average pooling** at multiple temporal scales
- Dataset-adaptive pooling kernels based on dominant period detection
- Default schedule for hourly data: `k_s ∈ {1, 4, 8, 24, 168}`

### 3. Resolution-Aware Dynamic Weight Aggregation
- **File**: `models/RPTF.py`, class `ResolutionAwareWeighting`
- Implements Equations (13)-(19)
- **Periodic-driven weighting**: MLP maps periodic context to scale importance scores
- **Softmax normalization** for differentiable scale selection
- **Multi-scale loss**: Combines scale-specific losses with aggregated loss
- **Entropy regularizer** prevents degenerate weight collapse

---

## File Structure

| File | Description |
|------|-------------|
| `models/RPTF.py` | Core model with all three innovation modules |
| `exp/exp_long_term_forecasting_rptf.py` | Training loop with multi-scale loss |
| `run.py` | Entry point (supports `RPTF` model option) |
| `run_rptf_etth1.sh` | Example script for ETTh1 |
| `run_rptf_all_datasets.sh` | Script for all benchmark datasets |

---

## Usage

### Run on ETTh1
```bash
bash run_rptf_etth1.sh
```

### Run on all datasets
```bash
bash run_rptf_all_datasets.sh
```

### Manual run
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_RPTF \
  --model RPTF \
  --data ETTh1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_pred_len 96 \
  --llm_ckp_dir ./llama \
  --learning_rate 0.0001 \
  --weight_decay 0.01 \
  --batch_size 32 \
  --train_epochs 20 \
  --pooling_kernels 1 4 8 24 168 \
  --period_list 24 \
  --max_offset_d 96 \
  --basis_dim 64 \
  --lambda_scale 0.5 \
  --eta_entropy 0.01 \
  --cosine \
  --gpu 0
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pooling_kernels` | `[1, 4, 8, 24, 168]` | Multi-resolution pooling kernels (Table 1) |
| `--period_list` | `[24]` | Detected periods for encoding |
| `--max_offset_d` | `96` | Max offset D for relative encoding |
| `--basis_dim` | `64` | Basis dimension d_b for periodic encoding |
| `--lambda_scale` | `0.5` | Loss weight λ for scale vs aggregated loss (Eq. 17) |
| `--eta_entropy` | `0.01` | Entropy regularization η (Eq. 18) |

### Per-Dataset Configuration

| Dataset | Dom. Period | Pooling Kernels | Period List |
|---------|------------|-----------------|-------------|
| ETTh1/h2/ECL | 24 | 1 4 8 24 168 | 24 |
| ETTm1/m2 | 96 | 1 16 32 96 192 | 96 |
| Weather | 144 | 1 24 48 144 288 | 144 |
| Solar | 144 | 1 24 48 144 288 | 144 |
| Traffic | 24 | 1 4 8 24 168 | 24 |

---

## Architecture

```
Input X ∈ R^{B×L×N}
    │
    ├── Normalize → X̃
    │
    ├── Multi-Resolution Decomposition (Eq. 8)
    │   ├── Scale 0: X̃_0 = X̃          (k=1,   L_0=L)
    │   ├── Scale 1: X̃_1 = Pool(X̃,4)   (L_1=L/4)
    │   ├── Scale 2: X̃_2 = Pool(X̃,8)   (L_2=L/8)
    │   ├── Scale 3: X̃_3 = Pool(X̃,24)  (L_3=L/24)
    │   └── Scale 4: X̃_4 = Pool(X̃,168) (L_4=L/168)
    │
    ├── For each scale s (Algorithm 1):
    │   ├── Temporal Encoding (Eq. 1-7)    → T^joint_s, e_s
    │   ├── Tokenize + Encode (Eq. 9)      → H_s
    │   ├── Concat(H_s, e_s⊗1) (Eq. 10)   → H'_s
    │   ├── LLM Forward (Eq. 12)           → Y_s
    │   └── Upsample to L_0 (Eq. 11)
    │
    ├── Dynamic Weighting (Eq. 13-14)
    │   └── w_s = softmax(f_s(e_per_s))
    │
    └── Weighted Aggregation (Eq. 15)
        └── Y = Σ_s w_s · Y_s

Training Loss (Eq. 16-19):
    L_total = λ·Σ_s L_s + (1-λ)·L_agg - η·Σ_s w_s·log(w_s)
```

---

## Citation

```bibtex
@article{rptf2025,
  title={Time Series Forecasting with Relative-Periodic Temporal Encoding 
         and Resolution-aware Dynamic Aggregation via Large Language Models},
  author={Wang, Haibin and Yu, Shenyang and Liu, Wenjie},
  year={2025}
}
```
