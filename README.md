# SignalReasoner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![verl](https://img.shields.io/badge/framework-verl-green.svg)](https://github.com/volcengine/verl)

**SignalReasoner** is a wireless-domain mathematical reasoning project that trains a
Qwen2.5-3B model via supervised fine-tuning (SFT) followed by reinforcement learning (RL),
using rule-based reward over `\boxed{}` answers. The data is derived from the
[WirelessMATHBench-XL](https://huggingface.co/datasets/XINLI1997/WirelessMATHBench-XL) and
[WirelessMathBench](https://huggingface.co/datasets/XINLI1997/WirelessMathBench) benchmarks,
with chain-of-thought reasoning traces distilled from DeepSeek-V3 for the SFT subset.

Built on [verl](https://github.com/volcengine/verl).

---

## Overview

| Stage | Script | Algorithm |
|-------|--------|-----------|
| SFT | `scripts/sft.sh` | full-parameter fine-tuning (FSDP) |
| RL | `scripts/grpo.sh` | GRPO |
| RL | `scripts/gmpo.sh` | GMPO (geometric-mean policy loss) |
| RL | `scripts/gspo.sh` | GSPO (sequence-mean policy loss) |
| Eval | `scripts/cal_metric.py` | distributed inference + Pass@1 |

The pipeline: **SFT** teaches the model to produce step-by-step chain-of-thought ending with a
`\boxed{...}` answer; **RL** then optimizes it with a rule-based reward over answer correctness.

---

## Repository structure

```
SignalReasoner/
├── README.md
├── data/
│   ├── sft_train.parquet     # SFT data (3,542 samples)
│   ├── sft_val.parquet       # SFT validation (55 samples)
│   ├── rl_train.parquet      # RL training (1,613 samples)
│   └── test.parquet          # evaluation (800 samples)
├── scripts/
│   ├── sft.sh                # SFT training
│   ├── grpo.sh               # GRPO training
│   ├── gmpo.sh               # GMPO training
│   ├── gspo.sh               # GSPO training
│   ├── cal_metric.py         # evaluation
│   ├── inspect_data.py       # inspect parquet datasets
│   └── filter_sft_data.py    # optional SFT token-length filtering
└── reward/
    └── wireless.py           # rule-based reward function
```

---

## Dataset

All data is included in the `data/` directory.

| File | # Samples | Purpose | Format |
|------|-----------|---------|--------|
| `sft_train.parquet` | 3,542 | supervised fine-tuning | `messages` |
| `sft_val.parquet` | 55 | SFT validation | `messages` |
| `rl_train.parquet` | 1,613 | RL training | verl PPO |
| `test.parquet` | 800 | evaluation | verl PPO |

### Formats

**SFT** (`sft_train.parquet`, `sft_val.parquet`) — conversational `messages` format:

```json
{
  "messages": [
    {"role": "user", "content": "**Background** ... Please reason step by step ..."},
    {"role": "assistant", "content": "Step-by-step reasoning ... \\boxed{...}"}
  ]
}
```

**RL / test** (`rl_train.parquet`, `test.parquet`) — verl PPO format:

```json
{
  "prompt": [{"role": "user", "content": "**Background** ... **Question** ..."}],
  "reward_model": {"style": "rule", "ground_truth": "\\boxed{...}"},
  "data_source": "XINLI1997/WirelessMATHBench-XL",
  "ability": "wireless_math",
  "extra_info": {
    "split": "train_rl",
    "question_id": 10907,
    "type": "fill_blank_75",
    "correct_answer": "\\boxed{...}"
  }
}
```

### Problem types

| Type | `rl_train` | `test` |
|------|-----------|--------|
| `fill_blank_25` | 176 | 98 |
| `fill_blank_50` | 325 | 160 |
| `fill_blank_75` | 439 | 218 |
| `fill_blank_100` | 389 | 191 |
| `MCQ` | 284 | 133 |
| **Total** | **1,613** | **800** |

---

## Installation

1. Install [verl](https://github.com/volcengine/verl) and its dependencies.
2. Clone this repo into (or alongside) your verl checkout.
3. Install Python deps:

```bash
pip install datasets transformers accelerate pandas torch
```

---

## Usage

### 1. Supervised fine-tuning

```bash
bash scripts/sft.sh
```

Key variables (override via env):

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_PATH` | `checkpoints/qwen_3b_base` | base model checkpoint |
| `TRAIN_DATA` | `data/sft_train.parquet` | SFT data |
| `VAL_DATA` | `data/sft_val.parquet` | validation data |
| `TOTAL_EPOCHS` | `1` | epochs |

### 2. Reinforcement learning

```bash
bash scripts/grpo.sh    # GRPO
bash scripts/gmpo.sh    # GMPO
bash scripts/gspo.sh    # GSPO
```

Common overrides:

```bash
MODEL_PATH=checkpoints/qwen_3b_base_sft TOTAL_EPOCHS=20 ROLLOUT_N=4 bash scripts/grpo.sh
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_PATH` | `checkpoints/qwen_3b_base_sftnew490` | policy model (post-SFT) |
| `ROLLOUT_N` | `4` | samples per prompt |
| `TOTAL_EPOCHS` | `20` | training epochs |

### 3. Reward function

`reward/wireless.py` implements a rule-based reward (`compute_score`) that extracts all
`\boxed{...}` answers from the model output and compares them against the ground truth
(after whitespace normalization):

| Case | Reward |
|------|--------|
| exact one-to-one positional match | `1.0` |
| some (not all) answers matched | `0.2` |
| `\boxed{}` present but nothing correct | `0.1` |
| no `\boxed{}` at all | `0.0` |

The strict one-to-one matching prevents the "membership reward-hack" where a model dumps many
boxed answers hoping one is correct.

### 4. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 scripts/cal_metric.py
```

Edit `model_path` (line ~67) inside `cal_metric.py` to point at your checkpoint. Reports
overall Pass@1 and per-type accuracy on `data/test.parquet`.

---

## Citation

```bibtex
@misc{SignalReasoner,
  author = {<AUTHORS>},
  title  = {SignalReasoner: Wireless-Domain Mathematical Reasoning via SFT and RL},
  year   = {2025},
  url    = {<REPO_URL>}
}
```

---

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl). Datasets sourced from
[WirelessMATHBench-XL](https://huggingface.co/datasets/XINLI1997/WirelessMATHBench-XL),
[WirelessMathBench](https://huggingface.co/datasets/XINLI1997/WirelessMathBench), and
[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT).
