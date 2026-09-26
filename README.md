# ReMemorize

Code for the paper **"ReMemorize: Reward-Guided Adaptive Memory Management for Faithful Clinical Text Summarization"** (under review).

## Overview

ReMemorize augments a frozen large language model backbone with a trainable recurrent memory interface. The memory state is initialized during source encoding and evolves through gated updates during decoding, trained via group-relative reward-weighted candidate learning and a clipped score-weighted auxiliary gate objective.

## Requirements

Install dependencies:

```bash
pip install torch transformers accelerate bitsandbytes peft
pip install rouge-score bert-score sentence-transformers
pip install minicheck
```

## Data

- **MIMIC-IV-Ext-BHC**: access requires a credentialed PhysioNet account. See [PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/). Not redistributed here.
- **SOAP**: the `Datasets/` folder contains the processed SOAP splits used in this paper.

Format each dataset as `.jsonl` with fields `source` and `target`.

## Training

```bash
python train.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset mimic \
    --train_file Datasets/mimic_train.jsonl \
    --eval_file Datasets/mimic_val.jsonl \
    --training_ablation full \
    --output_dir runs/llama31_mimic \
    --num_epochs 2 \
    --group_size 4
```

Key arguments:
- `--training_ablation`: `full` | `no_grpo` | `no_rl` | `no_hallucination_penalty` | `no_aux_loss` — training-time variants (each requires a separate training run).
- `--load_in_4bit`: enable 4-bit quantization.
- `--freeze_llm`: freeze backbone, train memory module only.

## Evaluation

```bash
python evaluate.py \
    --checkpoint runs/llama31_mimic/best_ckpt \
    --eval_file Datasets/mimic_test.jsonl \
    --dataset mimic \
    --ablation full \
    --output_dir results/llama31_mimic
```

`--ablation`: `full` | `no_memory` | `fixed_gate` | `no_phase2_update` — inference-time variants evaluated on the same trained checkpoint.

## Baselines

General-purpose / domain-adapted / reasoning-optimised instruction-tuned models:

```bash
python evaluate_baselines.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --eval_file Datasets/mimic_test.jsonl \
    --dataset mimic \
    --output_dir results/baselines/mistral_mimic
```

Extractive baseline (MemSum):

```bash
python run_memsum_baseline.py --eval_file Datasets/mimic_test.jsonl --dataset mimic --output_dir results/baselines/memsum_mimic
```

Reasoning-oriented baseline (DeepSeek-R1):

```bash
python run_deepseek_r1_inference.py --eval_file Datasets/mimic_test.jsonl --dataset mimic --output_dir results/baselines/deepseek_r1_mimic
```

Supervised fine-tuning baseline (no memory, no RL):

```bash
python train_sft_baseline.py --model meta-llama/Llama-3.1-8B-Instruct --train_file Datasets/mimic_train.jsonl --output_dir runs/sft_mimic
```

GRPO-direct baseline (RL on generation directly, no memory module):

```bash
python train_dpo_baseline.py --model meta-llama/Llama-3.1-8B-Instruct --train_file Datasets/mimic_train.jsonl --output_dir runs/dpo_mimic
python evaluate_dpo_baseline.py --checkpoint runs/dpo_mimic/best_ckpt --eval_file Datasets/mimic_test.jsonl --output_dir results/dpo_mimic
```

## LLM-as-a-Judge Evaluation

```bash
python multi_judge_eval.py \
    --predictions results/llama31_mimic/predictions.jsonl \
    --output results/llama31_mimic/judge_scores
```

## Ablation Study

```bash
python run_ablations.py --checkpoint runs/llama31_mimic/best_ckpt --eval_file Datasets/mimic_test.jsonl --output_dir results/ablation_study
```

## Reward-Weight Sensitivity Analysis

```bash
python tune_reward_weights.py --train_file Datasets/mimic_train.jsonl --eval_file Datasets/mimic_val.jsonl --output_dir results/sensitivity
```

## Repository Structure

```
├── train.py                      # Main GRPO training loop
├── evaluate.py                   # Evaluation with ReMemorize (incl. inference-time ablations)
├── evaluate_baselines.py         # General-purpose / domain-adapted baseline evaluation
├── run_deepseek_r1_inference.py  # Reasoning-oriented baseline (DeepSeek-R1)
├── run_memsum_baseline.py        # Extractive baseline (MemSum)
├── train_sft_baseline.py         # Supervised fine-tuning baseline
├── train_dpo_baseline.py         # GRPO-direct (no-memory) baseline
├── evaluate_dpo_baseline.py      # Evaluation for the GRPO-direct baseline
├── run_ablations.py              # Ablation study runner
├── tune_reward_weights.py        # Reward-weight sensitivity analysis
├── dataset.py                    # Data loading
├── llm_backbone.py               # LLM wrapper
├── memory_manager.py             # Memory module (recurrent update + gate)
├── policy_manager.py             # Gate policy wrapper
├── reward_manager.py             # Reward computation
├── scorers.py                    # Metric scorers (ROUGE, BERTScore, MiniCheck)
├── multi_judge_eval.py           # LLM-as-a-judge evaluation
└── Datasets/                     # SOAP dataset splits
```
