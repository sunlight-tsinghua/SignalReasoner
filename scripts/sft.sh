#!/usr/bin/env bash
# SFT | Open-CoT-Reasoning-Mini | FULL Fine-Tune (NO LoRA) | FSDP | 4x4090
# 
# 关键策略：
# 1. 关闭 LoRA (USE_PEFT=0)
# 2. 开启梯度检查点 (Gradient Checkpointing) 以节省激活显存
# 3. 微批次大小调小 (MICRO_BATCH_SIZE_PER_GPU=1 或 2)
# 4. 利用 FSDP 将权重/梯度/优化器状态分片到 4 张卡
#
# 运行命令：
#   bash run_qwen3b_cot_full_fsdp.sh 4 /path/to/save_dir
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_MODE=online
export HF_ENDPOINT=https://hf-mirror.com
set -xeuo pipefail

# ---- 用户可调参数（已针对全量微调优化） ----
MODEL_PATH=${MODEL_PATH:-checkpoints/qwen_3b_base}
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B}  # 模型路径（请修改为你的实际路径）
NPROC_PER_NODE=${NPROC_PER_NODE:-3}                       # 显卡数量
SP_SIZE=${SP_SIZE:-1}                           # 序列并行大小，设为1即不开启
USE_LIGER=${USE_LIGER:-1}                       # Liger 内核加速 (可选)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-36} 
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}  # 【关键】全量微调显存吃紧，建议从1开始
MAX_LENGTH=${MAX_LENGTH:-3072}                  # CoT 
DATA_MAX_LENGTH=${DATA_MAX_LENGTH:-1500}                  # 数据最大长度
LR=${LR:-3e-6}                                  # 全量微调学习率
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
PROJECT_NAME=${PROJECT_NAME:-cot-sft-full}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3b_cot_wireless}
SAVE_PATH=${SAVE_PATH:-checkpoints/qwen_3b_base_sftnew}  # 请改为你实际的保存路径
# 数据路径（请修改为你的实际路径）
TRAIN_DATA=${TRAIN_DATA:-data/sft_train.parquet}
VAL_DATA=${VAL_DATA:-data/sft_val.parquet}
# ---- 结束 ----

# 强制关闭 LoRA (全量微调)
USE_PEFT=0
extra_args=()
# 开启梯度检查点 —— 使用正确的参数名
extra_args+=("model.enable_gradient_checkpointing=True")
if [ "${USE_LIGER}" = "1" ]; then
    extra_args+=("model.use_liger=True")
fi

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    -m verl.trainer.sft_trainer \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.messages_key=messages \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.use_dynamic_bsz=True \
    data.max_length=${DATA_MAX_LENGTH} \
    data.max_token_len_per_gpu=${MAX_LENGTH} \
    data.truncation=error \
    data.pad_mode=no_padding \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=False \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=30 \
    trainer.test_freq=20 \
    trainer.seed=42 \
    "${extra_args[@]}" "$@"