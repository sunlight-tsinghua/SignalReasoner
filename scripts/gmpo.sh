#!/usr/bin/env bash
# GMPO | Qwen2.5-3B | Wireless | FSDP
# GMPO uses a geometric-mean aggregated ratio in the policy loss.

set -xeuo pipefail
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_MODE=online
export HF_ENDPOINT=https://hf-mirror.com
########################### user-adjustable ###########################

INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-checkpoints/qwen_3b_base_sftnew490}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-3}

train_batch_size=${TRAIN_BATCH_SIZE:-18}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-9}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-768}

ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-3840}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0.0}

clip_ratio=${CLIP_RATIO:-0.4}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.45}
rollout_n=${ROLLOUT_N:-4}

total_epochs=${TOTAL_EPOCHS:-20}

save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-200}

project_name=${PROJECT_NAME:-verl_wireless_new}
experiment_name=${EXPERIMENT_NAME:-qwen2.5_3b_sft_wireless_gmponew}

########################### DATA ###########################

DATA=(

    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False

    data.train_files=data/rl_train.parquet
    data.val_files=data/test.parquet

    data.train_batch_size=${train_batch_size}

    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}

    data.filter_overlong_prompts=True
    data.truncation='error'
)

########################### MODEL ###########################

MODEL=(

    actor_rollout_ref.model.path=${MODEL_PATH}

    actor_rollout_ref.model.use_remove_padding=True

    actor_rollout_ref.model.enable_gradient_checkpointing=True

)

########################### ACTOR ###########################

ACTOR=(

    # GMPO-specific policy loss settings
    actor_rollout_ref.actor.policy_loss.loss_mode=geo_mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio}

    actor_rollout_ref.actor.optim.lr=${actor_lr}

    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}

    actor_rollout_ref.actor.use_dynamic_bsz=True

    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}

    # GMPO uses KL loss
    actor_rollout_ref.actor.use_kl_loss=True

    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}

    actor_rollout_ref.actor.kl_loss_type=low_var_kl

    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}

    actor_rollout_ref.actor.fsdp_config.param_offload=True

    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

########################### ROLLOUT ###########################

ROLLOUT=(

    actor_rollout_ref.rollout.name=${INFER_BACKEND}

    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}

    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}

    actor_rollout_ref.rollout.n=${rollout_n}

    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True

    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}

    actor_rollout_ref.rollout.max_model_len=1280

    actor_rollout_ref.rollout.max_num_seqs=4
)

########################### REF ###########################

REF=(

    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True

    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}

    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

########################### REWARD ###########################

REWARD=(

    reward.custom_reward_function.path=reward/wireless.py

    reward.custom_reward_function.name=compute_score
)

########################### TRAINER ###########################

TRAINER=(

    trainer.balance_batch=True

    trainer.critic_warmup=0

    trainer.logger='["console","wandb"]'

    trainer.project_name=${project_name}

    trainer.experiment_name=${experiment_name}

    trainer.n_gpus_per_node=${NGPUS_PER_NODE}

    trainer.nnodes=${NNODES}

    trainer.save_freq=${save_freq}

    trainer.test_freq=${test_freq}

    trainer.total_epochs=${total_epochs}

    trainer.default_local_dir=checkpoints/qwen_3b_sft_wireless_gmponew
)

########################### EXTRA ###########################

EXTRA=(

    actor_rollout_ref.actor.strategy=fsdp2

    actor_rollout_ref.model.use_fused_kernels=True

    actor_rollout_ref.rollout.multi_stage_wake_up=True

    actor_rollout_ref.rollout.enable_chunked_prefill=True

    actor_rollout_ref.rollout.enforce_eager=True

    actor_rollout_ref.rollout.free_cache_engine=True
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
