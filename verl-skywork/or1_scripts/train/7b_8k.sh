#!/bin/bash
ray stop
export RAY_memory_usage_threshold=0.98
export RAY_memory_monitor_refresh_ms=0
export WANDB_DIR=/nfs100/zhangtianyu/wandb_logs
ray start --head --port=6379 --dashboard-port=8265 --temp-dir=/nfs100/zhangtianyu/ray_tmp

set -ex

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STATS_DIR="/nfs100/zhangtianyu/verl_ckpt/stats_${TIMESTAMP}"
mkdir -p $STATS_DIR

# export RAY_DEBUG=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nfs100/zhangtianyu/or1_data/train/train_7b_code.pkl \
    data.val_files=/nfs100/zhangtianyu/or1_data/eval/test_set_1221.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/nfs100/zhangtianyu/model/Qwen2.5-Coder-7B-Instruct-merged-v1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.adaptive_entropy.enabled=True \
    actor_rollout_ref.actor.adaptive_entropy.target_entropy=0.2 \
    actor_rollout_ref.actor.adaptive_entropy.max_ent_coef=0.005 \
    actor_rollout_ref.actor.adaptive_entropy.min_ent_coef=0 \
    actor_rollout_ref.actor.adaptive_entropy.delta_ent_coef=0.0001 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.67 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=yr \
    trainer.critic_warmup=0 \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=code_reasoning_grpo_coder \
    trainer.experiment_name=grpo_qwen_skywork_${TIMESTAMP} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.stats_path=$STATS_DIR \
    trainer.stats_save_freq=10 \
    trainer.default_local_dir=/nfs100/zhangtianyu/verl_ckpt/grpo_qwen_skywork_${TIMESTAMP} \
    trainer.resume_mode=disable \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}"
    