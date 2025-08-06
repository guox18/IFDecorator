#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail
ray stop
sleep 5
# ------------------------------
# Environment Configuration
# ------------------------------
setup_env() {
    export VERL_PPO_LOGGING_LEVEL='DEBUG'
    export VLLM_ATTENTION_BACKEND="XFORMERS"
    export VLLM_USE_MODELSCOPE="False"
    export WANDB_CONFIG_DIR="$HOME/.config/wandb"
}
setup_env

# ------------------------------
# Conda Environment Setup
# ------------------------------
activate_conda() {
    # Hard Code: InternLM3-8B支持
    workspace/custom_verl/soft_link.sh
    source public/verl_env/verl_env/bin/activate workspace/custom_verl/verl
}
activate_conda

# ------------------------------
# Path Configuration
# ------------------------------
setup_path() {
    # 如果断点续训的话, 这里需要修改
    # YYMMDD=$(date +%Y-%m-%d)
    # HHMMSS=$(date +%H-%M-%S)
    YYMMDD="2025-05-03"
    HHMMSS="10-46-59"

    CUSTOM_CODE_DIR="workspace/custom_verl/verl"
    VERL_DIR="workspace/custom_verl/verl"
    BASE_MODEL_PATH="hf_hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/afb2829595f63efa3548e9d6b13aa66e61aa0f38"
    TRAIN_DATA="./clean.parquet"
    VAL_DATA="./clean_testset.parquet"

    job_name=$(basename "${0}" .sh)
    experiment_name="qwen2_5-32b_cif-v3-${job_name}-${YYMMDD}-${HHMMSS}"
    project_name="verl_grpo_cif-v3-math_code_reason_filtered"
    
    OUTPUT_DIR="workspace/custom_verl/verl/models/${experiment_name}/${YYMMDD}/${HHMMSS}"
    mkdir -p "${OUTPUT_DIR}"
}
setup_path

# ------------------------------
# Main Training Command
# ------------------------------
run_training() {
    export PYTHONPATH="workspace/custom_verl/verl:${PYTHONPATH:-}"
    echo "PYTHONPATH: ${PYTHONPATH}"
    
    cd "${VERL_DIR}" || exit 1

    local num_gpus="${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}"
    local world_size="${WORLD_SIZE:-1}"
    local total_gpus=$((num_gpus * world_size))
    local prompt_max_tokens=2048 # 因为我自己控制 prompt token 时没有考虑 chat template 的长度
    local response_max_tokens=2048
    local per_gpu_max_tokens=$(( (prompt_max_tokens + response_max_tokens ) * 64 ))

    python3 -m verl.trainer.main_ppo \
        custom_reward_function.path="${CUSTOM_CODE_DIR}/rewards/cif.py" \
        custom_reward_function.name=cif_compute_score \
        +custom_valid_reward_function.path="${CUSTOM_CODE_DIR}/rewards/cif.py" \
        +custom_valid_reward_function.name=cif_compute_score \
        algorithm.adv_estimator="grpo" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.train_batch_size=64 \
        data.max_prompt_length=${prompt_max_tokens} \
        data.max_response_length=${response_max_tokens} \
        data.filter_overlong_prompts=True \
        data.truncation="error" \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        +actor_rollout_ref.actor.optim.lr_warmup_steps=15 \
        actor_rollout_ref.actor.optim.warmup_style=cosine \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        +actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name="vllm" \
        actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.temperature=1.0 \
        +actor_rollout_ref.rollout.val_temperature=0 \
        actor_rollout_ref.rollout.n=5 \
        +actor_rollout_ref.rollout.trust_remote_code=True \
        +actor_rollout_ref.rollout.n_val=1 \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.lam=0.95 \
        trainer.logger=wandb \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        +trainer.val_before_train=False \
        trainer.n_gpus_per_node="${num_gpus}" \
        trainer.resume_from_path=True \
        trainer.nnodes="${world_size}" \
        trainer.save_freq=20 \
        trainer.test_freq=2 \
        trainer.total_epochs=100 \
        reward_model.reward_manager="custom" "$@" ## 我自己来实现并行
    local training_status=$?

    # 显式传递训练状态
    if [ $training_status -ne 0 ]; then
        echo "Training failed with exit code $training_status"
        exit $training_status  # 退出码传递给全局
    fi
}


# ------------------------------
# Ray Cluster Setup
# ------------------------------
setup_ray() {
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
    export WORLD_SIZE=${WORLD_SIZE:-1}
    export RANK=${RANK:-0}

    echo "Ray Cluster Configuration:"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "RANK: $RANK"

    if [ "$WORLD_SIZE" -le 1 ]; then
        echo "Single node training, starting without Ray cluster..."
        run_training "$@"
    else
        if [ "$RANK" -eq 0 ]; then
            ray start --head \
                --node-ip-address="$MASTER_ADDR" \
                --port="$MASTER_PORT"
        else
            ray start --address "${MASTER_ADDR}:${MASTER_PORT}" \
                --block
        fi
        sleep 10
        run_training "$@"
    fi

    ray stop
}

# ------------------------------
check_permissions() {
    echo "Updating permissions for output directories..."
    chmod -R 777 "${VERL_DIR}/outputs" || true
    chmod -R 777 "${VERL_DIR}/wandb" || true
}

# ------------------------------
# Main Execution Flow
# ------------------------------
check_permissions
setup_ray "$@"
chmod -R 777 "${OUTPUT_DIR}" || true
echo "Training completed successfully: $(basename "${0}")"
exit 0