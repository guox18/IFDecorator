#!/bin/bash

# 传入一个参数, 表示 gpu 总数, 添加提示信息

gpu_num=$1
echo "gpu_num: $gpu_num"

# Model paths for evaluation
models=()

eval_configs=(
    "run_qwen2_5-7b_cif-v3-run_qwen2_5-7b_cif-v3_0327_tree-based-reward_lr1e-6_bsz64_roll5_qwen72Bjudge_set12_hard1234_clean_nokl::1000::./models/qwen2_5-7b_cif-v3-run_qwen2_5-7b_cif-v3_0327_tree-based-reward_lr1e-6_bsz64_roll5_qwen72Bjudge_set12_hard1234_clean_nokl-2025-05-02-10-10-10/2025-05-02/10-10-10::qwen2_5-7b_rl_"
)

for eval_config in "${eval_configs[@]}"; do
    # 使用临时变量存储配置
    temp_config=$eval_config
    
    # 逐个提取字段
    modelname=${temp_config%%::*}
    temp_config=${temp_config#*::}
    
    modelfinalcheck=${temp_config%%::*}
    temp_config=${temp_config#*::}
    
    modelpath=${temp_config%%::*}
    stepname=${temp_config#*::}
    
    echo "Model Name: $modelname"
    echo "Final Checkpoint: $modelfinalcheck"
    echo "Model Path: $modelpath"434
    echo "Step Name: $stepname"
    
    cuda_id=0
    for ((i=100; i<=modelfinalcheck; i+=100)); do

        model_path="$modelpath/global_step_${i}/actor/${stepname}step${i}"
        models+=("$model_path")
        echo "Added model path: $model_path"
        # Example command (commented out)
        CUDA_VISIBLE_DEVICES=$cuda_id python rule_based_hacking_prob.py \
            --model_path "$model_path" \
            --model_name "$modelname" \
            --checkpoint_idx "$i" \
            --batch_size 1000 \
            --input_file "./probe.jsonl" \
            --tp 1 \
            --n_sample 8 \
            --temperature 1.0 \
            --dtype "bfloat16" \
            --output_length 8192 \
            --top_p 0.95 \
            --seed 42 &
        cuda_id=$((cuda_id+1))
        if [ $cuda_id -eq $gpu_num ]; then
            wait
            cuda_id=0
        fi
    done
done

wait