export TOKENIZERS_PARALLELISM=false
export DEBUG=false

### 8g
LOGDIR="./logs"
ACTUAL_WORKERS=8
MULTIPROCESS_WORKERS=$((ACTUAL_WORKERS * 20))

## pipeline 2 for large scale data
PIPELINENUM=2
INPUT_FILE="./rawdata/tagged_merge_v2.jsonl"

DATAPIPELINE="./pipeline$PIPELINENUM"

### circular round
circle_round=5

### data wash
python w0.1_quality_filter.py --input_file $INPUT_FILE \
--output_file $DATAPIPELINE/v1_seed/intermediate_data.jsonl \
--output_stats_file $DATAPIPELINE/v1_seed/intermediate_data_filtered_stat.jsonl \
--url_level 1 \
--max_workers $MULTIPROCESS_WORKERS 2>&1 | tee $LOGDIR/w0.1_quality_filter_$PIPELINENUM.log

##next step
python w0.2_quality_filter.py --input_file $DATAPIPELINE/v1_seed/intermediate_data.jsonl \
--output_file $DATAPIPELINE/v1_seed/high_quality_data.jsonl \
--output_stats_file $DATAPIPELINE/v1_seed/high_quality_data_filtered_stat.jsonl \
--url_level 1 \
--max_workers $MULTIPROCESS_WORKERS 2>&1 | tee $LOGDIR/w0.2_quality_filter_$PIPELINENUM.log

### instruction decomposition
python w1_decompose.py \
    --input_file $DATAPIPELINE/v1_seed/high_quality_data.jsonl \
    --seed_data_file $DATAPIPELINE/v1_seed/high_quality_data.jsonl \
    --output_file $DATAPIPELINE/v2_decomposed/high_quality_data_decomposed.jsonl \
    --temperature 0.0 \
    --max_out 12000 \
    --max_workers $MULTIPROCESS_WORKERS \
    --url_level 1  2>&1 | tee $LOGDIR/w1_decompose_$PIPELINENUM.log

### classify_constraints
python w2_classify_constraints.py \
    --input_file $DATAPIPELINE/v2_decomposed/high_quality_data_decomposed.jsonl \
    --seed_data_file $DATAPIPELINE/v1_seed/high_quality_data.jsonl \
    --output_file $DATAPIPELINE/v3_classified/high_quality_data_classified.jsonl \
    --temperature 0.0 \
    --max_out 4096 \
    --max_workers $MULTIPROCESS_WORKERS \
    --url_level 1  2>&1 | tee $LOGDIR/w2_classify_constraints_$PIPELINENUM.log

### add checklist
python w3_add_checklist.py \
    --input_file $DATAPIPELINE/v3_classified/high_quality_data_classified.jsonl \
    --seed_data_file $DATAPIPELINE/v1_seed/high_quality_data.jsonl \
    --output_file $DATAPIPELINE/v4_checklist/high_quality_data_checklist.jsonl \
    --temperature 0.0 \
    --max_out 8192 \
    --max_workers $MULTIPROCESS_WORKERS \
    --url_level 1  2>&1 | tee $LOGDIR/w3_add_checklist_$PIPELINENUM.log

## initial round
## tag difficulty
python w5_tag_difficulty.py \
    --round_idx 0 \
    --input_file $DATAPIPELINE/v4_checklist/high_quality_data_checklist.jsonl \
    --output_file $DATAPIPELINE/v5_difficult/round_{n}/difficulty_taged_ins.jsonl \
    --output_easy_pool $DATAPIPELINE/v5_difficult/round_{n}/easy_pool.jsonl \
    --output_hard_pool $DATAPIPELINE/v5_difficult/round_{n}/hard_pool.jsonl \
    --output_toohard_pool $DATAPIPELINE/v5_difficult/round_{n}/toohard_pool.jsonl \
    --url_level 1 \
    --easy_level 1 \
    --senior_level 2 \
    --roll_batch_size 8 \
    --max_out 8192  \
    --temperature 1.0 \
    --threshold_easy 0.5 \
    --threshold_hard 0.032 \
    --int_flag_enable_senior 0 \
    --max_workers $MULTIPROCESS_WORKERS  2>&1 | tee $LOGDIR/w5_tag_difficulty_0_$PIPELINENUM.log

### circular round (5 times)
### evol
for n in $(seq 1 $circle_round); do
    python w4_evol.py \
        --round_idx $n \
        --max_workers $MULTIPROCESS_WORKERS \
        --easy_pool_path $DATAPIPELINE/v5_difficult/round_$((n-1))/easy_pool.jsonl \
        --save_data_path $DATAPIPELINE/v4_checklist/round_{n}/evolved_ins.jsonl \
        --evol_exception_path $DATAPIPELINE/v4_checklist/round_{n}/evolved_ins_exception.jsonl \
        --llm_judge_as_reasonable_path $DATAPIPELINE/v4_checklist/round_{n}/evolved_ins_reasonable.jsonl \
        --llm_judge_as_unreasonable_path $DATAPIPELINE/v4_checklist/round_{n}/evolved_ins_unreasonable.jsonl \
        --evol_temperature 0.7 \
        --evol_max_tokens 12000 \
        --url_level 1 \
        --max_evol_retries 3 \
        --length_constraint_probability 0.5 \
        --num_evol_attempts $n \
        --num_hard_constraints_lower_bound $n \
        --num_hard_constraints_upper_bound $((n * 3))  2>&1 | tee $LOGDIR/w4_evol_$n_$PIPELINENUM.log

    # if input_file_extra is not empty, then recall more instructions
    python w5_tag_difficulty.py \
        --round_idx $n \
        --input_file $DATAPIPELINE/v4_checklist/round_{n}/evolved_ins_reasonable.jsonl \
        --input_file_extra $DATAPIPELINE/v4_checklist/round_{n}/evolved_ins_unreasonable.jsonl \
        --output_file $DATAPIPELINE/v5_difficult/round_{n}/difficulty_taged_ins.jsonl \
        --output_easy_pool $DATAPIPELINE/v5_difficult/round_{n}/easy_pool.jsonl \
        --output_hard_pool $DATAPIPELINE/v5_difficult/round_{n}/hard_pool.jsonl \
        --output_toohard_pool $DATAPIPELINE/v5_difficult/round_{n}/toohard_pool.jsonl \
        --url_level 1 \
        --easy_level 1 \
        --senior_level 2 \
        --roll_batch_size 8 \
        --max_out 8192  \
        --temperature 1.0 \
        --threshold_easy 0.5 \
        --threshold_hard 0.032 \
        --int_flag_enable_senior 0 \
        --max_workers $MULTIPROCESS_WORKERS  2>&1 | tee $LOGDIR/w5_tag_difficulty_$n_$PIPELINENUM.log
done

