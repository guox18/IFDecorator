import os

# from tokenizer import count_tokens
from config import pipeline_num
from utils import PROJECT_PATH, readjson, readjsonl, writejson, writejsonl

pipeline_path = f"paper/experiments/datasets/{pipeline_num}"
path_to_reasonable_dataset = os.path.join(
    PROJECT_PATH, f"{pipeline_path}/cif/reasonable_dataset.jsonl"
)
reasonable_dataset = readjsonl(path_to_reasonable_dataset)

train_dataset = reasonable_dataset[: int(len(reasonable_dataset) - 512)]
test_dataset = reasonable_dataset[int(len(reasonable_dataset) - 512) :]

train_verl_format_dataset = []
prompt_tokens_counter = 0
number_of_too_long_prompt = 0
prompt_length_max = 2048

for item in train_dataset:
    verl_format_item = {
        "data_source": "cif_v2",
        "prompt": [{"content": item["prompt"], "role": "user"}],
        "ability": "if",
        "reward_model": {
            "ground_truth": {
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
                "complex_instruction_soft": item["prompt_wo_hard_constraints"],
                "constraints": item["formatted_ins"]["constraints"],
                "hard_constraints_checklist": item["formatted_ins"][
                    "hard_constraints_checklist"
                ],
            },
            "style": "rule",
        },
        "extra_info": {
            "index": item["id"],
            "question": item["prompt"],
            "prompt_tokens": (
                item["chosen_usage_1"]["prompt_tokens"]
                if "chosen_usage_1" in item
                else item["rejected_usage_1"]["prompt_tokens"]
            ),
            "pass_rate_1": item["pass_rate_1"],
            "split": "train",
        },
    }
    if verl_format_item["extra_info"]["prompt_tokens"] > prompt_length_max:
        number_of_too_long_prompt += 1
        continue
    prompt_tokens_counter += verl_format_item["extra_info"]["prompt_tokens"]
    train_verl_format_dataset.append(verl_format_item)

writejsonl(
    train_verl_format_dataset,
    os.path.join(
        PROJECT_PATH,
        f"{pipeline_path}/cif/verl_format/cif_tokens-{prompt_tokens_counter}_prompts-{len(train_verl_format_dataset)}_train_verl.jsonl",
    ),
)
print(f"train_verl_format_dataset 的 prompt_tokens 总数为: {prompt_tokens_counter}")
print(
    f"train_verl_format_dataset 中, 有 {number_of_too_long_prompt} 个 prompt 因为太长被丢弃"
)


test_verl_format_dataset = []
prompt_tokens_counter = 0

number_of_too_long_prompt = 0
for item in test_dataset:
    verl_format_item = {
        "data_source": "cif_v2",
        "prompt": [{"content": item["prompt"], "role": "user"}],
        "ability": "if",
        "reward_model": {
            "ground_truth": {
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
                "complex_instruction_soft": item["prompt_wo_hard_constraints"],
                "constraints": item["formatted_ins"]["constraints"],
                "hard_constraints_checklist": item["formatted_ins"][
                    "hard_constraints_checklist"
                ],
            },
            "style": "rule",
        },
        "extra_info": {
            "index": item["id"],
            "question": item["prompt"],
            "prompt_tokens": (
                item["chosen_usage_1"]["prompt_tokens"]
                if "chosen_usage_1" in item
                else item["rejected_usage_1"]["prompt_tokens"]
            ),
            "pass_rate_1": item["pass_rate_1"],
            "split": "test",
        },
    }

    if verl_format_item["extra_info"]["prompt_tokens"] > prompt_length_max:
        number_of_too_long_prompt += 1
        continue

    prompt_tokens_counter += verl_format_item["extra_info"]["prompt_tokens"]
    test_verl_format_dataset.append(verl_format_item)

print(f"test_verl_format_dataset 的 prompt_tokens 总数为: {prompt_tokens_counter}")
print(f"test_verl_format_dataset 中, 有 {number_of_too_long_prompt} 个 prompt 因为太长被丢弃")
writejsonl(
    test_verl_format_dataset,
    os.path.join(
        PROJECT_PATH,
        f"{pipeline_path}/cif/verl_format/cif_tokens-{prompt_tokens_counter}_prompts-{len(test_verl_format_dataset)}_test_verl.jsonl",
    ),
)


## verl还需要进一步把数据转为 parquet 格式, 需要再保存一份 parquet 格式的数据, 命名与jsonl数据格式相同, 后缀不同

parquet_train_save_path = os.path.join(
    PROJECT_PATH, f"{pipeline_path}/cif/verl_format/train_verl_format_dataset.parquet"
)
parquet_test_save_path = os.path.join(
    PROJECT_PATH, f"{pipeline_path}/cif/verl_format/test_verl_format_dataset.parquet"
)

import pandas as pd

train_df = pd.DataFrame(train_verl_format_dataset)
test_df = pd.DataFrame(test_verl_format_dataset)

train_df.to_parquet(parquet_train_save_path)
test_df.to_parquet(parquet_test_save_path)

## 重新读取一下 parquet 格式的数据, 确保数据没有问题

parquet_train_dataset = pd.read_parquet(parquet_train_save_path)
parquet_test_dataset = pd.read_parquet(parquet_test_save_path)

print(parquet_train_dataset.head())
print(parquet_test_dataset.head())
