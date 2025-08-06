# 收集所有 hard_pool中的数据
import os

from config import pipeline_num
from utils import PROJECT_PATH, readjson, readjsonl, writejson, writejsonl

# 收集 数据
# 数据管道路径, 自动获取该管道下的数据
# v5_difficult下有 round_0 -> round_5 的文件夹, 对应五个轮次的数据
# 现在需要写一个python脚本, 收集所有 hard_pool中的数据, 并且根据是否 reasonable, 将数据收集到不同的文件中

pipeline_path = f"paper/experiments/datasets/{pipeline_num}"
path_to_pipeline = os.path.join(
    PROJECT_PATH,
    f"/cpfs01/shared/llm_ddd/guoxu/data/ifsandbox/pipeline{pipeline_num}/v5_difficult",
)

reasonable_dataset = []
unreasonable_dataset = []

for round_idx in range(6):
    if round_idx == 0:
        round_0_data = readjsonl(
            os.path.join(path_to_pipeline, f"round_{round_idx}", "hard_pool.jsonl")
        )
        for item in round_0_data:
            item["round"] = round_idx
            item["reasonable_ins"] = True
            item["prompt_wo_hard_constraints"] = item["formatted_ins"][
                "original_prompt"
            ]  # 为了后续数据处理统一
        reasonable_dataset.extend(round_0_data)
    else:
        path_to_round = os.path.join(path_to_pipeline, f"round_{round_idx}")
        path_to_hard_pool = os.path.join(path_to_round, "hard_pool.jsonl")
        data = readjsonl(path_to_hard_pool)
        for item in data:
            item["round"] = round_idx
            if item["reasonable_ins"]:
                reasonable_dataset.append(item)
            else:
                unreasonable_dataset.append(item)

import random

random.shuffle(reasonable_dataset)
random.shuffle(unreasonable_dataset)

writejsonl(
    reasonable_dataset,
    os.path.join(PROJECT_PATH, f"{pipeline_path}/cif/reasonable_dataset.jsonl"),
)
writejsonl(
    unreasonable_dataset,
    os.path.join(PROJECT_PATH, f"{pipeline_path}/cif/unreasonable_dataset.jsonl"),
)
