"""
Tag difficulty module for instruction following experiments
"""

import argparse
import copy
import os
import random

from rich import print

from global_config import PROJECT_ROOT
from infer.apis import get_res_cycle
from infer.multiprocess_wrapper import (
    CheckpointManager,
    MultiProcessHandler,
    parallel_wrapper,
    preprocess_list,
)
from logs.logger import logger
from modules.enhance.check_instruction import (
    test_instruction_following_llm,
    test_instruction_following_strict,
)
from modules.utils import readjsonl, seed_everything, writejsonl

debug_value = os.environ.get("DEBUG", "false").lower() == "true"


def judge_item(item, args):
    # user_prompt = item['prompt']
    if item.get("prompt_wo_hard_constraints", None) is None:
        user_prompt = item["formatted_ins"]["original_prompt"]
    else:
        user_prompt = item.get("prompt_wo_hard_constraints", None)
    response = item["response"]
    instruction_id_list = item["instruction_id_list"]
    ins_kwargs = item["kwargs"]
    list_of_constraints = item["formatted_ins"]["constraints"]
    list_of_checklist = item["formatted_ins"]["hard_constraints_checklist"]

    if not test_instruction_following_strict(response, instruction_id_list, ins_kwargs):
        return False

    if not test_instruction_following_llm(
        user_prompt,
        response,
        list_of_constraints,
        list_of_checklist,
        model_name="qwen",
        max_tokens=1024,
        url_level=args.url_level,
    ):
        return False

    return True


def roll_out(item, args):
    roll_out_list = []
    user_prompt = item["prompt"]
    for i in range(args.roll_batch_size):
        new_item = copy.deepcopy(item)
        response = get_res_cycle(
            user_prompt,
            temperate=args.temperature,
            max_retry=2,
            url_level=args.url_level,
            max_tokens=args.max_out,
            return_extra_info=True,
        )
        if response is None:
            res = ""
            extra_info_list = [{}]
        else:
            res, extra_info_list = response
        new_item["response"] = res
        new_item["roll_usage"] = extra_info_list[0]
        roll_out_list.append(new_item)
    return roll_out_list


def roll_and_judge(item, args):
    """Process a batch of items with roll out and judge"""
    roll_outs = roll_out(item, args)

    # Judge results for each roll out
    judge_results = []
    for one_roll in roll_outs:
        judge_results.append(judge_item(one_roll, args))

    # randomly choose a chosen and rejected roll out for SFT or DPO

    positive_example = ""
    negative_example = ""

    true_indices = [i for i, result in enumerate(judge_results) if result]
    false_indices = [i for i, result in enumerate(judge_results) if not result]

    if true_indices:
        positive_example = roll_outs[random.choice(true_indices)]["response"]
        item[f"chosen_usage_{args.url_level}"] = roll_outs[random.choice(true_indices)][
            "roll_usage"
        ]
    if false_indices:
        negative_example = roll_outs[random.choice(false_indices)]["response"]
        item[f"rejected_usage_{args.url_level}"] = roll_outs[
            random.choice(false_indices)
        ]["roll_usage"]

    item[f"rejected_{args.url_level}"] = negative_example
    item[f"chosen_{args.url_level}"] = positive_example
    logger.info(f"One item pass rate is {sum(judge_results)/len(judge_results)}")
    item[f"pass_rate_{args.url_level}"] = sum(judge_results) / len(judge_results)

    return item


def tag_difficulty(input_list, args):
    """Tag difficulty with checkpoint support"""
    logger.info(f"args:{args}")
    processed_data = parallel_wrapper(
        preprocess_list(input_list, [args]),
        roll_and_judge,
        max_workers=args.max_workers,
        savebatch=10000,
    )
    # Write final results
    writejsonl(processed_data, args.output_file)
    return processed_data


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser(description="tag difficulty")
    # general para
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v2_evol/2.3_structured_classified_checklisted_ins.jsonl"
        ),
    )
    parser.add_argument("--input_file_extra", type=str, default="")
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v3_difficult/round_{n}/3.1_difficulty_taged_ins.jsonl"
        ),
    )
    parser.add_argument("--round_idx", type=int, default=0)
    parser.add_argument(
        "--output_easy_pool",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v3_difficult/round_{n}/easy_pool.jsonl"
        ),
    )
    parser.add_argument(
        "--output_hard_pool",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v3_difficult/round_{n}/hard_pool.jsonl"
        ),
    )
    parser.add_argument(
        "--output_toohard_pool",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v3_difficult/round_{n}/toohard_pool.jsonl"
        ),
    )
    parser.add_argument("--cache_dir", type=str, default="./")
    parser.add_argument("--roll_batch_size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold_easy", type=float, default=0.5)
    parser.add_argument("--threshold_hard", type=float, default=0.032)
    parser.add_argument("--max_out", type=int, default=4096)
    parser.add_argument("--int_flag_enable_senior", type=int, default=0)
    parser.add_argument("--url_level", type=int, default=1)
    parser.add_argument("--easy_level", type=int, default=1)
    parser.add_argument("--senior_level", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=64)
    # parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    if "round_" in args.input_file:
        args.input_file = args.input_file.format(n=args.round_idx)
    if "round_" in args.output_file:
        args.output_file = args.output_file.format(n=args.round_idx)
    if "round_" in args.output_easy_pool:
        args.output_easy_pool = args.output_easy_pool.format(n=args.round_idx)
    if "round_" in args.output_hard_pool:
        args.output_hard_pool = args.output_hard_pool.format(n=args.round_idx)
    if "round_" in args.output_toohard_pool:
        args.output_toohard_pool = args.output_toohard_pool.format(n=args.round_idx)
    if "round_" in args.input_file_extra:
        args.input_file_extra = args.input_file_extra.format(n=args.round_idx)

    print(args)

    input_list = readjsonl(args.input_file)
    if len(args.input_file_extra) > 0:
        input_list_extra = readjsonl(args.input_file_extra)
        input_list.extend(input_list_extra)
    # args.url_level = 1
    tagged_list = tag_difficulty(input_list, args)  # tag for models level 0

    if args.int_flag_enable_senior:
        args.url_level = 2
        tagged_list = tag_difficulty(tagged_list, args)

    easy_pool = []
    # if senior mode is on, delete questions which is too hard
    hard_pool = []  # for RL
    too_hard_pool = []
    for item in tagged_list:
        if (
            args.int_flag_enable_senior
            and item[f"pass_rate_{args.senior_level}"] < args.threshold_hard
        ):
            too_hard_pool.append(item)
            continue  # discard this item
        if item[f"pass_rate_{args.easy_level}"] > args.threshold_easy:
            easy_pool.append(item)
            continue
        hard_pool.append(item)

    logger.info(f"Easy pool contains {len(easy_pool)} items.")
    logger.info(f"Hard pool contains {len(hard_pool)} items.")
    logger.info(f"Too hard pool contains {len(too_hard_pool)} items.")

    writejsonl(tagged_list, args.output_file.format(n=args.round_idx))
    writejsonl(easy_pool, args.output_easy_pool.format(n=args.round_idx))
    writejsonl(hard_pool, args.output_hard_pool.format(n=args.round_idx))
    writejsonl(too_hard_pool, args.output_toohard_pool.format(n=args.round_idx))
