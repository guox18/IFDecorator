import argparse
import os

from rich import print

from global_config import PROJECT_ROOT
from infer.apis import get_res_cycle
from infer.multiprocess_wrapper import parallel_wrapper, preprocess_list

##
from logs.logger import logger
from modules.utils import readjson, readjsonl, seed_everything, writejson, writejsonl

debug_value = os.environ.get("DEBUG", "false").lower() == "true"

# prompt used to judge if the response satisfy **seed instruction**
add_checklist_en_prompt = """Design a checklist to evaluate whether the *target constraint* specified in the *instruction* is met. FOCUS SOLELY on verifying the *target constraint*, and ignore all other constraints or requirements outside the *target constraint*. The checklist should include a series of yes/no questions or conditions, ensuring that each item directly checks the satisfaction of the *target constraint* in the response.
**Checklist Format:**
- Each item should be written as a question or statement that verifies whether the *target constraint* is fulfilled.
- The checklist should be clear and concise, ideally in the form of yes/no questions or conditions that are easy to verify.
- The output should contain each checklist item as a separate bullet point.
**[Instruction]**
<Instruction>
{instruction}
</Instruction>

**[Target Constraint]**  
<TargetConstraint>
{target_constraint}
</TargetConstraint>

**[Checklist]** 
<Checklist>
"""


def process_single_item(item, args):
    item["instruction_id_list"] = []
    item["kwargs"] = []
    item["formatted_ins"]["hard_constraints_checklist"] = []
    # 设计检查项
    user_prompt = item["prompt"]
    # 遍历
    for c_idx, (constraint, constraint_type) in enumerate(
        zip(
            item["formatted_ins"]["constraints"],
            item["formatted_ins"]["constraints_type"],
        )
    ):
        if constraint_type == "hard":
            user_query = add_checklist_en_prompt.format(
                instruction=user_prompt, target_constraint=constraint
            )
            res = get_res_cycle(
                user_query,
                temperate=args.temperature,
                max_tokens=args.max_out,
                url_level=args.url_level,
            )
            if debug_value:
                print(f"- QUERY:\n{user_query}")
                print(f"- RESPONSE:\n{res}")
            item["formatted_ins"]["hard_constraints_checklist"].append(res)
    return item


def main():
    parser = argparse.ArgumentParser(description="add_checklist")
    # general para
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v2_evol/2.2_structured_classified_ins.jsonl"
        ),
    )
    parser.add_argument(
        "--seed_data_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v1_seed/seed_data_high_quality_smallset.jsonl"
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v2_evol/2.3_structured_classified_checklisted_ins.jsonl"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_out", type=int, default=2048)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--url_level", type=int, default=1)
    # parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    print(args)

    ### rollout
    input_list = readjsonl(args.input_file)
    seed_data_list = readjsonl(args.seed_data_file)
    assert len(input_list) % len(seed_data_list) == 0
    implicit_batch_size = len(input_list) // len(seed_data_list)
    logger.info(f"implicit_batch_size:{implicit_batch_size}")
    requests = []
    idx_list = []
    ## 初始化, 加上已有指令的checklist和空kwargs, instruction_id_list

    new_results = parallel_wrapper(
        preprocess_list(input_list, [args]),
        process_single_item,
        cache_dir="./",
        cache_file="cache_tmp",
        max_workers=args.max_workers,
        savebatch=10000,
    )
    writejsonl(new_results, args.output_file)


if __name__ == "__main__":
    seed_everything(42)
    ## main function
    main()
