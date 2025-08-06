import os
import random
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from collections import namedtuple
from functools import partial

import requests
import rich
from apis import APIAgent
from multiprocess_wrapper import MultiProcessHandler
from openai import OpenAI
from rewards.urls_lib import URLS_7B, URLS_32B
from tqdm import tqdm

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards._vendor.ifeval import instructions_registry

## 72b judge
URLS = URLS_32B

agent = APIAgent(
    url_groups=URLS,  # Wrap in list since we're using a single group
    api_key="token-abc123",
    model_name="qwen",
    max_tokens=8192,
    temperature=0.0,
    save_badcases=True,
)

judge_overall_en_prompt = """### Structured Evaluation Protocol
Perform a rigorous analysis of the instruction-response pair through these sequential checks:
1. Instruction-Response Alignment
- Verify explicit understanding of core instruction objectives
- Check for missing required components from instruction
2. Logical Coherence Evaluation
- Trace logical flow from instruction premises to response conclusions
- Detect reasoning gaps or unwarranted assumptions
- Flag contradictions within the response
3. Context-aware Instruction Verification
- Analyze response against instruction type:
  ⤷ Query-type: Verify question resolution completeness
  ⤷ Task-type: Validate step-by-step executable logic
  ⤷ Creative-type: Assess objective-aligned originality
4. Safety & Compliance Check
- Identify potential risks or harmful implications
- Verify adherence to ethical guidelines

### Evaluation Parameters
- Strict true/false determination for each checkpoint
- Zero tolerance for partial fulfillment
- Mandatory failure for any single unmet criterion

### Evaluation Target
**Instruction:** 
<Instruction>
{instruction_wo_strict}
</Instruction>

### Response: 
<Response>
{response}
</Response>

### Output Format
First, present analysis in ordered checklist format. Then, conclude with final verdict using strict formatting:
**Final Verification:** <YES/NO>
"""

judge_checklist_en_prompt = """You are an impartial judge. Your task is to judge to evaluate whether the *target constraint* specified in the *instruction* is met in the *response* based on the  *checklist*. Focus solely on verifying the *target constraint*, and disregard any other constraints that may be present in the *instruction*.

### Instruction:
<Instruction>
{instruction_wo_strict}
</Instruction>

### Target Constraint:
<TargetConstraint>
{target_constraint}
</TargetConstraint>

### Response:
<Response>
{response}
</Response>

### Checklist:
<Checklist>
{checklist}
</Checklist>

### Output Format:
First, present analysis in ordered checklist format. Then, conclude with final verdict using strict formatting in English:
**Final Verification:** <YES/NO>
"""


### both True or False are OK
def unified_judge_parse(response: str, flag_strict=False):
    """
    Parse and judge the verification response from the model.

    Args:
        response (str): The model's response text
        flag_strict (bool): Whether to use strict matching rules

    Returns:
        bool: True if verification passes, False otherwise
    """
    if response is None:
        return False
    if flag_strict:
        if (
            ("**Final Verification:** YES" in response)
            or ("Final Verification: YES" in response)
            or ("**Final Verification:** 'YES'" in response)
            or ("Final Verification: 'YES'" in response)
            or ("**Final Verification:** <<YES>>" in response)
            or ("Final Verification: <<YES>>" in response)
            or ("**Final Verification:** **YES**" in response)
            or ("Final Verification: **YES**" in response)
        ):
            return True
        else:
            return False
    else:
        ## loose judge for no start and end
        response = response.lower()

        if "final_ver" in response:
            split_response = response.split("final_ver")
        elif "final verdict" in response:
            split_response = response.split("final verdict")
        else:
            split_response = response.split("final verification")
        if len(split_response) < 2:
            print(f"final verification not found in judge response: {response}")
            return False

        response = split_response[-1]

        if "yes" in response:
            return True
        else:
            return False


def get_res(user_prompt: str):
    response = agent.query(
        [{"role": "user", "content": user_prompt}], url_level=0, receive_length=False
    )
    if response is None:
        return None
    return response.content


# 同时检测think格式
def test_instruction_following_strict(response, instruction_list, instruction_kwargs):
    """Tests response to see if instrutions are followed."""
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        # if instruction_id not in instructions_registry.INSTRUCTION_DICT.keys():
        #     continue
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        ## 由于parquet的缘故 需要把key为None的洗掉
        new_instruction_kwargs = []
        for ins_args in instruction_kwargs:
            # 使用字典推导式过滤掉值为 None 的项
            current_item = {}
            for key, value in ins_args.items():
                if value is not None:
                    # float 转 int, 这个问题源自我从jsonl保存到parquet的时候, int 不小心成float了
                    if isinstance(value, float):
                        value = int(value)
                    current_item[key] = value
            new_instruction_kwargs.append(current_item)
            # new_instruction_kwargs.append({key: value for key, value in ins_args.items() if value is not None})

        instruction.build_description(**new_instruction_kwargs[index])

        # if '</THINK>' in response:
        #     return False

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    print(
        f"\n----------------\nis_following_list:\n{is_following_list}\n--------------\n"
    )

    return all(is_following_list)


def test_instruction_following_llm(
    prompt: str,
    response: str,
    list_constraint: list,
    list_of_checklist: list,
):
    """Tests response to see if soft constraints are followed."""
    ## level one judge
    judge_query_overall = judge_overall_en_prompt.format(
        instruction_wo_strict=prompt, response=response
    )

    print(
        f"\n----------------\njudge_query_overall:\n{judge_query_overall}\n--------------\n"
    )

    judge_res = get_res(judge_query_overall)

    print(f"\n----------------\njudge_res:\n{judge_res}\n--------------\n")

    if unified_judge_parse(judge_res) == False:
        return False

    ## level two judge
    for constraint, checklist in zip(list_constraint, list_of_checklist):
        judge_query_checklist = judge_checklist_en_prompt.format(
            instruction_wo_strict=prompt,
            target_constraint=constraint,
            checklist=checklist,
            response=response,
        )
        print(
            f"\n----------------\njudge_query_checklist:\n{judge_query_checklist}\n--------------\n"
        )
        judge_res = get_res(judge_query_checklist)
        print(f"\n----------------\njudge_res:\n{judge_res}\n--------------\n")
        if unified_judge_parse(judge_res) == False:
            return False

    ## level three judge
    ## todo

    return True


def extract_solution(solution_str):
    # print(f'\n----- Solution before extract_solution -----\n{solution_str}\n------------------\n')

    ## 由于版本变更, 此处无效了
    ## 掐头
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant")[-1].strip()
    if "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )[-1].strip()

    ## 判断 response 是否正常结束
    if "<|eot_id|>" not in solution_str and "<|im_end|>" not in solution_str:
        return None

    ## 掐尾
    solution_str = solution_str.split(
        "<|eot_id|>" if "<|eot_id|>" in solution_str else "<|im_end|>"
    )[0].strip()

    # print(f'\n----- Solution after extract_solution -----\n{solution_str}\n------------------\n')
    # breakpoint()
    ### 这里决定是否强制要</think>
    # if '</think>' not in solution_str:
    #     return None
    # else:
    #     solution_str = solution_str.split('</think>')[-1].strip()
    return solution_str


# def postprocess_solution(solution_str):
#     if "<|im_end|>" in solution_str:
#         return solution_str[:solution_str.index("<|im_end|>")]
#     return solution_str


def compute_single_score(solution_str, ground_truth):
    ## 还带有尾巴
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        print(
            f"\n----- Result -----\nSolution without tail or extract_solution failed. Reward: -1\n------------------\n"
        )
        print(
            f"\n----- Solution before extract_solution -----\n{solution_str}\n------------------\n"
        )
        return -1

    print(f"\n----- Answer-----\n{answer}\n------------------\n")
    if not test_instruction_following_strict(
        answer, ground_truth["instruction_id_list"], ground_truth["kwargs"]
    ):
        print(
            f"----- Result -----\nHard constraints not satisfied. Reward: 0\n------------------\n"
        )
        return 0
    # {true true false false}
    judge_res = test_instruction_following_llm(
        prompt=ground_truth["complex_instruction_soft"],
        response=answer,
        list_constraint=ground_truth["constraints"],
        list_of_checklist=ground_truth["hard_constraints_checklist"],
    )
    if not judge_res:
        print(
            f"----- Result -----\nPrompt without hard constraints not satisfied. Reward: 0\n------------------\n"
        )
        return 0
    print(f"----- Result -----\nOne generation passed! Reward: 1\n------------------\n")
    return 1


def cif_compute_score(batch_data_sources, batch_solution_str, batch_ground_truth):
    if "cif" not in batch_data_sources[0]:
        raise ValueError("batch_data_sources must contain 'cif'")

    rewards = []

    # ##### 测试通过, 准备实现并行加速
    # for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
    #     raw_solution_str = solution_str
    #     try:
    #         solution_str = extract_solution(solution_str)
    #         rewards.append(
    #             compute_single_score(solution_str, ground_truth)
    #         )
    #     except Exception as err:
    #         print(f"An error occurred while processing the solution: {repr(raw_solution_str)}")
    #         print(f"Ground Truth: {repr(ground_truth)}")
    #         print(f"Error: {err}")
    #         rewards.append(0.)
    # return rewards
    # #############################

    # 默认取 level 0 的 url 数量来决定并发的多线程数量
    num_workers = len(URLS[0]) * 256
    print(f"max_workers: {num_workers}")
    multi_process_handler = MultiProcessHandler(
        process_item_func=compute_single_score,
        max_workers=num_workers,
        show_progress=True,
    )

    # try:
    #     rewards = multi_process_handler.run(list(zip(batch_solution_str, batch_ground_truth)))
    # except Exception as err:
    #     rich.print(f"[bold red]An error occurred while processing the solutions: {err}[/bold red]")
    #     return [0.] * len(batch_solution_str)
    rewards = multi_process_handler.run(
        list(zip(batch_solution_str, batch_ground_truth))
    )

    return rewards


if __name__ == "__main__":
    # prompt
    complex_instruction = 'In this task, you will be given a sentence. You need to reconize the name of the disorder or disease. Disease is a disorder of structure or function in a human, animal, or plant, especially one that produces specific symptoms or that affects a specific location and is not simply a direct result of physical injury. Although there might be several correct answers, you need to write one of them. \n\ninput :  In individuals with mutations in either region 2 or region 3 , the average number of adenomas tended to be lower than those in individuals with mutations in region 1 , although age at diagnosis was similar . Include a title wrapped in double angular brackets, i.e. <<title>>. Answer in lowercase letters only, throughout your entire answer. Respond with at least 3 sentences. The word "adenomas" should not appear in your response. the total number of words in your response should be at least 22. '
    # wo hard
    complex_instruction_soft = "In this task, you will be given a sentence. You need to reconize the name of the disorder or disease. Disease is a disorder of structure or function in a human, animal, or plant, especially one that produces specific symptoms or that affects a specific location and is not simply a direct result of physical injury. Although there might be several correct answers, you need to write one of them. \n\ninput :  In individuals with mutations in either region 2 or region 3 , the average number of adenomas tended to be lower than those in individuals with mutations in region 1 , although age at diagnosis was similar ."

    id_list = [
        "detectable_format:title",
        "change_case:english_lowercase",
        "length_constraints:number_sentences",
        "keywords:forbidden_words",
        "length_constraints:number_words",
    ]

    kwargs = [
        {},
        {},
        {"relation": "at least", "num_sentences": 3},
        {"forbidden_words": ["adenomas"]},
        {"relation": "at least", "num_words": 22},
    ]

    data = {
        "data_source": "cif",
        "prompt": [
            {
                "content": "[System]\nYou are an AI Assistant designed to handle complex user instructions. You first thinks through the instruction, then generates a response. The thinking process is enclosed within `<think>` and `</think>` tags, followed by the response.\n\nExample format:\n<think> Thinking process here </think>\nResponse here\n\n[User]\nState the name of an identifier for a variable that is used to store a Boolean value.\nmake sure that words with all capital letters appear less than 3 times. There should be no commas in your reply. Please include at least 2 placeholders represented by square brackets, such as [address]. ",
                "role": "user",
            }
        ],
        "ability": "if",
        "reward_model": {
            "ground_truth": {
                "instruction_id_list": [
                    "change_case:capital_word_frequency",
                    "punctuation:no_comma",
                    "detectable_content:number_placeholders",
                ],
                "kwargs": [
                    {"capital_relation": "less than", "capital_frequency": 3},
                    {},
                    {"num_placeholders": 2},
                ],
                "complex_instruction_soft": "State the name of an identifier for a variable that is used to store a Boolean value.\n",
            },
            "style": "rule",
        },
        "extra_info": {
            "index": 0,
            "question": "State the name of an identifier for a variable that is used to store a Boolean value.\nmake sure that words with all capital letters appear less than 3 times. There should be no commas in your reply. Please include at least 2 placeholders represented by square brackets, such as [address]. ",
            "split": "train",
        },
    }

    print(data)

    response = """<think> the sentence discusses individuals with mutations in different regions, leading to variations in the number of adenomas. however, the term \"adenomas\" should not appear in the response. since adenomas are typically associated with polyposis syndromes, like familial adenomatous polyposis (fap), the disease name could be inferred from that context. fap is a disorder where numerous polyps develop in the colon and rectum, aligning with the theme of mutations affecting growth in specific regions.</think>\n<<familial adenomatous polyposis>> familial adenomatous polyposis is a disorder characterized by mutations leading to the development of numerous polyps, often in the colon and rectum. the presence of such mutations in different regions can significantly influence the number and age of onset of these growths. this condition exemplifies how specific genetic changes can affect human health."""

    print(
        test_instruction_following_strict(
            response,
            data["reward_model"]["ground_truth"]["instruction_id_list"],
            data["reward_model"]["ground_truth"]["kwargs"],
        )
    )
