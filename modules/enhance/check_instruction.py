from openai import OpenAI

from infer.apis import get_res, get_res_cycle

from infer.multiprocess_wrapper import MultiProcessHandler

##
from logs.logger import logger
from modules._vendor.ifeval.instructions import *
from modules._vendor.ifeval.instructions_registry import INSTRUCTION_DICT
from modules.utils import unified_judge_parse

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


def test_instruction_following_strict(response, instruction_list, instruction_kwargs):
    """Tests response to see if instrutions are followed."""
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
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

        instruction.build_description(**new_instruction_kwargs[index])

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)
    # print(f'\n----------------\nis_following_list:\n{is_following_list}\n--------------\n')
    return all(is_following_list)


def test_instruction_following_llm(
    prompt: str,
    response: str,
    list_constraint: list,
    list_of_checklist: list,
    # api_endpoint : str = 'http://127.0.0.1:8007/v1',
    model_name: str = "qwen",
    max_tokens: int = 1024,
    url_level: int = 1,
):
    """Tests response to see if soft constraints are followed."""
    ## level one judge
    judge_query_overall = judge_overall_en_prompt.format(
        instruction_wo_strict=prompt, response=response
    )
    judge_res = get_res_cycle(
        judge_query_overall,
        url_level=url_level,
        model_name=model_name,
        max_tokens=max_tokens,
    )

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
        judge_res = get_res_cycle(
            judge_query_checklist,
            url_level=url_level,
            model_name=model_name,
            max_tokens=max_tokens,
        )
        if unified_judge_parse(judge_res) == False:
            return False

    ## level three judge
    ## todo

    return True


# if __name__ == '__main__':
#     print(test_instruction_following_strict('a',[],[]))
