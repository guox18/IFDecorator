# prompt
filter_en_prompt = """You are a professional data labeling expert. Your job is to examine a given user instruction with corresponding response and classify it into one of the following categories, or determine that it does not belong to any of them.

### Categories:
1. **Math Problem** – The instruction is asking to solve a math problem, perform calculations, involves mathematical reasoning, or mathematical tasks with any possible background.
2. **Code Task** – The instruction is related to programming or code. This includes writing code, reviewing/explaining code, debugging, or other coding tasks.
3. **Reasoning Task** – The instruction involves logical reasoning or puzzle-solving. It might be a brain teaser, a logic puzzle, or any task requiring reasoning.

If the instruction clearly fits **one** of the above categories, return the **name** of that category (exactly "Math Problem", "Code Task", or "Reasoning Task"). If it does **not** fit any of these categories, return **"Other"**.

When deciding on the category, consider both the content of the instruction and what a likely answer would involve, to ensure you choose the correct category.

### Input:
---Instruction---:
<instruction>
{instruction}
</instruction>

---Response---:
<response>
{response}
</response>

### Output Format:
Provide your answer as a JSON object, for example:  
{
    "instruction": "What is 2+2?",
    "reason": "The instruction is asking to solve a math problem, perform calculations, or involves mathematical reasoning.",
    "label": "Math Problem"
}  
Make sure to output only the JSON object with the correct label and nothing else."""

### toolkit jsonprocess
import json
import os

### printcounter
import rich

from modules.utils import readjson, readjsonl, seed_everything, writejson, writejsonl


class PrintCounter:
    def __init__(self, max_prints, with_box=True):
        self.max_prints = max_prints
        self.print_count = 0
        self.with_box = with_box

    def print(self, *args, **kwargs):
        if self.print_count < self.max_prints:
            if self.with_box:
                print("\n")
                print("#" * 15)

            rich.print(*args, **kwargs)
            self.print_count += 1

            if self.with_box:
                print("#" * 15)
                print("\n")
        else:
            pass


def process_model_output(json_str):
    """
    解析模型返回的 JSON 数据，并根据 label 进行处理。
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: 无效的 JSON 格式。")
        return

    # 检查是否包含必要的键
    if "instruction" not in data or "label" not in data:
        print("Error: JSON 数据中缺少 'instruction' 或 'label' 字段。")
        return

    instruction = data["instruction"]
    label = data["label"]

    return label


from infer.apis import APIAgent
from infer.multiprocess_wrapper import MultiProcessHandler, preprocess_list

agent = APIAgent(
    [
        [
            "http://x.x.x.x:8000/v1",
        ]
    ],
    model_name="qwen2.5",
    max_tokens=2048,
    temperature=0.0,
    save_badcases=False,
)
mp_handler = MultiProcessHandler(
    agent.query, max_workers=256, threading=True, show_progress=True
)

# 示例调用
if __name__ == "__main__":
    model_response = """
    {
        "instruction": "What is 2+2?",
        "label": "Math Problem"
    }
    """
    process_model_output(model_response)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="./cif_set1_set2_hard1234.jsonl"
    )
    args = parser.parse_args()
    args.filtered_dir = os.path.join(os.path.dirname(args.input_file), "filtered")

    input_list = readjsonl(args.input_file)
    infer_conversations = [
        [
            {
                "role": "user",
                "content": item["reward_model"]["ground_truth"][
                    "complex_instruction_soft"
                ],
            }
        ]
        for item in input_list
    ]
    result_list = mp_handler.run(preprocess_list(infer_conversations))
    assert len(input_list) == len(result_list)
    input_list = [
        item for item, result in zip(input_list, result_list) if result is not None
    ]
    result_list = [result for result in result_list if result is not None]
    assert len(input_list) == len(result_list)

    contents = [result.content for result in result_list]
    assert len(input_list) == len(contents)
    judge_conversations = [
        [
            {
                "role": "user",
                "content": filter_en_prompt.replace(
                    "{instruction}",
                    item["reward_model"]["ground_truth"]["complex_instruction_soft"],
                ).replace("{response}", content),
            }
        ]
        for item, content in zip(input_list, contents)
    ]

    result_list = mp_handler.run(preprocess_list(judge_conversations))
    assert len(input_list) == len(result_list)
    input_list = [
        item for item, result in zip(input_list, result_list) if result is not None
    ]
    result_list = [result for result in result_list if result is not None]
    assert len(input_list) == len(result_list)
    labels = [result.content for result in result_list]

    assert len(input_list) == len(labels)

    print_counter = PrintCounter(max_prints=6)
    for conversation, label in zip(judge_conversations, labels):

        print_counter.print(conversation)
        print_counter.print(label)

    math_filtered_list = []
    code_filtered_list = []
    logic_filtered_list = []
    clean_list = []

    for item, label in zip(input_list, labels):
        parsed_label = process_model_output(label)
        if parsed_label is None:
            parsed_label = "Drop"

        if parsed_label == "Math Problem":
            math_filtered_list.append(item)
        elif parsed_label == "Code Task":
            code_filtered_list.append(item)
        elif parsed_label == "Reasoning Task":
            logic_filtered_list.append(item)
        else:
            clean_list.append(item)

    print(f"math_filtered_list: {len(math_filtered_list)}")
    print(f"code_filtered_list: {len(code_filtered_list)}")
    print(f"logic_filtered_list: {len(logic_filtered_list)}")
    print(f"clean_list: {len(clean_list)}")

    writejsonl(
        math_filtered_list,
        args.filtered_dir
        + "/"
        + os.path.basename(args.input_file).replace(".jsonl", "/math_filtered.jsonl"),
    )
    writejsonl(
        code_filtered_list,
        args.filtered_dir
        + "/"
        + os.path.basename(args.input_file).replace(".jsonl", "/code_filtered.jsonl"),
    )
    writejsonl(
        logic_filtered_list,
        args.filtered_dir
        + "/"
        + os.path.basename(args.input_file).replace(".jsonl", "/logic_filtered.jsonl"),
    )
    writejsonl(
        clean_list,
        args.filtered_dir
        + "/"
        + os.path.basename(args.input_file).replace(".jsonl", "/clean.jsonl"),
    )
