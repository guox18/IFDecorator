import argparse
import copy
import json
import os

from rich import print

from global_config import PROJECT_ROOT
from infer.apis import get_res_cycle
from infer.multiprocess_wrapper import parallel_wrapper, preprocess_list
from logs.logger import logger
from modules.utils import (
    is_en_lang,
    readjson,
    readjsonl,
    seed_everything,
    writejson,
    writejsonl,
)

debug_value = os.environ.get("DEBUG", "false").lower() == "true"

decompose_en_prompt = """You are a prompt engineering specialist. Given a prompt, perform the following clearly defined tasks:

### Tasks:
1. **Extract Task Description**: Clearly state the primary objective of the prompt.
2. **List Constraints**: Identify and list explicit rules, formats, styles, conditions, or limitations specified in the prompt. If none exist, output `NULL`.
3. **Determine Input Requirements**: Identify any specific data or inputs explicitly required from the user. If none exist, output `NULL`.

### Processing Guidelines:
- Use `NULL` for Constraints and Input fields if the prompt does not explicitly mention them.
- Do not duplicate content between Task Description, Constraints, and Input fields.
- Ensure extracted information is semantically consistent with the original prompt.

### Output Format:
#Task Description: [Concise statement of the primary objective]
#Constraints: [List constraints clearly] or NULL
#Input: [Specific user-provided data required] or NULL

### Examples
<Beginning of Example One>
#### Example One:
---INPUT---
#Prompt: Your task is to provide exactly three bullet points highlighting different impacts of climate change. Each bullet must be 20 words or fewer, with at least one explicitly mentioning an economic impact. Ensure no repetition of information between bullet points. Conclude your response with: "Hope that can help you."
---OUTPUT---
#Task Description: Provide exactly three bullet points highlighting different impacts of climate change.  
#Constraints:  
- Exactly three bullet points  
- Each bullet point must be 20 words or fewer  
- Include at least one bullet explicitly mentioning an economic impact  
- Avoid repetition across bullet points  
- End with "Hope that can help you."  
#Input: NULL
<End of Example One>

#### Example Two:
<Beginning of Example Two>
---INPUT---
#Prompt: Your task is to briefly summarize the following paragraph in two sentences or fewer. Your summary should clearly convey the main idea. Do not use more than 30 words in total. Input: Artificial intelligence (AI) has rapidly advanced in recent years, significantly altering industries from healthcare to transportation. Despite its benefits, AI raises ethical questions about privacy, employment, and bias.
---OUTPUT---
#Task Description: Summarize the given paragraph clearly and briefly.
#Constraints:
- Maximum of two sentences
- Total summary length must not exceed 30 words
- Clearly convey the main idea
#Input:  Artificial intelligence (AI) has rapidly advanced in recent years, significantly altering industries from healthcare to transportation. Despite its benefits, AI raises ethical questions about privacy, employment, and bias.
<End of Example Two>

Now, analyze and respond to the provided prompt following the same method:

---INPUT---
#Prompt: {prompt}
---OUTPUT---
"""


def process_llm_response(content):
    """Helper function to process LLM response"""
    task_desc = None
    constraints = []
    input_text = None

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#Task Description:"):
            task_desc = line.replace("#Task Description:", "").strip()
        elif line.startswith("#Constraints:"):
            constraint_text = line.replace("#Constraints:", "").strip()
            if constraint_text.lower() == "null":
                constraints = []
                continue
        elif line.startswith("#Input:"):
            input_text = line.replace("#Input:", "").strip()
            if input_text.lower() == "null":
                input_text = None
        elif line.startswith("-") and task_desc is not None:
            # Add constraint if we've seen task description
            constraint = line.strip("- ").strip()
            if constraint:
                constraints.append(constraint)

    return {
        "task_description": task_desc,
        "constraints": constraints,  # could be []
        "input": input_text,
    }


def process_single_item(item, args):
    user_prompt = item["prompt"]
    query = decompose_en_prompt.format(prompt=user_prompt)

    res = get_res_cycle(
        query,
        temperate=args.temperature,
        max_tokens=args.max_out,
        url_level=args.url_level,
    )

    if debug_value:
        print(f"- QUERY: {query}")
        print(f"- RESPONSE: {res}")

    item["formatted_ins"] = process_llm_response(res)
    item["formatted_ins"]["original_prompt"] = item["prompt"]
    return item


def main():
    parser = argparse.ArgumentParser(description="llm judge")
    # general para
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v1_seed/seed_data_high_quality_smallset.jsonl"
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
        default=os.path.join(PROJECT_ROOT, "data/v2_evol/2.1_structured_ins.jsonl"),
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_out", type=int, default=8196)
    parser.add_argument("--batch_parallel", type=int, default=8)
    parser.add_argument("--start_device", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--url_level", type=int, default=2)
    # parser.add_argument("--debug", type=int, default=1)
    args = parser.parse_args()

    print(args)

    try:
        # Load input data
        input_list = readjsonl(args.input_file)
        seed_data_list = readjsonl(args.seed_data_file)

        if not input_list or not seed_data_list:
            raise ValueError("Empty input or seed data files")

        if len(input_list) % len(seed_data_list) != 0:
            raise ValueError(
                "Input list length must be divisible by seed data list length"
            )

        implicit_batch_size = len(input_list) // len(seed_data_list)
        logger.info(f"implicit_batch_size:{implicit_batch_size}")

        new_results = parallel_wrapper(
            preprocess_list(input_list, [args]),
            process_single_item,
            cache_dir="./",
            cache_file="cache_tmp",
            max_workers=args.max_workers,
            savebatch=10000,
        )
        writejsonl(new_results, args.output_file)
        logger.info(f"Successfully processed {len(new_results)} items")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    seed_everything(42)
    main()
