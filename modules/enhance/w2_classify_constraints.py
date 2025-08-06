import argparse
import os

from rich import print

from global_config import PROJECT_ROOT
from infer.apis import get_res_cycle
from infer.multiprocess_wrapper import parallel_wrapper, preprocess_list
from logs.logger import logger
from modules.utils import readjson, readjsonl, seed_everything, writejson, writejsonl

debug_value = os.environ.get("DEBUG", "false").lower() == "true"

classification_en_prompt_old = """You are a prompt engineering specialist. Your task is to analyze whether a given constraint in a prompt belongs to **hard constraints** or **soft constraints** based on the definitions below.

### Definitions:
1. **Hard Constraints**: 
   - Explicit verifiable requirements with clear yes/no validation
   - Can be checked programmatically (e.g., word count, specific format)
   - Examples: JSON format requirement, 3 bullet points, exactly 100 words

2. **Soft Constraints**:
   - Open-ended requirements with subjective interpretation
   - Requires human judgment to evaluate compliance
   - Examples: Specify emotional tone, encourage ambiguity, raise standards

### Analysis Steps:
1. Determine verification feasibility:
   - If measurable through scripts/pattern matching → Hard
   - If requires subjective interpretation → Soft

2. Consider constraint specificity:
   - Numeric/structural requirements → Hard
   - Qualitative/stylistic requirements → Soft

### Format
<Beginning of Format Specifications>
---Input---
#Prompt: [user's original prompt]
#Constraint: [specific constraint to classify]

---Output---
#reasoning : [concise explanation]
#verification_method: [describe how this could/should be verified]
#constraint_type: [hard/soft]
<End of Format Specifications>

---Input---
#Prompt: {prompt}
#Constraint: {constraint}
---Output---
"""

classification_en_prompt = """You are a prompt engineering specialist. Your task is to analyze whether a given constraint in a prompt belongs to **hard constraints** or **soft constraints** based on the definitions below.

### Definitions:
1. **Hard Constraints**: 
   - Explicit verifiable requirements with clear yes/no validation
   - Examples: JSON format requirement, 3 bullet points, exactly 100 words

2. **Soft Constraints**:
   - Open-ended requirements with subjective interpretation
   - Examples: Specify emotional tone, encourage ambiguity, raise standards

### Analysis Steps:
1. Determine verification feasibility:
   - If measurable through scripts/pattern matching → Hard
   - If requires subjective interpretation → Soft

2. Consider constraint specificity:
   - Numeric/structural requirements → Hard
   - Qualitative/stylistic requirements → Soft

### Format
<Beginning of Format Specifications>
---Input---
#Prompt: [user's original prompt]
#Constraint: [specific constraint to classify]

---Output---
#reasoning : [concise explanation]
#verification_method: [describe how this could/should be verified]
#constraint_type: [hard/soft]
<End of Format Specifications>

---Input---
#Prompt: {prompt}
#Constraint: {constraint}
---Output---
"""


def process_constraint_response(content):
    """Helper function to process constraint response"""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#constraint_type:"):
            constraint_type = line.replace("#constraint_type:", "").strip().lower()
            return "hard" if constraint_type == "hard" else "soft"
    return "soft"  # Default to soft if no type found


def process_single_item(item, args):
    user_prompt = item["prompt"]
    constraints_list = item["formatted_ins"].get("constraints", [])

    constraints_type = []
    for _, constraint in enumerate(constraints_list):
        query = classification_en_prompt.format(
            prompt=user_prompt, constraint=constraint
        )
        res = get_res_cycle(
            query,
            temperate=args.temperature,
            max_tokens=args.max_out,
            url_level=args.url_level,
        )
        if debug_value:
            print(f"- QUERY: {query}")
            print(f"- RESPONSE: {res}")
        constraint_type = process_constraint_response(res)
        constraints_type.append(constraint_type)

    item["formatted_ins"]["constraints_type"] = constraints_type
    return item


def main():
    parser = argparse.ArgumentParser(description="llm judge")
    # general para
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data/v2_evol/2.1_structured_ins.jsonl"),
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
            PROJECT_ROOT, "data/v2_evol/2.2_structured_classified_ins.jsonl"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_out", type=int, default=2048)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--url_level", type=int, default=1)
    # parser.add_argument("--debug", type=int, default=0)

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
