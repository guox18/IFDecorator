import argparse
import copy
import json
import os
import random
import re

from rich import print
from tqdm import tqdm

from global_config import PROJECT_DATA, PROJECT_ROOT
from infer.apis import get_res_cycle
from infer.multiprocess_wrapper import (
    CheckpointManager,
    MultiProcessHandler,
    parallel_wrapper,
    preprocess_list,
)
from logs.logger import logger
from modules._vendor.ifeval import instructions_registry, instructions_util, taxonomy
from modules.utils import (
    chance,
    readjson,
    readjsonl,
    seed_everything,
    unified_judge_parse,
    writejson,
    writejsonl,
)

debug_value = os.environ.get("DEBUG", "false").lower() == "true"


quality_en_prompt = """Assess whether the instruction is sufficiently clear and actionable. Respond YES if it can be reasonably understood and executed without major issues. Respond NO only if it contains critical flaws such as:

- Complete lack of clarity in purpose
- Contradictory requirements
- Unintelligible language

### Instruction:
<Instruction>
{prompt}
</Instruction>

###[Evaluation Requirements]
1. Detailed analysis
2. Conclude with final verdict using strict formatting:
**Final Verification:** <YES/NO>
"""


def check_quality(item):
    """通过OpenAI API检查指令的语法、拼写、清晰度和相关性"""
    prompt = item["prompt"]
    try:
        query = quality_en_prompt.format(prompt=prompt)
        res = get_res_cycle(query, max_tokens=1024, url_level=args.url_level)
        if debug_value:
            print(f"- QUERY:\n{query}", flush=True)
            print(f"- RESPONSE:\n{res}", flush=True)
        return unified_judge_parse(res)
    except Exception as e:
        logger.error(f"Error in quality check: {e}")
        return False


evol_get_keywords_en_prompt = """You are provided with an instruction. Your task is to come up with some keywords that may be used to answer the instruction. They are usually related to the task described in the instruction. You should output your thinking process and the keywords you come up with.
### Output Instructions:
1. Think about the instruction and identify key concepts or terms that are relevant to it.
2. List these key concepts or terms as keywords.
### Output Format
Do not provide any additional explanations—only the thinking process and the keywords are needed. Output your final keywords using strict formatting:
**Thinking Process:** <thinking process>
**Keywords:** <keywords>
---INPUT---
**Instruction:**
<Instruction>
Explain Generative Adversarial Networks (GANs) to me using bullet points. Do not contain any commas in your response.
</Instruction>
---OUTPUT---
**Thinking Process:**
The instruction is to explain GANs, hence, 'architecture', 'training' and 'generator' may be appropriate keywords to use in the answer.
**Keywords:**
['architecture', 'training', 'generator']
---INPUT---
**Instruction:**
<Instruction>
{}
</Instruction>
---OUTPUT---
"""

example_list = [
    "Situation, Scenario-specific",
    "Situation, Story-driven",
    "Situation, Role-based",
    "Style, Tonal",
    "Style, Structural",
    "Style, Creative",
    "Content, Structural",
    "Content, Language",
    "Content, Open-scope",
]


def parse_prompt_template(
    template_dict, distribution_dict=None, shuffle_constraints=True
):
    """Parse the prompt template dictionary into a string format.

    Args:
        template_dict: Dictionary containing the prompt template sections
        distribution_dict: Dictionary containing the frequency distribution of constraint types
                         Format: {'constraint_type': frequency_count}
        shuffle_constraints: Whether to randomly shuffle constraints in each section

    Returns:
        String format of the prompt template
    """
    sections = []

    # Add main instruction
    sections.append(template_dict["instruction"])

    # Add guidelines with weighted sampling if distribution provided
    guidelines = template_dict["guidelines"]
    sections.append("\n### Guidelines:")
    for g in guidelines:
        sections.append(f"- {g}")

    # Add enhancement framework with weighted sampling
    sections.append("\n### Enhancement Framework:")
    framework = template_dict["enhancement_framework"]
    for category, details in random.sample(list(framework.items()), len(framework)):
        sections.append(f"\n{category.replace('_', ' ').title()}:")
        types = details["types"]
        examples = details["examples"]

        # Keep types and examples in sync when shuffling
        if distribution_dict and category in distribution_dict:
            # Calculate weights for types
            weights = [1.0 / (distribution_dict[category].get(t, 1) + 1) for t in types]
            total = sum(weights)
            weights = [w / total for w in weights]
            # Create indices for synchronized sampling
            indices = list(range(len(types)))
            sampled_indices = random.choices(indices, weights=weights, k=len(types))
            types = [types[i] for i in sampled_indices]
            examples = [examples[i] for i in sampled_indices]
        elif shuffle_constraints:
            # Create indices for synchronized shuffling
            indices = list(range(len(types)))
            random.shuffle(indices)
            types = [types[i] for i in indices]
            examples = [examples[i] for i in indices]

        sections.append(f"Types: {', '.join(types)}")
        sections.append("Examples:")
        for e in examples:
            sections.append(f"- {e}")

    # Add special rules with weighted sampling
    rules = template_dict["special_rules"]
    if distribution_dict and "special_rules" in distribution_dict:
        # Calculate weights for rules
        weights = [
            1.0 / (distribution_dict["special_rules"].get(r, 1) + 1) for r in rules
        ]
        total = sum(weights)
        weights = [w / total for w in weights]
        # Sample rules based on weights
        rules = random.choices(rules, weights=weights, k=len(rules))
    elif shuffle_constraints:
        random.shuffle(rules)
    sections.append("\n### Special Rules:")
    for r in rules:
        sections.append(f"- {r}")

    # Add output requirements
    sections.append("\n### Output Requirements:")
    output = template_dict["output_requirements"]
    for field, desc in output["fields"].items():
        if field == "constraint_type":
            constraint_type = random.choice(example_list)
            sections.append(
                f"#{field}: Format as 'Category, Type' (e.g. '{constraint_type}')"
            )
        else:
            sections.append(f"#{field}: {desc}")

    sections.append(
        """---INPUT---\n#**Original Instruction**:\n<Instruction>\n{instruction}\n</Instruction>\n\n---OUTPUT---\n"""
    )

    return "\n".join(sections)


def evol_dynamic_instruction_enchancement_en_prompt(instruction: str):
    # Test template
    template = {
        "instruction": "You are an Instruction Enhancement Expert. Analyze the **Original Instruction** and select the most appropriate enhancement category from [Content, Situation, Style]. Apply ONE relevant constraint to refine the instruction while following these guidelines:",
        "guidelines": [
            "Preserve all non-text elements (tables, code, etc.) from original",
            "Maintain logical coherence and human readability",
            "Add only 10-20 meaningful words for constraint integration",
            "Select constraints based on instruction type and enhancement potential",
        ],
        "enhancement_framework": {
            "content_constraints": {
                "types": ["Open-scope", "Language", "Structural"],
                "examples": [
                    "Add related subtask/question",
                    "Specify language complexity level",
                    "Require specific format/structure",
                ],
            },
            "situation_constraints": {
                "types": ["Role-based", "Scenario-specific", "Story-driven"],
                "examples": [
                    "Define role/persona requirements",
                    "Set environmental/contextual parameters",
                    "Add plot/character development elements",
                ],
            },
            "style_constraints": {
                "types": ["Tonal", "Structural", "Creative"],
                "examples": [
                    "Specify emotional tone",
                    "Request specific narrative style",
                    "Add ambiguity/humor elements",
                ],
            },
        },
        "output_requirements": {
            "fields": {
                "rationale": "Brief explanation of constraint selection (20 words)",
                "constraint_type": "Selected constraint category and specific type",
                "constraint": "The constraint to be added to the instruction",
                "enhanced_instruction": "Modified instruction with natural constraint integration",
            }
        },
        "special_rules": [
            "Prioritize constraint additions that create measurable boundaries",
            "Maintain original instruction intent while adding specificity",
            "Avoid overlapping/conflicting constraints in single enhancement",
        ],
    }

    # Test with default settings
    return parse_prompt_template(template).replace("{instruction}", instruction)


EVOL_DICT = {
    "content": {"language": 0, "open-scope": 0, "structural": 0},
    "situation": {"story-driven": 0, "role-based": 0, "scenario-specific": 0},
    "style": {"structural": 0, "creative": 0, "tonal": 0},
}


def parse_evol_response(response: str):
    """Parse the response from the evolution prompt into structured format.

    Args:
        response (str): Raw response text from the LLM

    Returns:
        dict: Parsed response with keys:
            - rationale (str): Explanation for constraint selection
            - constraint_type (tuple): Category and specific type
            - constraint (str): The constraint to be added to the instruction
            - enhanced_instruction (str): Modified instruction

    Raises:
        ValueError: If response cannot be parsed properly
    """
    try:
        # Split response into sections
        sections = response.strip().split("#")
        parsed = {}

        for section in sections:
            if not section.strip():
                continue

            # Split each section into key and content
            parts = section.strip().split(":", 1)
            if len(parts) != 2:
                continue

            key, content = parts[0].strip(), parts[1].strip()

            if "constraint_type" in key.lower():
                # Parse constraint type into tuple
                types = [t.strip() for t in content.split(",")]
                if len(types) != 2:
                    raise ValueError(f"Invalid constraint type format: {content}")
                parsed[key] = tuple(types)
            else:
                parsed[key] = content
        if debug_value:
            constraint_type = parsed["constraint_type"]
            logger.info(f"constraint_type: {constraint_type}")
            constraint = parsed["constraint"]
            logger.info(f"constraint: {constraint}")

        return parsed

    except Exception as e:
        logger.warning(f"Evolved instruction is not formatted. error info: {e}")
        return None


judge_include_en_prompt = """### Task: Text Matching Verification
You are given two pieces of text: **Text 1** and **Text 2**. Your task is to determine whether **Text 2** appears within **Text 1** as a substring.
### Output Instructions:
1. If Text 2 is largely present within Text 1, allowing for some minor differences, output YES.
2. Otherwise, output NO.
### Output Format
Do not provide any additional explanations—only the final judgment is needed. Output your final verdict using strict formatting:
**Final Verification:** <YES/NO>
---INPUT---
**Text 1:**
<text1>
{text1}
</text1>

**Text 2:**
<text2>
{text2}
</text2>

---OUTPUT---
"""


def judge_include(text1: str, text2: str, max_tokens: int = 100):
    if debug_value:
        logger.info(f"judge_include: {text1}, {text2}")
    res = unified_judge_parse(
        get_res_cycle(
            judge_include_en_prompt.format(text1=text1, text2=text2),
            max_tokens=max_tokens,
            url_level=args.url_level,
        )
    )
    if debug_value:
        logger.info(f"judge_include: {res}")
    return res


def get_keyword(seed_inst):
    prompt = evol_get_keywords_en_prompt.format(seed_inst)
    keyword_pattern = r"keywords:\n?(.*)\n?"
    response = get_res_cycle(
        user_prompt=prompt, temperate=0.7, max_tokens=1024, url_level=args.url_level
    )
    try:
        keywords = re.findall(keyword_pattern, response)[0]
        keywords = eval(keywords)
    except:
        keywords = random.choices(instructions_util.WORD_LIST, k=3)
    if debug_value:
        logger.info(f"keywords: {keywords}")
    return keywords


def merge_evol(seed_data, old_instruction, new_instruction, constraint):
    # original_inst = seed_data['formatted_ins']['original_prompt']
    original_input = seed_data["formatted_ins"]["input"]

    flag_include_input = False
    if original_input is None:
        original_input = ""
    if constraint is None:
        logger.warning("during merge_evol, a constraint is None, return None")
        constraint = ""
        return None

    if (
        original_input is None
        or original_input == ""
        or original_input in new_instruction
        or judge_include(text1=new_instruction, text2=original_input)
    ):
        flag_include_input = True

    flag_include_instruction = False
    if judge_include(text1=new_instruction, text2=old_instruction):
        flag_include_instruction = True

    if flag_include_input and flag_include_instruction:
        # 理想情况, 直接替换
        new_inst = new_instruction
    elif flag_include_instruction:
        # 模型忽略了input部分, 但是没有忽略任何instruction
        # if old_instruction is None:
        #     logger.error(f'A new instruction is None. constraint:{constraint}, original input:{original_input}')
        #     return None
        new_inst = new_instruction + "\n" + original_input
    else:
        # 直接添加, 可能语义不太好
        logger.warning(f"Concating a constraint with instruction and input.")
        # if old_instruction is None:
        #     logger.error(f'An old instruction is None. constraint:{constraint}, original input:{original_input}')
        #     return None
        new_inst = old_instruction + " " + constraint + "\n" + original_input

    # logger.info(f'\nevoling...\nconstraint:\n{constraint}\nold inst:\n{old_instruction}\nafter adding soft constraint:\n{seed_inst}\n')
    if debug_value:
        logger.info(f"- evolving")
        logger.info(f"-- constraint:\n{constraint}")
        logger.info(f"-- old inst:\n{old_instruction}")
        logger.info(f"-- after adding soft constraint:\n{new_inst}")
    return new_inst


def evol_single_seed_data(seed_data, args):
    # Select random number between 1-2 for evol attempts
    num_evol_attempts = random.randint(
        1 + args.num_evol_attempts // 2, args.num_evol_attempts
    )
    logger.info(f"Number of evol attempts: {num_evol_attempts}")
    # Track successful evolutions
    successful_evols = 0
    # Try to evolve instruction num_evol_attempts times
    current_instruction = seed_data["formatted_ins"]["original_prompt"]

    seed_data["soft_constraint_type"] = []
    seed_data["soft_constraint"] = []
    seed_data["num_successful_evols"] = 0
    for i in range(num_evol_attempts):
        try:
            # Get old and new instructions and constraint
            # old_instruction = seed_data['prompt']

            query_prompt = evol_dynamic_instruction_enchancement_en_prompt(
                current_instruction
            )
            if debug_value:
                logger.info(f"query_prompt: {query_prompt}")

            max_retries = args.max_evol_retries
            retry_count = 0
            parsed_dict = None

            while retry_count < max_retries and parsed_dict is None:
                parsed_dict = parse_evol_response(
                    get_res_cycle(
                        user_prompt=query_prompt,
                        temperate=args.evol_temperature,
                        max_tokens=args.evol_max_tokens,
                        url_level=args.url_level,
                    )
                )
                if parsed_dict is None:
                    retry_count += 1
                    logger.info(
                        f"Retry {retry_count}/{max_retries} for parse_evol_response"
                    )
            new_instruction = parsed_dict["enhanced_instruction"]
            constraint = parsed_dict["constraint"]
            if debug_value:
                logger.info(f"new_instruction: {new_instruction}")

            # Attempt evolution
            evolved_inst = merge_evol(
                seed_data=seed_data,
                old_instruction=current_instruction,
                new_instruction=new_instruction,
                constraint=constraint,
            )

            if evolved_inst:
                successful_evols += 1
                seed_data["num_successful_evols"] += 1

                current_instruction = evolved_inst
                seed_data["prompt"] = evolved_inst

                seed_data["soft_constraint_type"].append(parsed_dict["constraint_type"])
                seed_data["soft_constraint"].append(parsed_dict["constraint"])
                logger.info(f"Evolution {i+1} successful")
            else:
                logger.warning(f"Evolution {i+1}, return original instruction")
        except Exception as e:
            logger.warning(f"Failed evolution attempt {i+1}: {str(e)}")
            continue

    return seed_data


def evol_diverse(args):
    seed_datas = readjsonl(args.easy_pool_path)

    # pools to use
    evol_pool = []
    evol_exception_pool = []
    evols_data_list = parallel_wrapper(
        preprocess_list(seed_datas, [args]),
        evol_single_seed_data,
        cache_dir="./",
        cache_file="cache_tmp",
        max_workers=args.max_workers,
        savebatch=10000,
    )

    for seed_data, evol_data in zip(seed_datas, evols_data_list):
        if evol_data is None:
            evol_exception_pool.append(seed_data)
        else:
            evol_pool.append(evol_data)

    writejsonl(evol_pool, args.save_data_path)
    writejsonl(evol_exception_pool, args.evol_exception_path)


def process_single_seed_data(seed_data, args, hard_keys, cons_dict, conflict_dict):
    """Process a single seed data item with difficulty evolution"""
    lower_bound = args.num_hard_constraints_lower_bound
    upper_bound = args.num_hard_constraints_upper_bound

    select_num = random.randint(lower_bound, upper_bound)
    instruction_id_list = []
    index = 0
    recur_count = 0
    while index < select_num:
        random_key = random.choice(hard_keys)
        if index == 0 and "constrained_response" in random_key:
            instruction_id_list.append(random_key)
            break
        # json
        if index == 0 and ("json_format" in random_key or "xml_format" in random_key):
            instruction_id_list.append(random_key)
            instruction_id_list.append(
                random.choice(["keywords:forbidden_words", "keywords:existence"])
            )
            break
        if list(set(conflict_dict[random_key]) & set(instruction_id_list)):
            recur_count += 1
            if recur_count > 1000:
                logger.info(instruction_id_list)
                logger.info(random_key)
                break
            continue
        instruction_id_list.append(random_key)
        index += 1
    if debug_value:
        logger.info(f"instruction_id_list is {instruction_id_list}")

    seed_inst = seed_data["prompt"]
    # directly incorporated
    woGPT_list = [
        "detectable_format:multiple_sections",
        "change_case:capital_word_frequency",
        "detectable_format:number_highlighted_sections",
        "detectable_format:number_bullet_lists",
        "detectable_content:postscript",
        "detectable_content:number_placeholders",
        "length_constraints:number_words",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
        "language:response_language",
        "keywords:letter_frequency",
        "startend:end_checker",
    ]
    # employ ChatGPT to come up the keyword
    wGPT_list = [
        "keywords:existence",
        "keywords:frequency",
        "keywords:forbidden_words",
        "length_constraints:nth_paragraph_first_word",
    ]

    constraints = []
    kwargs = []
    flag = -1
    position = ""
    hard_instruction_id_list = []
    for i, inst_id in enumerate(instruction_id_list):
        if len(cons_dict[inst_id]["args"]) == 0:
            kwargs.append({})
            constraints.append(random.choice(cons_dict[inst_id]["description"]))
            hard_instruction_id_list.append(inst_id)
        elif inst_id in woGPT_list:
            if "number_words" in inst_id:
                description = random.choice(cons_dict[inst_id]["description"])
                relation = random.choice(cons_dict[inst_id]["args"]["relation"])
                ### 参考当前response的长度, 随机设置一个数值
                if relation == "at least":
                    least_num_words = (
                        len(seed_data["response"]) * random.uniform(0.5, 1.2) // 50 * 50
                    )
                    if least_num_words >= 50 and chance(
                        args.length_constraint_probability
                    ):
                        num_words = least_num_words
                    else:
                        # 一定概率按照当前response的长度
                        num_words = random.randint(25, 50)
                elif relation == "less than":
                    max_num_words = (
                        len(seed_data["response"]) * random.uniform(0.8, 1.5) // 50 * 50
                    )
                    if max_num_words >= 100 and chance(
                        args.length_constraint_probability
                    ):
                        num_words = max_num_words
                    else:
                        num_words = random.randint(30, 100)
                kwargs.append({"relation": relation, "num_words": num_words})
                description = description.replace("<relation>", relation)
                description = description.replace("<num_words>", str(num_words))
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
            elif "response_language" in inst_id:
                description = random.choice(cons_dict[inst_id]["description"])
                lang_code = random.choice(cons_dict[inst_id]["args"]["language"])
                lang_dict = instructions_util.LANGUAGE_CODES
                language = lang_dict[lang_code]
                kwargs.append({"language": lang_code})
                description = description.replace("<language>", language)
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
            else:
                description = random.choice(cons_dict[inst_id]["description"])
                args_keys = cons_dict[inst_id]["args"].keys()
                temp = {}
                for args_key in args_keys:
                    temp[args_key] = random.choice(cons_dict[inst_id]["args"][args_key])
                    description = description.replace(
                        "<" + args_key + ">", str(temp[args_key])
                    )
                kwargs.append(temp)
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
        elif inst_id in wGPT_list:
            if "existence" in inst_id:
                description = random.choice(cons_dict[inst_id]["description"])
                keywords = get_keyword(seed_inst)
                keywords = (
                    random.choice(keywords) if type(keywords) == list else keywords
                )
                keywords = keywords.split(" ")[0]
                kwargs.append({"keywords": [keywords]})
                description = description.replace("<keywords>", keywords)
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
            elif "frequency" in inst_id:
                description = random.choice(cons_dict[inst_id]["description"])
                keywords = get_keyword(seed_inst)
                keywords = random.choice(keywords)
                keywords = keywords.split(" ")[0]
                relation = random.choice(cons_dict[inst_id]["args"]["relation"])
                frequency = random.choice(cons_dict[inst_id]["args"]["frequency"])
                kwargs.append(
                    {"relation": relation, "keyword": keywords, "frequency": frequency}
                )
                description = description.replace("<relation>", str(relation))
                description = description.replace("<keyword>", str(keywords))
                description = description.replace("<frequency>", str(frequency))
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
            elif "forbidden_words" in inst_id:
                description = random.choice(cons_dict[inst_id]["description"])
                keywords = get_keyword(seed_inst)
                keywords = random.choice(keywords)
                keywords = keywords.split(" ")[0]
                kwargs.append({"forbidden_words": [keywords]})
                description = description.replace("<forbidden_words>", keywords)
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
            elif "nth_paragraph_first_word" in inst_id:
                description = random.choice(cons_dict[inst_id]["description"])
                keywords = get_keyword(seed_inst)
                keywords = (
                    random.choice(keywords) if type(keywords) == list else keywords
                )
                keywords = keywords.split(" ")[0]
                first_word = keywords
                num_paragraphs = random.choice(
                    cons_dict[inst_id]["args"]["num_paragraphs"]
                )
                nth_paragraph = random.randint(1, num_paragraphs)
                kwargs.append(
                    {
                        "first_word": first_word,
                        "num_paragraphs": num_paragraphs,
                        "nth_paragraph": nth_paragraph,
                    }
                )
                nth_paragraph = cons_dict[inst_id]["args"]["nth_paragraph"][
                    nth_paragraph
                ]
                description = description.replace("<first_word>", first_word)
                description = description.replace(
                    "<num_paragraphs>", str(num_paragraphs)
                )
                description = description.replace("<nth_paragraph>", str(nth_paragraph))
                constraints.append(description)
                hard_instruction_id_list.append(inst_id)
            else:
                logger.error("error in seed inst {}".format(s))
                exit(-1)
        elif "repeat_prompt" in inst_id:
            flag = i
            position = random.choice(cons_dict[inst_id]["args"]["position"])
            if position == "end":
                description = random.choice(cons_dict[inst_id]["description"][:4])
            else:
                description = random.choice(cons_dict[inst_id]["description"][4:])
            kwargs.append({"prompt_to_repeat": ""})
            constraints.append(description)
            hard_instruction_id_list.append(inst_id)
        else:
            logger.error("error {} not supported".format(seed_data["id"]))
            exit(-1)

    new_data = copy.deepcopy(seed_data)
    new_inst = seed_inst
    if flag == -1:
        for constraint in constraints:
            new_inst = new_inst + constraint + ". "

        new_data["prompt"] = new_inst
        new_data["instruction_id_list"] = hard_instruction_id_list
        new_data["kwargs"] = kwargs
        new_data["constraints"] = constraints
        new_data["prompt_wo_hard_constraints"] = seed_inst

    else:
        for ci, constraint in enumerate(constraints):
            if ci == flag:
                continue
            new_inst = new_inst + constraint + ". "
        kwargs[flag]["prompt_to_repeat"] = new_inst

        if position == "end":
            new_inst = new_inst + constraints[flag]
        else:
            new_inst = constraints[flag] + new_inst

        new_data["prompt"] = new_inst
        new_data["instruction_id_list"] = hard_instruction_id_list
        new_data["kwargs"] = kwargs
        new_data["constraints"] = constraints
        new_data["prompt_wo_hard_constraints"] = seed_inst

    return new_data


def evol_difficulty(args):
    ### evol 截断完毕, 下一阶段是进行hard constraint的选取
    seed_datas = readjsonl(args.save_data_path)

    # Clear the target file first by opening in write mode
    with open(args.llm_judge_as_reasonable_path, "w", encoding="utf-8") as f:
        pass
    output_reasonable = open(args.llm_judge_as_reasonable_path, "a", encoding="utf-8")

    with open(args.llm_judge_as_unreasonable_path, "w", encoding="utf-8") as f:
        pass
    output_unreasonable = open(
        args.llm_judge_as_unreasonable_path, "a", encoding="utf-8"
    )

    cons_dict = taxonomy.taxonomy
    conflict_dict = instructions_registry.INSTRUCTION_CONFLICTS
    conflict_dict = instructions_registry.conflict_make(conflict_dict)

    hard_keys = [key for key in cons_dict.keys()]

    input_list = []
    for seed_data in seed_datas:
        input_list.append((seed_data, args, hard_keys, cons_dict, conflict_dict))
    output_list = parallel_wrapper(
        input_list,
        process_single_seed_data,
        cache_dir="./",
        cache_file="cache_tmp",
        max_workers=args.max_workers,
        savebatch=10000,
    )

    judge_list = parallel_wrapper(
        preprocess_list(output_list),
        check_quality,
        cache_dir="./",
        cache_file="cache_tmp",
        max_workers=args.max_workers,
        savebatch=10000,
    )
    for output_data, judge_result in zip(output_list, judge_list):
        output_data["reasonable_ins"] = judge_result
        if judge_result:
            output_reasonable.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        else:
            output_unreasonable.write(
                json.dumps(output_data, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument(
        "--round_idx", type=int, default=0, help="Evolution round index"
    )
    parser.add_argument(
        "--easy_pool_path",
        type=str,
        default=os.path.join(PROJECT_DATA, "v5_difficulty/round_{n}/easy_pool.jsonl"),
        help="Path to easy pool",
    )

    ## save_data_path and evol_exception_path is the whole dataset
    parser.add_argument(
        "--save_data_path",
        type=str,
        default=os.path.join(PROJECT_DATA, "v6_evol/round_{n}/3.2_evolved_ins.jsonl"),
        help="Path to the output compositional data",
    )
    parser.add_argument(
        "--evol_exception_path",
        type=str,
        default=os.path.join(
            PROJECT_DATA, "v6_evol/round_{n}/3.2_evolved_ins_exception.jsonl"
        ),
        help="Path to the exception data",
    )

    ## split save_data into two parts:
    parser.add_argument(
        "--llm_judge_as_reasonable_path",
        type=str,
        default=os.path.join(
            PROJECT_DATA, "v6_evol/round_{n}/3.2_evolved_ins_reasonable.jsonl"
        ),
        help="Path to the output compositional data",
    )
    parser.add_argument(
        "--llm_judge_as_unreasonable_path",
        type=str,
        default=os.path.join(
            PROJECT_DATA, "v6_evol/round_{n}/3.2_evolved_ins_unreasonable.jsonl"
        ),
        help="Path to the exception data",
    )

    ## evol dict save path
    # parser.add_argument("--evol_dict_path",type=str,default= os.path.join(PROJECT_DATA, 'v6_evol/round_{n}/3.2_evolved_ins_dict.json'),help="Path to the evol dict")

    # parser.add_argument("--debug",type=int,default=1)
    parser.add_argument("--evol_temperature", type=float, default=0.7)
    parser.add_argument("--evol_max_tokens", type=int, default=3000)
    parser.add_argument("--url_level", type=int, default=1)
    parser.add_argument("--max_evol_retries", type=int, default=3)
    parser.add_argument("--length_constraint_probability", type=float, default=0.5)

    parser.add_argument("--num_evol_attempts", type=int, default=2)
    parser.add_argument("--num_hard_constraints_lower_bound", type=int, default=2)
    parser.add_argument("--num_hard_constraints_upper_bound", type=int, default=5)
    args = parser.parse_args()

    ## parse
    # args.easy_pool_path = args.easy_pool_path.format(n=args.round_idx)
    args.save_data_path = args.save_data_path.format(n=args.round_idx)
    args.evol_exception_path = args.evol_exception_path.format(n=args.round_idx)
    args.llm_judge_as_reasonable_path = args.llm_judge_as_reasonable_path.format(
        n=args.round_idx
    )
    args.llm_judge_as_unreasonable_path = args.llm_judge_as_unreasonable_path.format(
        n=args.round_idx
    )
    # args.evol_dict_path = args.evol_dict_path.format(n=args.round_idx)

    evol_diverse(args)
    evol_difficulty(args)

    # writejson(EVOL_DICT, args.evol_dict_path)
