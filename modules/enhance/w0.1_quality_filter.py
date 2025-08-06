# 导入必要的库
import argparse
import os

import torch
from langdetect import detect
from rich import print
from sentence_transformers import util
from tqdm import tqdm
from transformers import AutoTokenizer

from global_config import MODEL_7B_PATH, PROJECT_ROOT
from infer.apis import get_res_cycle
from infer.multiprocess_wrapper import (
    CheckpointManager,
    MultiProcessHandler,
    parallel_wrapper,
    preprocess_list,
)
from infer.parallel_embedding import parallel_SentenceTransformer
from logs.logger import logger
from modules.utils import (
    readjson,
    readjsonl,
    seed_everything,
    unified_judge_parse,
    writejson,
    writejsonl,
)

debug_value = os.environ.get("DEBUG", "false").lower() == "true"
logger.info(f"- Debug_value: {debug_value}")
api_key = "api-key"


# 检查可用的 GPU 数量
device_ids = list(range(torch.cuda.device_count()))


# 1. 检查是否是英文
def is_english(item):
    """检查文本是否为英文"""
    text = item["prompt"]
    try:
        return detect(text) == "en"
    except:
        return False


# 2. 长度过滤
def is_within_length(item, max_length=8192):
    """检查tokenized后的文本长度是否在限制范围内"""
    text = item["prompt"]
    try:
        # 初始化工具
        tokenizer = AutoTokenizer.from_pretrained(MODEL_7B_PATH)  # 用于长度过滤
        tokens = tokenizer.encode(text)
        return len(tokens) <= max_length
    except:
        return False


# 3. 数据去重
def remove_duplicates(input_list, threshold=0.9):
    """使用sentenceBERT去重，基于余弦相似度"""
    if len(input_list) <= 1:
        return input_list

    prompts = [item["prompt"] for item in input_list]
    # 获取嵌入向量
    embeddings = parallel_SentenceTransformer(prompts)
    embeddings = torch.tensor(embeddings).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 分批计算相似度矩阵以避免OOM
    num_samples = embeddings.shape[0]
    similarity_matrix = torch.zeros((num_samples, num_samples), device="cpu")

    # 使用更小的批次大小并在CPU上计算相似度
    batch_size = 1000  # 减小批次大小
    for i in tqdm(
        range(0, num_samples, batch_size),
        desc="Computing similarity matrix",
        mininterval=5.0,
        maxinterval=10.0,
        leave=True,
    ):
        batch_end = min(i + batch_size, num_samples)
        batch_embeddings = embeddings[i:batch_end].cpu()

        # 将计算移到CPU上
        for j in range(0, num_samples, batch_size):
            j_end = min(j + batch_size, num_samples)
            similarities = util.pytorch_cos_sim(
                batch_embeddings, embeddings[j:j_end].cpu()
            )
            similarity_matrix[i:batch_end, j:j_end] = similarities

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 标记需要移除的重复项
    to_remove = set()
    for i in tqdm(
        range(num_samples),
        desc="Deduplicating",
        mininterval=5.0,
        maxinterval=10.0,
        leave=True,
    ):
        if i not in to_remove:
            # 只需检查i之后的样本
            similarities = similarity_matrix[i, (i + 1) :]
            duplicates = torch.where(similarities > threshold)[0]
            # 将索引调整回原始位置
            to_remove.update(int(j + i + 1) for j in duplicates.numpy())

    # Print examples of duplicates being removed
    if debug_value:
        print("\n=== Deduplication Examples ===")

    for i in range(num_samples):
        if i in to_remove:
            # Find which sample caused this to be removed
            similarities = similarity_matrix[i]
            most_similar_idx = int(
                torch.argmax(similarities[:i])
            )  # Only look at previous samples
            similarity = similarities[most_similar_idx].item()

            if debug_value:
                print(f"\nRemoved as duplicate (similarity: {similarity:.3f}):")
                print(
                    f"Original ({most_similar_idx}): {input_list[most_similar_idx]['prompt'][:200]}"
                )
                print(f"Duplicate ({i}): {input_list[i]['prompt'][:200]}")

            # Only print first 5 examples to avoid flooding output
            if len([x for x in to_remove if x < i]) >= 5:
                remaining = len(to_remove) - 5
                print(f"\n totoally {len(to_remove)} duplicates")
                break

    # 返回去重后的列表
    return [input_list[i] for i in range(num_samples) if i not in to_remove]


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


# 4. 使用大模型检查指令质量 ## 性能瓶颈环节, 引入并行 warpper
def check_quality(item):
    """通过OpenAI API检查指令的语法、拼写、清晰度和相关性"""
    prompt = item["prompt"]
    try:

        # res = get_res_cycle( f"Evaluate the following instruction for clarity, relevance, grammatical correctness, and absence of spelling errors. Respond with 'pass' if it meets all criteria, otherwise 'fail':\n\n{prompt}",max_tokens=10,url_level=1 )
        query = quality_en_prompt.format(prompt=prompt)
        res = get_res_cycle(query, max_tokens=1024, url_level=args.url_level)

        if debug_value:
            print(f"- QUERY:\n{query}", flush=True)
            print(f"- RESPONSE:\n{res}", flush=True)

        return unified_judge_parse(res)
    except Exception as e:
        logger.error(f"Error in quality check: {e}")
        return False


# 整合过滤步骤
def filter_prompts(input_list, max_workers):
    """执行所有过滤步骤并统计结果"""
    stats = {
        "original": len(input_list),  # 原始指令数量
        "english_filtered": 0,  # 非英文过滤掉的数量
        "duplicates_filtered": 0,  # 重复过滤掉的数量
        "length_filtered": 0,  # 超长过滤掉的数量
        "passed": 0,  # 最终通过的数量
    }

    # Step 1: 过滤非英文指令
    judge_list = parallel_wrapper(
        preprocess_list(input_list), is_english, max_workers=max_workers
    )
    english_prompts = [p for p, flag in zip(input_list, judge_list) if flag]
    # english_prompts = [p for p in input_list if is_english(p)]
    stats["english_filtered"] = stats["original"] - len(english_prompts)
    logger.info(f"Step 1 --- english_filtered: {stats}")

    # Step 2: 去重
    unique_prompts = remove_duplicates(
        english_prompts, threshold=args.deduplication_threshold
    )
    stats["duplicates_filtered"] = len(english_prompts) - len(unique_prompts)
    logger.info(f"Step 2 --- duplicates_filtered: {stats}")

    # Step 3: 长度过滤
    judge_list = parallel_wrapper(
        preprocess_list(unique_prompts), is_within_length, max_workers=max_workers
    )
    within_length_prompts = [p for p, flag in zip(unique_prompts, judge_list) if flag]
    # within_length_prompts = [p for p in english_prompts if is_within_length(p)]
    stats["length_filtered"] = len(unique_prompts) - len(within_length_prompts)
    logger.info(f"Step 3 --- length_filtered: {stats}")

    # Write Intermediate results
    writejsonl(within_length_prompts, args.output_file)
    writejson(stats, args.output_stats_file)
    return within_length_prompts, stats


def test_remove_duplicates(args):
    input_list = readjsonl(args.input_file)
    filtered_prompts = remove_duplicates(input_list)
    writejsonl(filtered_prompts, args.output_file)


# 使用示例
if __name__ == "__main__":
    # 测试数据
    seed_everything(42)

    parser = argparse.ArgumentParser(description="tag difficulty")
    # general para
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data/v1_seed/manually_selected_data.jsonl"),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v1_seed/high_quality_data_smallset.jsonl"
        ),
    )
    parser.add_argument(
        "--output_stats_file",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "data/v1_seed/filter_stats_data_smallset.jsonl"
        ),
    )
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--url_level", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="./")
    parser.add_argument("--deduplication_threshold", type=float, default=0.9)
    # parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    print(args)

    input_list = readjsonl(args.input_file)

    # 执行过滤
    filtered_prompts, stats = filter_prompts(input_list, args.max_workers)
    print("\n### Statistics:")
    print(f"stats: {stats}")
